from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.unet_downscaler import UNetDownscaler
from training.train_downscaler import (
    compute_input_stats,
    compute_training_loss,
    normalize_input_batch,
    residual_base,
    set_seed,
    split_dataset,
)


GRAD_GROUPS = {
    "enc1": ("enc1.",),
    "enc2": ("enc2.",),
    "bottleneck": ("bottleneck.",),
    "dec2": ("dec2.",),
    "dec1": ("dec1.",),
    "high_res": ("high_res.",),
    "out": ("out.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether U-Net can memorize one ERA5->PRISM sample.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/training_sanity/single_sample")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-modes", nargs="+", choices=["direct", "residual"], default=["residual"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--padding-mode", choices=["reflection", "zero", "replicate"], default="reflection")
    parser.add_argument("--upsampling-mode", choices=["bilinear", "convtranspose", "pixelshuffle"], default="bilinear")
    parser.add_argument("--loss-mode", choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"], default="mse_l1")
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--normalization-source", choices=["full_train", "sample"], default="full_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--stop-rmse", type=float, default=0.05)
    parser.add_argument("--grad-log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(arr, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx**2 + gy**2)


def box_mean(arr: np.ndarray, window: int = 7) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    squeeze = False
    if data.ndim == 2:
        data = data[None, ...]
        squeeze = True
    pad = window // 2
    padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(data, dtype=np.float64)
    h, w = data.shape[-2:]
    for dy in range(window):
        for dx in range(window):
            out += padded[:, dy : dy + h, dx : dx + w]
    out /= float(window * window)
    return out[0] if squeeze else out


def high_pass(arr: np.ndarray, window: int = 7) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) - box_mean(arr, window)


def power_grid(field: np.ndarray, spacing_km: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    h, w = arr.shape[-2:]
    fy = np.fft.fftfreq(h, d=spacing_km)
    fx = np.fft.fftfreq(w, d=spacing_km)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    wavelength = np.divide(1.0, kr, out=np.full_like(kr, np.inf), where=kr > 0)
    power = np.abs(np.fft.fft2(arr)) ** 2
    power[kr == 0] = 0.0
    return wavelength, power


def band_power(field: np.ndarray, low_km: float, high_km: Optional[float], spacing_km: float = 4.0) -> float:
    wavelength, power = power_grid(field, spacing_km=spacing_km)
    if high_km is None:
        mask = wavelength >= low_km
    else:
        mask = (wavelength >= low_km) & (wavelength < high_km)
    mask &= np.isfinite(wavelength)
    return float(np.mean(power[mask])) if np.any(mask) else 0.0


def border_mask(shape: Tuple[int, int], fraction: float = 0.10) -> np.ndarray:
    h, w = shape
    width = max(1, int(round(min(h, w) * fraction)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-8))


def tensor_grad_norm(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None or tensor.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.grad.detach()).item())


def param_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    out: Dict[str, float] = {}
    named = list(model.named_parameters())
    for group, prefixes in GRAD_GROUPS.items():
        total = 0.0
        for name, param in named:
            if any(name.startswith(prefix) for prefix in prefixes) and param.grad is not None:
                total += float(torch.sum(param.grad.detach() ** 2).item())
        out[f"grad_{group}"] = float(np.sqrt(total))
    return out


def attach_skip_hooks(model: UNetDownscaler, store: Dict[str, torch.Tensor]) -> List[Any]:
    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if torch.is_tensor(output) and output.requires_grad:
                output.retain_grad()
                store[name] = output

        return hook

    return [
        model.enc1.register_forward_hook(capture("skip_enc1")),
        model.enc2.register_forward_hook(capture("skip_enc2")),
    ]


def save_panel(
    *,
    output_path: Path,
    era5_up: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    raw_pred: np.ndarray,
    residual_target: Optional[np.ndarray],
    title: str,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    panels: List[Tuple[str, np.ndarray, str]] = [
        ("PRISM target", target, "coolwarm"),
        ("U-Net prediction", pred, "coolwarm"),
        ("ERA5 bilinear", era5_up, "coolwarm"),
    ]
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    temp_arrays = [era5_up, pred, target]
    temp_vmin = float(min(np.min(a) for a in temp_arrays))
    temp_vmax = float(max(np.max(a) for a in temp_arrays))
    for ax, (label, arr, cmap) in zip(np.ravel(axes), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=temp_vmin, vmax=temp_vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_log_curve(rows: Sequence[Dict[str, Any]], output_path: Path, title: str) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    rmses = [float(row["rmse"]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, losses, label="loss")
    ax.plot(epochs, rmses, label="same-sample RMSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log scale")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def normalization_indices(args: argparse.Namespace, dataset: ERA5_PRISM_Dataset, sample_index: int) -> List[int]:
    if args.normalization_source == "sample":
        return [int(sample_index)]
    _, _, train_indices, _ = split_dataset(dataset, float(args.val_fraction), int(args.split_seed))
    return [int(i) for i in train_indices]


def run_mode(args: argparse.Namespace, dataset: ERA5_PRISM_Dataset, target_mode: str, output_dir: Path) -> Dict[str, Any]:
    set_seed(int(args.seed))
    device = torch.device(args.device)
    sample_index = int(args.sample_index)
    x, y = dataset[sample_index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    norm_indices = normalization_indices(args, dataset, sample_index)
    input_mean_np, input_std_np = compute_input_stats(dataset, norm_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)
    model = UNetDownscaler(
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=int(y.shape[1]),
        base_channels=int(args.unet_base_channels),
        padding_mode=str(args.padding_mode),
        upsample_mode=str(args.upsampling_mode),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    rows: List[Dict[str, Any]] = []
    final_pred = None
    final_raw = None
    base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
    residual_context = (y - base).squeeze().detach().cpu().numpy()
    print(
        "residual_target "
        f"mean={float(np.mean(residual_context)):.4f}C "
        f"std={float(np.std(residual_context)):.4f}C "
        f"min={float(np.min(residual_context)):.4f}C "
        f"max={float(np.max(residual_context)):.4f}C"
    )
    activation_store: Dict[str, torch.Tensor] = {}
    hooks = attach_skip_hooks(model, activation_store)
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            activation_store.clear()
            x_model = normalize_input_batch(x, input_mean, input_std)
            raw = model(x_model, target_size=(y.shape[-2], y.shape[-1]))
            if target_mode == "residual":
                loss_target = y - base
                pred = base + raw
            else:
                loss_target = y
                pred = raw
            loss, _ = compute_training_loss(
                raw_preds=raw,
                loss_target=loss_target,
                final_preds=pred,
                y=y,
                loss_mode=str(args.loss_mode),
                l1_weight=float(args.l1_weight),
                grad_weight=float(args.grad_weight),
            )
            optimizer.zero_grad()
            loss.backward()
            grad_row = param_grad_norms(model)
            grad_row["grad_skip_enc1_activation"] = tensor_grad_norm(activation_store.get("skip_enc1"))
            grad_row["grad_skip_enc2_activation"] = tensor_grad_norm(activation_store.get("skip_enc2"))
            optimizer.step()

            rmse = float(torch.sqrt(F.mse_loss(pred, y)).item())
            row = {"epoch": epoch, "loss": float(loss.item()), "rmse": rmse}
            row.update(grad_row)
            rows.append(row)
            final_pred = pred.detach()
            final_raw = raw.detach()
            if epoch % max(1, int(args.grad_log_interval)) == 0 or epoch == 1:
                print(
                    f"{target_mode} epoch={epoch:04d} "
                    f"loss={float(loss.item()):.6f} rmse={rmse:.5f} "
                    f"enc1={grad_row['grad_enc1']:.4g} "
                    f"enc2={grad_row['grad_enc2']:.4g} "
                    f"bottleneck={grad_row['grad_bottleneck']:.4g} "
                    f"decoder1={grad_row['grad_dec1']:.4g} "
                    f"decoder2={grad_row['grad_dec2']:.4g}"
                )
            if rmse <= float(args.stop_rmse):
                break
    finally:
        for hook in hooks:
            hook.remove()

    assert final_pred is not None and final_raw is not None
    target_np = y.squeeze().detach().cpu().numpy()
    pred_np = final_pred.squeeze().detach().cpu().numpy()
    raw_np = final_raw.squeeze().detach().cpu().numpy()
    era5_np = base.squeeze().detach().cpu().numpy()
    residual_target_np = (y - base).squeeze().detach().cpu().numpy() if target_mode == "residual" else None
    err = pred_np - target_np
    pred_grad = gradient_magnitude(pred_np)
    target_grad = gradient_magnitude(target_np)
    pred_hf = float(np.mean(high_pass(pred_np) ** 2))
    target_hf = float(np.mean(high_pass(target_np) ** 2))
    pred_4_8 = band_power(pred_np, 4.0, 8.0)
    target_4_8 = band_power(target_np, 4.0, 8.0)
    mask2d = border_mask(target_np.shape)
    residual_target_full = (y - base).squeeze().detach().cpu().numpy()
    rmse_by_epoch = {int(row["epoch"]): float(row["rmse"]) for row in rows}
    metrics: Dict[str, Any] = {
        "target_mode": target_mode,
        "sample_index": sample_index,
        "normalization_source": str(args.normalization_source),
        "normalization_n_indices": int(len(norm_indices)),
        "padding_mode": str(args.padding_mode),
        "upsampling_mode": str(args.upsampling_mode),
        "epochs_run": int(rows[-1]["epoch"]),
        "rmse_epoch_100": rmse_by_epoch.get(100),
        "rmse_epoch_500": rmse_by_epoch.get(500),
        "rmse_epoch_1000": rmse_by_epoch.get(1000),
        "final_loss": float(rows[-1]["loss"]),
        "final_rmse": float(np.sqrt(np.mean(err**2))),
        "final_mae": float(np.mean(np.abs(err))),
        "border_rmse": float(np.sqrt(np.mean(err[mask2d] ** 2))),
        "interior_rmse": float(np.sqrt(np.mean(err[~mask2d] ** 2))),
        "border_interior_ratio": float(np.sqrt(np.mean(err[mask2d] ** 2)) / max(float(np.sqrt(np.mean(err[~mask2d] ** 2))), 1e-8)),
        "gradient_ratio": ratio(float(np.mean(pred_grad)), float(np.mean(target_grad))),
        "high_frequency_ratio": ratio(pred_hf, target_hf),
        "retention_4_8km": ratio(pred_4_8, target_4_8),
        "residual_target_mean": float(np.mean(residual_target_full)),
        "residual_target_std": float(np.std(residual_target_full)),
        "target_min": float(np.min(target_np)),
        "target_max": float(np.max(target_np)),
        "prediction_min": float(np.min(pred_np)),
        "prediction_max": float(np.max(pred_np)),
        "batch_norm_layers": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (output_dir / "training_curve.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.save(output_dir / "target.npy", target_np)
    np.save(output_dir / "prediction.npy", pred_np)
    np.save(output_dir / "raw_prediction.npy", raw_np)
    np.save(output_dir / "era5_bilinear.npy", era5_np)
    save_panel(
        output_path=output_dir / "three_panel.png",
        era5_up=era5_np,
        pred=pred_np,
        target=target_np,
        raw_pred=raw_np,
        residual_target=residual_target_np,
        title=f"single-sample overfit: {target_mode} | {args.upsampling_mode}",
    )
    save_log_curve(rows, output_dir / "loss_curve_log.png", f"single-sample overfit: {target_mode} | {args.upsampling_mode}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": int(x.shape[1] * x.shape[2]),
                "out_channels": int(y.shape[1]),
                "base_channels": int(args.unet_base_channels),
                "padding_mode": str(args.padding_mode),
                "upsample_mode": str(args.upsampling_mode),
            },
            "target_mode": target_mode,
            "history_length": int(args.history_length),
            "input_norm": {"mean": input_mean_np.tolist(), "std": input_std_np.tolist()},
            "sample_index": sample_index,
            "args": vars(args),
        },
        output_dir / "model.pt",
    )
    return metrics


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    output_dir = repo_path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set="core4_topo",
        static_covariate_path=str(repo_path(args.static_covariate_path)),
        verbose=False,
    )
    rows = []
    for mode in args.target_modes:
        metrics = run_mode(args, dataset, mode, output_dir / mode)
        rows.append(metrics)
        print(
            f"{mode:>8} | rmse={metrics['final_rmse']:.5f} "
            f"epochs={metrics['epochs_run']} grad_ratio={metrics['gradient_ratio']:.3f} "
            f"hf_ratio={metrics['high_frequency_ratio']:.3f}"
        )
    fields = [
        "target_mode",
        "sample_index",
        "normalization_source",
        "normalization_n_indices",
        "padding_mode",
        "upsampling_mode",
        "epochs_run",
        "rmse_epoch_100",
        "rmse_epoch_500",
        "rmse_epoch_1000",
        "final_loss",
        "final_rmse",
        "final_mae",
        "border_rmse",
        "interior_rmse",
        "border_interior_ratio",
        "gradient_ratio",
        "high_frequency_ratio",
        "retention_4_8km",
        "residual_target_mean",
        "residual_target_std",
        "target_min",
        "target_max",
        "prediction_min",
        "prediction_max",
        "batch_norm_layers",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
