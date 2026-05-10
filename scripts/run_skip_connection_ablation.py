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


class NoSkipUNetAblation(UNetDownscaler):
    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self._prepare_input(x)
        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, stride=2))
        b = self.bottleneck(F.avg_pool2d(e2, kernel_size=2, stride=2))

        d2 = self._upsample(b, size=e2.shape[-2:], layer=getattr(self, "up2", None))
        d2 = self.dec2(torch.cat([d2, torch.zeros_like(e2)], dim=1))

        d1 = self._upsample(d2, size=e1.shape[-2:], layer=getattr(self, "up1", None))
        d1 = self.dec1(torch.cat([d1, torch.zeros_like(e1)], dim=1))

        high = self._upsample(d1, size=target_size, layer=getattr(self, "up_high", None))
        return self.out(self.high_res(high))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-sample skip vs no-skip U-Net ablation.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/training_pipeline_diagnosis/skip_ablation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--loss-mode", choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"], default="mse_l1")
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--normalization-source", choices=["full_train", "sample"], default="full_train")
    parser.add_argument("--border-fraction", type=float, default=0.10)
    parser.add_argument("--grid-spacing-km", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
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


def high_pass(arr: np.ndarray, window: int = 7) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 2:
        data = data[None, ...]
    pad = window // 2
    padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(data, dtype=np.float64)
    h, w = data.shape[-2:]
    for dy in range(window):
        for dx in range(window):
            out += padded[:, dy : dy + h, dx : dx + w]
    out /= float(window * window)
    detail = data - out
    return detail[0] if arr.ndim == 2 else detail


def radial_psd(field: np.ndarray, spacing_km: float) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=np.float64)
    h, w = arr.shape[-2:]
    fy = np.fft.fftfreq(h, d=spacing_km)
    fx = np.fft.fftfreq(w, d=spacing_km)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    bins = np.linspace(0.0, float(kr.max()), min(h, w) // 2 + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    power = np.abs(np.fft.fft2(arr - float(np.mean(arr)))) ** 2
    which = np.digitize(kr.reshape(-1), bins) - 1
    psd = []
    for idx in range(len(centers)):
        mask = which == idx
        psd.append(float(np.mean(power.reshape(-1)[mask])) if np.any(mask) else 0.0)
    return centers, np.asarray(psd, dtype=np.float64)


def band_power(field: np.ndarray, spacing_km: float, min_wavelength_km: float = 8.0, max_wavelength_km: float = 32.0) -> float:
    freqs, psd = radial_psd(field, spacing_km)
    wavelengths = 1.0 / np.maximum(freqs, 1e-8)
    mask = (wavelengths >= min_wavelength_km) & (wavelengths < max_wavelength_km)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


def border_mask(shape: Tuple[int, int], fraction: float) -> np.ndarray:
    h, w = shape
    width = max(1, int(round(min(h, w) * fraction)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def param_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    groups = {
        "enc1": ("enc1.",),
        "enc2": ("enc2.",),
        "bottleneck": ("bottleneck.",),
        "decoder1": ("dec1.",),
        "decoder2": ("dec2.",),
    }
    out: Dict[str, float] = {}
    named = list(model.named_parameters())
    for group, prefixes in groups.items():
        total = 0.0
        for name, param in named:
            if any(name.startswith(prefix) for prefix in prefixes) and param.grad is not None:
                total += float(torch.sum(param.grad.detach() ** 2).item())
        out[f"grad_{group}"] = float(np.sqrt(total))
    return out


def normalization_indices(args: argparse.Namespace, dataset: ERA5_PRISM_Dataset) -> List[int]:
    if args.normalization_source == "sample":
        return [int(args.sample_index)]
    _, _, train_indices, _ = split_dataset(dataset, float(args.val_fraction), int(args.split_seed))
    return [int(i) for i in train_indices]


def make_model(name: str, in_channels: int, out_channels: int, base_channels: int) -> UNetDownscaler:
    cls = UNetDownscaler if name == "skip_unet" else NoSkipUNetAblation
    return cls(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        padding_mode="reflection",
        upsample_mode="bilinear",
    )


def train_one(
    *,
    model_name: str,
    args: argparse.Namespace,
    dataset: ERA5_PRISM_Dataset,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    set_seed(int(args.seed))
    x, y = dataset[int(args.sample_index)]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    model = make_model(
        model_name,
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=int(y.shape[1]),
        base_channels=int(args.unet_base_channels),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
    rows: List[Dict[str, Any]] = []
    final_pred = None
    final_raw = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        raw = model(normalize_input_batch(x, input_mean, input_std), target_size=(y.shape[-2], y.shape[-1]))
        pred = base + raw
        loss, _ = compute_training_loss(
            raw_preds=raw,
            loss_target=y - base,
            final_preds=pred,
            y=y,
            loss_mode=str(args.loss_mode),
            l1_weight=float(args.l1_weight),
            grad_weight=float(args.grad_weight),
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norms = param_grad_norms(model)
        optimizer.step()
        rmse = float(torch.sqrt(F.mse_loss(pred, y)).item())
        row = {"epoch": epoch, "loss": float(loss.item()), "rmse": rmse}
        row.update(grad_norms)
        rows.append(row)
        final_pred = pred.detach()
        final_raw = raw.detach()

    assert final_pred is not None and final_raw is not None
    pred_np = final_pred.squeeze().cpu().numpy()
    target_np = y.squeeze().detach().cpu().numpy()
    era5_np = base.squeeze().detach().cpu().numpy()
    err = pred_np - target_np
    mask2d = border_mask(target_np.shape, float(args.border_fraction))
    target_hf = band_power(target_np, float(args.grid_spacing_km))
    pred_hf = band_power(pred_np, float(args.grid_spacing_km))
    metrics = {
        "model": model_name,
        "epochs": int(args.epochs),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "border_rmse": float(np.sqrt(np.mean(err[mask2d] ** 2))),
        "interior_rmse": float(np.sqrt(np.mean(err[~mask2d] ** 2))),
        "border_interior_ratio": float(np.sqrt(np.mean(err[mask2d] ** 2)) / max(float(np.sqrt(np.mean(err[~mask2d] ** 2))), 1e-8)),
        "high_frequency_power": float(pred_hf),
        "target_high_frequency_power": float(target_hf),
        "high_frequency_retention": float(pred_hf / max(target_hf, 1e-8)),
        "final_loss": float(rows[-1]["loss"]),
    }
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "training_curve.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.save(model_dir / "prediction.npy", pred_np)
    save_panel(model_dir / "three_panel.png", target_np, pred_np, era5_np, f"{model_name} residual overfit")
    return metrics


def save_panel(path: Path, target: np.ndarray, pred: np.ndarray, era5: np.ndarray, title: str) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    vmin = float(min(np.min(target), np.min(pred), np.min(era5)))
    vmax = float(max(np.max(target), np.max(pred), np.max(era5)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, label, arr in zip(axes, ["PRISM target", "U-Net prediction", "ERA5 bilinear"], [target, pred, era5]):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_psd_plot(path: Path, target: np.ndarray, metrics_by_model: Dict[str, np.ndarray], spacing_km: float) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, field in [("PRISM target", target), *metrics_by_model.items()]:
        freqs, psd = radial_psd(field, spacing_km)
        wavelengths = 1.0 / np.maximum(freqs, 1e-8)
        ax.plot(wavelengths, psd, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Radially averaged PSD")
    ax.set_title("Skip ablation PSD")
    ax.grid(alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_panel(path: Path, target: np.ndarray, era5: np.ndarray, predictions: Dict[str, np.ndarray]) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    arrays = [target, era5] + [predictions[name] for name in sorted(predictions)]
    vmin = float(min(np.min(arr) for arr in arrays))
    vmax = float(max(np.max(arr) for arr in arrays))
    labels = ["PRISM target", "ERA5 bilinear"] + sorted(predictions)
    panels = [target, era5] + [predictions[name] for name in sorted(predictions)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    for ax, label, arr in zip(np.ravel(axes), labels, panels):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle("Skip connection ablation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    output_dir = repo_path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set="core4_topo",
        static_covariate_path=str(repo_path(args.static_covariate_path)),
        verbose=False,
    )
    norm_indices = normalization_indices(args, dataset)
    input_mean_np, input_std_np = compute_input_stats(dataset, norm_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)

    rows = []
    predictions: Dict[str, np.ndarray] = {}
    x0, y0 = dataset[int(args.sample_index)]
    target_np = y0.squeeze().numpy().astype(np.float64)
    era5_np = residual_base(x0.unsqueeze(0), target_size=(y0.shape[-2], y0.shape[-1])).squeeze().numpy().astype(np.float64)
    for model_name in ("skip_unet", "no_skip_unet"):
        metrics = train_one(
            model_name=model_name,
            args=args,
            dataset=dataset,
            input_mean=input_mean,
            input_std=input_std,
            device=device,
            output_dir=output_dir,
        )
        pred_path = output_dir / model_name / "prediction.npy"
        # Save the final prediction by reading metrics panel data indirectly is impossible, so rerun a cheap eval below.
        predictions[model_name] = np.load(pred_path) if pred_path.exists() else np.asarray([])
        rows.append(metrics)
        print(
            f"{model_name}: rmse={metrics['rmse']:.4f} "
            f"hf_retention={metrics['high_frequency_retention']:.4f} "
            f"border_rmse={metrics['border_rmse']:.4f}"
        )

    # The predictions are written by loading from model-specific files after final eval.
    predictions = {}
    for model_name in ("skip_unet", "no_skip_unet"):
        pred_path = output_dir / model_name / "prediction.npy"
        if not pred_path.exists():
            continue
        predictions[model_name] = np.load(pred_path)
    if predictions:
        save_psd_plot(output_dir / "skip_ablation_psd.png", target_np, predictions, float(args.grid_spacing_km))
        save_comparison_panel(output_dir / "skip_ablation_comparison.png", target_np, era5_np, predictions)

    fields = ["model", "epochs", "rmse", "mae", "border_rmse", "interior_rmse", "border_interior_ratio", "high_frequency_retention", "final_loss"]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
