from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.unet_downscaler import UNetDownscaler
from training.train_downscaler import compute_input_stats, compute_training_loss, normalize_input_batch, residual_base, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether U-Net can memorize one ERA5->PRISM sample.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/training_sanity/single_sample")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-modes", nargs="+", choices=["direct", "residual"], default=["direct", "residual"])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--loss-mode", choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"], default="mse_l1")
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--stop-rmse", type=float, default=0.05)
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


def ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-8))


def save_panel(
    *,
    output_path: Path,
    era5_up: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    raw_pred: np.ndarray,
    residual_target: np.ndarray | None,
    title: str,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    panels: List[Tuple[str, np.ndarray, str]] = [
        ("ERA5 upsampled", era5_up, "coolwarm"),
        ("prediction", pred, "coolwarm"),
        ("PRISM target", target, "coolwarm"),
        ("|error|", np.abs(pred - target), "magma"),
    ]
    if residual_target is not None:
        panels.extend(
            [
                ("pred residual", raw_pred, "coolwarm"),
                ("target residual", residual_target, "coolwarm"),
            ]
        )
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    temp_arrays = [era5_up, pred, target]
    temp_vmin = float(min(np.min(a) for a in temp_arrays))
    temp_vmax = float(max(np.max(a) for a in temp_arrays))
    res_vmax = None
    if residual_target is not None:
        res_vmax = float(max(np.max(np.abs(raw_pred)), np.max(np.abs(residual_target))))
    for ax, (label, arr, cmap) in zip(np.ravel(axes), panels):
        if label == "|error|":
            im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=float(np.max(arr)))
        elif "residual" in label and res_vmax is not None:
            im = ax.imshow(arr, cmap=cmap, vmin=-res_vmax, vmax=res_vmax)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=temp_vmin, vmax=temp_vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_mode(args: argparse.Namespace, dataset: ERA5_PRISM_Dataset, target_mode: str, output_dir: Path) -> Dict[str, Any]:
    set_seed(int(args.seed))
    device = torch.device(args.device)
    sample_index = int(args.sample_index)
    x, y = dataset[sample_index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    input_mean_np, input_std_np = compute_input_stats(dataset, [sample_index])
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)
    model = UNetDownscaler(
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=int(y.shape[1]),
        base_channels=int(args.unet_base_channels),
        padding_mode="replicate",
        upsample_mode="bilinear",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    rows: List[Dict[str, Any]] = []
    final_pred = None
    final_raw = None
    base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
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
        optimizer.step()
        rmse = float(torch.sqrt(F.mse_loss(pred, y)).item())
        rows.append({"epoch": epoch, "loss": float(loss.item()), "rmse": rmse})
        final_pred = pred.detach()
        final_raw = raw.detach()
        if rmse <= float(args.stop_rmse):
            break

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
    metrics: Dict[str, Any] = {
        "target_mode": target_mode,
        "sample_index": sample_index,
        "epochs_run": int(rows[-1]["epoch"]),
        "final_loss": float(rows[-1]["loss"]),
        "final_rmse": float(np.sqrt(np.mean(err**2))),
        "final_mae": float(np.mean(np.abs(err))),
        "gradient_ratio": ratio(float(np.mean(pred_grad)), float(np.mean(target_grad))),
        "high_frequency_ratio": ratio(pred_hf, target_hf),
        "target_min": float(np.min(target_np)),
        "target_max": float(np.max(target_np)),
        "prediction_min": float(np.min(pred_np)),
        "prediction_max": float(np.max(pred_np)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_curve.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "loss", "rmse"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_panel(
        output_path=output_dir / "overfit_panel.png",
        era5_up=era5_np,
        pred=pred_np,
        target=target_np,
        raw_pred=raw_np,
        residual_target=residual_target_np,
        title=f"single-sample overfit: {target_mode}",
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
        history_length=3,
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
        "epochs_run",
        "final_loss",
        "final_rmse",
        "final_mae",
        "gradient_ratio",
        "high_frequency_ratio",
        "target_min",
        "target_max",
        "prediction_min",
        "prediction_max",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
