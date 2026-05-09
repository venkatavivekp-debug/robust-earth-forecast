from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether U-Net can overfit small ERA5->PRISM subsets.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/training_sanity/small_subset")
    parser.add_argument("--subset-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--loss-mode", choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"], default="mse_l1")
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
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


def save_panel(output_path: Path, pred: np.ndarray, target: np.ndarray, title: str) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    error = np.abs(pred - target)
    vmin = float(min(np.min(pred), np.min(target)))
    vmax = float(max(np.max(pred), np.max(target)))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    panels = [
        ("prediction", pred, "coolwarm", vmin, vmax),
        ("target", target, "coolwarm", vmin, vmax),
        ("|error|", error, "magma", 0.0, float(np.max(error))),
        ("target gradient", gradient_magnitude(target), "viridis", 0.0, float(np.max(gradient_magnitude(target)))),
    ]
    for ax, (label, arr, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_model(x: torch.Tensor, y: torch.Tensor, base_channels: int, device: torch.device) -> UNetDownscaler:
    return UNetDownscaler(
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=int(y.shape[1]),
        base_channels=int(base_channels),
        padding_mode="replicate",
        upsample_mode="bilinear",
    ).to(device)


def run_batch(
    *,
    model: UNetDownscaler,
    batch: Tuple[torch.Tensor, torch.Tensor],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    loss_mode: str,
    l1_weight: float,
    grad_weight: float,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
    raw = model(normalize_input_batch(x, input_mean, input_std), target_size=(y.shape[-2], y.shape[-1]))
    pred = base + raw
    loss_target = y - base
    loss, _ = compute_training_loss(
        raw_preds=raw,
        loss_target=loss_target,
        final_preds=pred,
        y=y,
        loss_mode=loss_mode,
        l1_weight=l1_weight,
        grad_weight=grad_weight,
    )
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    rmse = float(torch.sqrt(F.mse_loss(pred, y)).item())
    return float(loss.item()), rmse


def evaluate(
    *,
    model: UNetDownscaler,
    loader: DataLoader,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
            raw = model(normalize_input_batch(x, input_mean, input_std), target_size=(y.shape[-2], y.shape[-1]))
            pred = base + raw
            preds.append(pred.detach().cpu().numpy()[:, 0])
            targets.append(y.detach().cpu().numpy()[:, 0])
    pred_cube = np.concatenate(preds, axis=0)
    target_cube = np.concatenate(targets, axis=0)
    err = pred_cube - target_cube
    pred_grad = gradient_magnitude(pred_cube)
    target_grad = gradient_magnitude(target_cube)
    pred_hf = float(np.mean(high_pass(pred_cube) ** 2))
    target_hf = float(np.mean(high_pass(target_cube) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "gradient_ratio": float(np.mean(pred_grad) / max(float(np.mean(target_grad)), 1e-8)),
        "high_frequency_ratio": float(pred_hf / max(target_hf, 1e-8)),
        "first_prediction": pred_cube[0],
        "first_target": target_cube[0],
    }


def run_subset(args: argparse.Namespace, dataset: ERA5_PRISM_Dataset, subset_size: int, output_dir: Path) -> Dict[str, Any]:
    set_seed(int(args.seed))
    device = torch.device(args.device)
    train_set, val_set, train_indices, val_indices = split_dataset(dataset, float(args.val_fraction), int(args.split_seed))
    subset_indices = train_indices[: int(subset_size)]
    train_loader = DataLoader(Subset(dataset, subset_indices), batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    input_mean_np, input_std_np = compute_input_stats(dataset, subset_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)
    sample_x, sample_y = dataset[subset_indices[0]]
    model = make_model(sample_x.unsqueeze(0), sample_y.unsqueeze(0), int(args.unet_base_channels), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batch_losses: List[float] = []
        batch_rmses: List[float] = []
        for batch in train_loader:
            loss, rmse = run_batch(
                model=model,
                batch=batch,
                input_mean=input_mean,
                input_std=input_std,
                loss_mode=str(args.loss_mode),
                l1_weight=float(args.l1_weight),
                grad_weight=float(args.grad_weight),
                optimizer=optimizer,
                device=device,
            )
            batch_losses.append(loss)
            batch_rmses.append(rmse)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            train_eval = evaluate(model=model, loader=train_loader, input_mean=input_mean, input_std=input_std, device=device)
            val_eval = evaluate(model=model, loader=val_loader, input_mean=input_mean, input_std=input_std, device=device)
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(batch_losses)),
                    "train_batch_rmse": float(np.mean(batch_rmses)),
                    "train_rmse": train_eval["rmse"],
                    "val_rmse": val_eval["rmse"],
                }
            )

    train_eval = evaluate(model=model, loader=train_loader, input_mean=input_mean, input_std=input_std, device=device)
    val_eval = evaluate(model=model, loader=val_loader, input_mean=input_mean, input_std=input_std, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_curve.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "train_batch_rmse", "train_rmse", "val_rmse"])
        writer.writeheader()
        writer.writerows(rows)
    save_panel(output_dir / "train_panel.png", train_eval["first_prediction"], train_eval["first_target"], f"train subset {subset_size}")
    save_panel(output_dir / "val_panel.png", val_eval["first_prediction"], val_eval["first_target"], f"validation subset {subset_size}")
    metrics = {
        "subset_size": int(subset_size),
        "train_indices": [int(i) for i in subset_indices],
        "val_indices": [int(i) for i in val_indices],
        "epochs": int(args.epochs),
        "train_rmse": float(train_eval["rmse"]),
        "train_mae": float(train_eval["mae"]),
        "train_gradient_ratio": float(train_eval["gradient_ratio"]),
        "train_high_frequency_ratio": float(train_eval["high_frequency_ratio"]),
        "val_rmse": float(val_eval["rmse"]),
        "val_mae": float(val_eval["mae"]),
        "val_gradient_ratio": float(val_eval["gradient_ratio"]),
        "val_high_frequency_ratio": float(val_eval["high_frequency_ratio"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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
    for subset_size in args.subset_sizes:
        metrics = run_subset(args, dataset, int(subset_size), output_dir / f"subset_{subset_size}")
        rows.append(metrics)
        print(
            f"subset={subset_size:>2} | train_rmse={metrics['train_rmse']:.4f} "
            f"val_rmse={metrics['val_rmse']:.4f} train_hf={metrics['train_high_frequency_ratio']:.3f}"
        )
    fields = [
        "subset_size",
        "epochs",
        "train_rmse",
        "train_mae",
        "train_gradient_ratio",
        "train_high_frequency_ratio",
        "val_rmse",
        "val_mae",
        "val_gradient_ratio",
        "val_high_frequency_ratio",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
