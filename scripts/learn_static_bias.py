from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    normalize_input_batch,
    residual_base,
    set_seed,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net on the static temporal-mean PRISM minus ERA5 residual map."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--input-set", default="core4_topo", choices=["core4_elev", "core4_topo"])
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--padding-mode", choices=["reflection", "zero", "replicate"], default="reflection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", default="results/static_bias_learning")
    parser.add_argument("--image-out", default="docs/images/static_bias_learning_result.png")
    parser.add_argument("--doc-out", default="docs/research/static_bias_learning_findings.md")
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


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return 0.0
    r = float(np.corrcoef(aa, bb)[0, 1])
    return r if np.isfinite(r) else 0.0


def high_pass(arr: np.ndarray, window: int = 7) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    pad = window // 2
    padded = np.pad(data, ((pad, pad), (pad, pad)), mode="reflect")
    smooth = np.zeros_like(data, dtype=np.float64)
    h, w = data.shape
    for dy in range(window):
        for dx in range(window):
            smooth += padded[dy : dy + h, dx : dx + w]
    smooth /= float(window * window)
    return data - smooth


def compute_static_bias(dataset: ERA5_PRISM_Dataset, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    residuals: List[torch.Tensor] = []
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[int(idx)]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
            residuals.append((y - base).detach().cpu())
    return torch.mean(torch.cat(residuals, dim=0), dim=0, keepdim=True).to(device)


def save_training_curve(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    rmses = [float(row["rmse"]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, losses, label="MSE loss")
    ax.plot(epochs, rmses, label="static-bias RMSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log scale")
    ax.set_title("Static bias learning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_static_bias_panel(
    *,
    target: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    error = pred - target
    main_v = float(max(np.max(np.abs(target)), np.max(np.abs(pred)), 1e-6))
    err_v = float(max(np.max(np.abs(error)), 1e-6))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(target, cmap="coolwarm", vmin=-main_v, vmax=main_v)
    axes[0].set_title("Actual mean residual")
    axes[0].axis("off")
    im1 = axes[1].imshow(pred, cmap="coolwarm", vmin=-main_v, vmax=main_v)
    axes[1].set_title("Predicted mean residual")
    axes[1].axis("off")
    im2 = axes[2].imshow(error, cmap="coolwarm", vmin=-err_v, vmax=err_v)
    axes[2].set_title("Prediction error")
    axes[2].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.78, label="deg C")
    fig.colorbar(im1, ax=axes[1], shrink=0.78, label="deg C")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, label="deg C")
    fig.suptitle("Static mean residual learning")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(
    *,
    output_path: Path,
    summary: Dict[str, Any],
    image_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learned = bool(summary["spatial_pattern_learned"])
    read = (
        "The model can learn a stable residual map once day-to-day variability is removed."
        if learned
        else "The model did not recover the static residual map cleanly in this run."
    )
    output_path.write_text(
        "\n".join(
            [
                "# Static Bias Learning Findings",
                "",
                "This test removes temporal variability from the residual target. The model sees",
                "one ERA5/topography input and learns the training-split temporal mean of",
                "`PRISM - ERA5_bilinear`.",
                "",
                f"- Epochs: `{summary['epochs']}`",
                f"- Final RMSE: `{summary['rmse']:.4f}` deg C",
                f"- Final MAE: `{summary['mae']:.4f}` deg C",
                f"- Pixelwise correlation: `{summary['pearson_r']:.4f}`",
                f"- Target std: `{summary['target_std']:.4f}` deg C",
                f"- Predicted std: `{summary['prediction_std']:.4f}` deg C",
                f"- High-frequency retention: `{summary['high_frequency_retention']:.4f}`",
                "",
                read,
                "",
                f"Panel: `{image_path}`.",
                "",
                "If this test succeeds while daily residual prediction remains smooth or noisy,",
                "the failure mode is more consistent with data/target variability than a broken decoder.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    set_seed(int(args.seed))
    device = torch.device(args.device)

    static_path = repo_path(args.static_covariate_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        static_covariate_path=str(static_path),
        verbose=False,
    )
    _train_set, _val_set, train_indices, _val_indices = split_dataset(
        dataset, float(args.val_fraction), int(args.split_seed)
    )
    sample_index = int(args.sample_index)
    x, y = dataset[sample_index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    target = compute_static_bias(dataset, train_indices, device)

    input_mean_np, input_std_np = compute_input_stats(dataset, train_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)

    model = UNetDownscaler(
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=1,
        base_channels=int(args.unet_base_channels),
        padding_mode=str(args.padding_mode),
        upsample_mode="bilinear",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    rows: List[Dict[str, Any]] = []
    pred = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        raw = model(normalize_input_batch(x, input_mean, input_std), target_size=(target.shape[-2], target.shape[-1]))
        loss = F.mse_loss(raw, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            rmse = float(torch.sqrt(F.mse_loss(raw, target)).item())
            mae = float(torch.mean(torch.abs(raw - target)).item())
        rows.append({"epoch": epoch, "loss": float(loss.item()), "rmse": rmse, "mae": mae})
        pred = raw.detach()
        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch={epoch:04d} loss={float(loss.item()):.6f} rmse={rmse:.4f} mae={mae:.4f}")

    assert pred is not None
    target_np = target.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()
    err = pred_np - target_np
    target_hf = float(np.mean(high_pass(target_np) ** 2))
    pred_hf = float(np.mean(high_pass(pred_np) ** 2))
    summary = {
        "dataset_version": str(args.dataset_version),
        "input_set": str(args.input_set),
        "sample_index": sample_index,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "padding_mode": str(args.padding_mode),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson_r": pearson_r(pred_np, target_np),
        "target_mean": float(np.mean(target_np)),
        "target_std": float(np.std(target_np)),
        "prediction_mean": float(np.mean(pred_np)),
        "prediction_std": float(np.std(pred_np)),
        "high_frequency_retention": float(pred_hf / max(target_hf, 1e-8)),
        "spatial_pattern_learned": bool(pearson_r(pred_np, target_np) > 0.5),
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_curve.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "static_bias_target.npy", target_np)
    np.save(output_dir / "static_bias_prediction.npy", pred_np)
    save_training_curve(rows, output_dir / "loss_curve.png")

    image_path = repo_path(args.image_out)
    save_static_bias_panel(target=target_np, pred=pred_np, output_path=image_path)
    write_doc(
        output_path=repo_path(args.doc_out),
        summary=summary,
        image_path=image_path.relative_to(PROJECT_ROOT),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
