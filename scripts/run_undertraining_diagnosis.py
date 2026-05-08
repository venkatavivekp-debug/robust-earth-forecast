from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-budget U-Net training diagnostics for ERA5->PRISM undertraining checks."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/undertraining_diagnosis")
    parser.add_argument("--budgets", nargs="+", type=int, default=[80, 160, 300])
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-mode", choices=["direct"], default="direct")
    parser.add_argument("--padding-mode", choices=["reflection", "zero", "replicate"], default="replicate")
    parser.add_argument("--upsampling-mode", choices=["bilinear", "convtranspose"], default="bilinear")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates every validation sample")
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--center-crop-fraction", type=float, default=0.5)
    parser.add_argument("--skip-training", action="store_true", help="Reuse budget checkpoints already in --output-dir")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output-dir before running")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cmd(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 6:
            print(f"... {len(lines) - 6} training log lines omitted")
        print("\n".join(lines[-6:]))


def budget_name(epochs: int) -> str:
    return f"epochs_{int(epochs):03d}"


def train_budget(args: argparse.Namespace, epochs: int, output_dir: Path) -> Tuple[Path, Path]:
    run_name = budget_name(int(epochs))
    budget_dir = output_dir / run_name
    checkpoint = budget_dir / "checkpoints" / "unet_best.pt"
    log_dir = budget_dir / "training_logs"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_training and checkpoint.exists():
        return checkpoint, log_dir / f"unet_{run_name}_training_log.csv"

    cmd = [
        sys.executable,
        "training/train_downscaler.py",
        "--dataset-version",
        str(args.dataset_version),
        "--model",
        "unet",
        "--input-set",
        str(args.input_set),
        "--history-length",
        str(args.history_length),
        "--target-mode",
        str(args.target_mode),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--l1-weight",
        str(args.l1_weight),
        "--grad-clip",
        str(args.grad_clip),
        "--scheduler",
        str(args.scheduler),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--scheduler-factor",
        str(args.scheduler_factor),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--val-fraction",
        str(args.val_fraction),
        "--split-seed",
        str(args.split_seed),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--unet-padding-mode",
        str(args.padding_mode),
        "--unet-upsampling-mode",
        str(args.upsampling_mode),
        "--checkpoint-out",
        str(checkpoint),
        "--training-results-dir",
        str(log_dir),
        "--run-name",
        f"unet_{run_name}",
    ]
    if args.era5_path is not None:
        cmd.extend(["--era5-path", str(args.era5_path)])
    if args.prism_path is not None:
        cmd.extend(["--prism-path", str(args.prism_path)])

    print(f"training U-Net fixed budget: {epochs} epochs")
    run_cmd(cmd)
    (budget_dir / "training_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    return checkpoint, log_dir / f"unet_{run_name}_training_log.csv"


def read_training_curve(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def training_summary(curve_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not curve_rows:
        return {
            "actual_epochs": 0,
            "best_epoch": 0,
            "best_val_loss": float("nan"),
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
            "final_train_rmse": float("nan"),
            "final_val_rmse": float("nan"),
        }
    best = min(curve_rows, key=lambda row: float(row["val_loss"]))
    final = curve_rows[-1]
    return {
        "actual_epochs": int(final["epoch"]),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "final_train_rmse": float(final["train_rmse"]),
        "final_val_rmse": float(final["val_rmse"]),
    }


def mask_bundle(shape: Tuple[int, int], border_width: int, center_crop_fraction: float) -> Dict[str, np.ndarray]:
    h, w = shape
    if border_width < 1:
        raise ValueError("border-pixels must be >= 1")
    if border_width * 2 >= min(h, w):
        raise ValueError(f"border-pixels={border_width} is too large for target shape {(h, w)}")
    if not 0.0 < center_crop_fraction <= 1.0:
        raise ValueError("center-crop-fraction must be in (0, 1]")

    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    top[:border_width, :] = True
    bottom[-border_width:, :] = True
    left[:, :border_width] = True
    right[:, -border_width:] = True
    border = top | bottom | left | right
    corner = (top | bottom) & (left | right)

    crop_h = max(1, int(round(h * center_crop_fraction)))
    crop_w = max(1, int(round(w * center_crop_fraction)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    center_crop = np.zeros((h, w), dtype=bool)
    center_crop[y0 : y0 + crop_h, x0 : x0 + crop_w] = True

    return {
        "border": border,
        "center": ~border,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "corner": corner,
        "center_crop": center_crop,
    }


def distance_from_edge(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    return np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.int32)


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_flat.size < 2 or b_flat.size < 2:
        return 0.0
    value = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return value if np.isfinite(value) else 0.0


def compute_metrics(
    *,
    epochs: int,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
    masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    errors = pred_cube - target_cube
    abs_errors = np.abs(errors)
    row: Dict[str, Any] = {
        "epoch_budget": int(epochs),
        "rmse": rmse(errors),
        "mae": mae(errors),
        "bias": float(np.mean(errors)),
        "correlation": corrcoef(pred_cube, target_cube),
        "pred_mean": float(np.mean(pred_cube)),
        "target_mean": float(np.mean(target_cube)),
        "pred_variance": float(np.var(pred_cube)),
        "target_variance": float(np.var(target_cube)),
    }
    for name, mask2d in masks.items():
        mask = np.broadcast_to(mask2d, errors.shape)
        row[f"{name}_rmse"] = rmse(errors[mask])
        row[f"{name}_mae"] = mae(abs_errors[mask])
    row["border_center_rmse_ratio"] = float(row["border_rmse"] / max(row["center_rmse"], 1e-8))
    row["variance_ratio"] = float(row["pred_variance"] / max(row["target_variance"], 1e-8))
    row.update(metadata)
    return row


def distance_rows(*, epochs: int, errors: np.ndarray, distance_map: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for distance in range(int(distance_map.max()) + 1):
        mask2d = distance_map == distance
        mask = np.broadcast_to(mask2d, errors.shape)
        values = errors[mask]
        rows.append(
            {
                "epoch_budget": int(epochs),
                "distance_from_edge": int(distance),
                "rmse": rmse(values),
                "mae": mae(values),
                "n_values": int(values.size),
            }
        )
    return rows


def save_prediction_panel(
    *,
    era5: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    abs_error = np.abs(prediction - target)
    main_vmin = float(min(np.min(era5), np.min(prediction), np.min(target)))
    main_vmax = float(max(np.max(era5), np.max(prediction), np.max(target)))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    panels = [
        ("ERA5 upsampled", era5, "coolwarm", main_vmin, main_vmax),
        ("Prediction", prediction, "coolwarm", main_vmin, main_vmax),
        ("PRISM target", target, "coolwarm", main_vmin, main_vmax),
        ("Absolute error", abs_error, "magma", 0.0, float(np.max(abs_error))),
    ]
    for ax, (label, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        if label == "Absolute error":
            fig.colorbar(im, ax=ax, shrink=0.8, label="deg C")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_map(mean_abs_error: np.ndarray, output_path: Path, title: str) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.3), constrained_layout=True)
    im = ax.imshow(mean_abs_error, cmap="magma", vmin=0.0, vmax=float(np.max(mean_abs_error)))
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean |error| (deg C)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_distance_plot(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for epochs in sorted({int(row["epoch_budget"]) for row in rows}):
        budget_rows = [row for row in rows if int(row["epoch_budget"]) == epochs]
        distances = [int(row["distance_from_edge"]) for row in budget_rows]
        maes = [float(row["mae"]) for row in budget_rows]
        ax.plot(distances, maes, marker="o", markersize=2.5, linewidth=1.5, label=f"{epochs} epochs")
    ax.set_xlabel("Distance from nearest image edge (pixels)")
    ax.set_ylabel("Mean absolute error (deg C)")
    ax.set_title("Boundary error profile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curves(curves_by_budget: Dict[int, List[Dict[str, Any]]], output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for epochs, rows in sorted(curves_by_budget.items()):
        x = [int(row["epoch"]) for row in rows]
        train = [float(row["train_loss"]) for row in rows]
        val = [float(row["val_loss"]) for row in rows]
        axes[0].plot(x, train, linewidth=1.5, label=f"{epochs} epochs")
        axes[1].plot(x, val, linewidth=1.5, label=f"{epochs} epochs")
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def evaluate_budget(
    args: argparse.Namespace,
    epochs: int,
    checkpoint_path: Path,
    curve_path: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device, set_seed
    from models.baselines import upsample_latest_era5

    apply_dataset_version_to_args(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    curve_rows = read_training_curve(curve_path)
    train_info = training_summary(curve_rows)

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        verbose=False,
    )
    model, ckpt_history, input_norm, checkpoint_input_set, _, val_indices = load_checkpoint_model(
        "unet", checkpoint_path, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"{epochs} checkpoint history {ckpt_history} != {args.history_length}")
    if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
        raise RuntimeError(f"{epochs} checkpoint input_set {checkpoint_input_set} != {args.input_set}")
    if val_indices is None:
        raise RuntimeError(f"{epochs} checkpoint does not include validation indices")

    eval_indices = [int(i) for i in val_indices]
    if int(args.num_samples) > 0:
        eval_indices = eval_indices[: int(args.num_samples)]

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    era5_panels: List[np.ndarray] = []

    with torch.no_grad():
        for sample_idx in eval_indices:
            x, y = dataset[int(sample_idx)]
            xb = x.unsqueeze(0).to(device)
            yb = y.unsqueeze(0).to(device)
            target_size = (yb.shape[-2], yb.shape[-1])
            era5_up = upsample_latest_era5(xb, target_size=target_size)
            pred = model(normalize_input_batch(xb, input_norm), target_size=target_size)
            if pred.shape != yb.shape:
                raise RuntimeError(f"{epochs} prediction shape {tuple(pred.shape)} != target {tuple(yb.shape)}")
            predictions.append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
            targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))
            era5_panels.append(era5_up.squeeze().detach().cpu().numpy().astype(np.float64))

    pred_cube = np.stack(predictions, axis=0)
    target_cube = np.stack(targets, axis=0)
    masks = mask_bundle(target_cube.shape[-2:], int(args.border_pixels), float(args.center_crop_fraction))
    metadata = {
        "dataset_version": args.dataset_version,
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "target_mode": args.target_mode,
        "padding_mode": args.padding_mode,
        "upsampling_mode": args.upsampling_mode,
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "num_samples": int(len(eval_indices)),
        "border_pixels": int(args.border_pixels),
        "center_crop_fraction": float(args.center_crop_fraction),
    }
    metadata.update(train_info)
    metrics = compute_metrics(
        epochs=int(epochs),
        pred_cube=pred_cube,
        target_cube=target_cube,
        masks=masks,
        metadata=metadata,
    )
    errors = pred_cube - target_cube
    rows = distance_rows(epochs=int(epochs), errors=errors, distance_map=distance_from_edge(target_cube.shape[-2:]))

    budget_dir = output_dir / budget_name(int(epochs))
    save_prediction_panel(
        era5=era5_panels[0],
        prediction=predictions[0],
        target=targets[0],
        output_path=budget_dir / "prediction_panel.png",
        title=f"U-Net {epochs} epochs",
    )
    save_error_map(
        np.mean(np.abs(errors), axis=0),
        budget_dir / "absolute_error_map.png",
        f"U-Net {epochs} epochs mean absolute error",
    )
    save_distance_plot(rows, budget_dir / "error_vs_boundary_distance.png")
    (budget_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, rows, curve_rows


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    distance_metric_rows: List[Dict[str, Any]] = []
    curves_by_budget: Dict[int, List[Dict[str, Any]]] = {}

    for epochs in [int(item) for item in args.budgets]:
        checkpoint, curve_path = train_budget(args, epochs, output_dir)
        metrics, distance_rows_for_budget, curve_rows = evaluate_budget(
            args=args,
            epochs=epochs,
            checkpoint_path=checkpoint,
            curve_path=curve_path,
            output_dir=output_dir,
        )
        metric_rows.append(metrics)
        distance_metric_rows.extend(distance_rows_for_budget)
        curves_by_budget[epochs] = curve_rows
        print(
            f"{epochs:>4} epochs | rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
            f"border={metrics['border_rmse']:.4f} center={metrics['center_rmse']:.4f} "
            f"ratio={metrics['border_center_rmse_ratio']:.3f} best_epoch={metrics['best_epoch']}"
        )

    fields = [
        "epoch_budget",
        "actual_epochs",
        "best_epoch",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "final_train_rmse",
        "final_val_rmse",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "top_rmse",
        "bottom_rmse",
        "left_rmse",
        "right_rmse",
        "corner_rmse",
        "center_crop_rmse",
        "variance_ratio",
        "dataset_version",
        "input_set",
        "history_length",
        "target_mode",
        "padding_mode",
        "upsampling_mode",
        "seed",
        "split_seed",
        "num_samples",
        "border_pixels",
    ]
    write_csv(output_dir / "summary.csv", metric_rows, fields)
    write_csv(
        output_dir / "error_by_distance.csv",
        distance_metric_rows,
        ["epoch_budget", "distance_from_edge", "rmse", "mae", "n_values"],
    )
    save_distance_plot(distance_metric_rows, output_dir / "error_vs_boundary_distance.png")
    save_training_curves(curves_by_budget, output_dir / "training_curves.png")
    (output_dir / "undertraining_summary.json").write_text(
        json.dumps(
            {
                "dataset_version": args.dataset_version,
                "input_set": args.input_set,
                "history_length": int(args.history_length),
                "target_mode": args.target_mode,
                "padding_mode": args.padding_mode,
                "upsampling_mode": args.upsampling_mode,
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "budgets": [int(item) for item in args.budgets],
                "metrics": metric_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote undertraining diagnosis outputs under {output_dir}")


if __name__ == "__main__":
    main()
