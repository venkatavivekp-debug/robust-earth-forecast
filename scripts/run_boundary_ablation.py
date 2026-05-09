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

VARIANTS: Dict[str, Dict[str, str]] = {
    "reflection_bilinear": {"padding_mode": "reflection", "upsample_mode": "bilinear"},
    "zero_bilinear": {"padding_mode": "zero", "upsample_mode": "bilinear"},
    "replicate_bilinear": {"padding_mode": "replicate", "upsample_mode": "bilinear"},
    "reflection_convtranspose": {"padding_mode": "reflection", "upsample_mode": "convtranspose"},
}


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled U-Net padding and upsampling ablations for ERA5->PRISM boundary behavior."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/boundary_ablation")
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-mode", choices=["direct"], default="direct")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates every validation sample")
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--center-crop-fraction", type=float, default=0.5)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--skip-training", action="store_true", help="Reuse checkpoints already in --output-dir")
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


def train_variant(args: argparse.Namespace, variant: str, output_dir: Path) -> Path:
    variant_cfg = VARIANTS[variant]
    variant_dir = output_dir / variant
    checkpoint = variant_dir / "checkpoints" / "unet_best.pt"
    log_dir = variant_dir / "training_logs"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_training and checkpoint.exists():
        return checkpoint

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
        str(args.epochs),
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
        variant_cfg["padding_mode"],
        "--unet-upsampling-mode",
        variant_cfg["upsample_mode"],
        "--checkpoint-out",
        str(checkpoint),
        "--training-results-dir",
        str(log_dir),
        "--run-name",
        f"unet_{variant}_{args.input_set}_h{args.history_length}_{args.target_mode}",
    ]
    if args.era5_path is not None:
        cmd.extend(["--era5-path", str(args.era5_path)])
    if args.prism_path is not None:
        cmd.extend(["--prism-path", str(args.prism_path)])

    print(f"training boundary ablation: {variant}")
    run_cmd(cmd)
    (variant_dir / "training_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    return checkpoint


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


def gradient_magnitude(cube: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(cube, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx**2 + gy**2)


def box_mean(cube: np.ndarray, window: int = 7) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    pad = window // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float64)
    h, w = arr.shape[-2:]
    for dy in range(window):
        for dx in range(window):
            out += padded[:, dy : dy + h, dx : dx + w]
    out /= float(window * window)
    return out[0] if squeeze else out


def high_pass(cube: np.ndarray, window: int = 7) -> np.ndarray:
    return np.asarray(cube, dtype=np.float64) - box_mean(cube, window)


def local_contrast(cube: np.ndarray, window: int = 7) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    mean = box_mean(arr, window)
    mean_sq = box_mean(arr * arr, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def compute_metrics(
    *,
    variant: str,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
    masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    errors = pred_cube - target_cube
    abs_errors = np.abs(errors)
    row: Dict[str, Any] = {
        "variant": variant,
        "rmse": rmse(errors),
        "mae": mae(errors),
        "bias": float(np.mean(errors)),
        "correlation": corrcoef(pred_cube, target_cube),
        "pred_mean": float(np.mean(pred_cube)),
        "target_mean": float(np.mean(target_cube)),
        "pred_variance": float(np.var(pred_cube)),
        "target_variance": float(np.var(target_cube)),
    }
    pred_grad = gradient_magnitude(pred_cube)
    target_grad = gradient_magnitude(target_cube)
    pred_detail = high_pass(pred_cube)
    target_detail = high_pass(target_cube)
    pred_contrast = local_contrast(pred_cube)
    target_contrast = local_contrast(target_cube)
    row["prediction_gradient_mean"] = float(np.mean(pred_grad))
    row["target_gradient_mean"] = float(np.mean(target_grad))
    row["gradient_ratio"] = float(row["prediction_gradient_mean"] / max(row["target_gradient_mean"], 1e-8))
    row["gradient_magnitude_rmse"] = rmse(pred_grad - target_grad)
    row["prediction_high_frequency_energy"] = float(np.mean(pred_detail**2))
    row["target_high_frequency_energy"] = float(np.mean(target_detail**2))
    row["high_frequency_ratio"] = float(
        row["prediction_high_frequency_energy"] / max(row["target_high_frequency_energy"], 1e-8)
    )
    row["prediction_local_contrast"] = float(np.mean(pred_contrast))
    row["target_local_contrast"] = float(np.mean(target_contrast))
    row["local_contrast_ratio"] = float(row["prediction_local_contrast"] / max(row["target_local_contrast"], 1e-8))
    for name, mask2d in masks.items():
        mask = np.broadcast_to(mask2d, errors.shape)
        row[f"{name}_rmse"] = rmse(errors[mask])
        row[f"{name}_mae"] = mae(abs_errors[mask])
        if name in {"border", "center", "top", "bottom", "left", "right"}:
            pred_var = float(np.var(pred_cube[mask]))
            target_var = float(np.var(target_cube[mask]))
            row[f"{name}_pred_variance"] = pred_var
            row[f"{name}_target_variance"] = target_var
            row[f"{name}_variance_ratio"] = float(pred_var / max(target_var, 1e-8))
    row["border_center_rmse_ratio"] = float(row["border_rmse"] / max(row["center_rmse"], 1e-8))
    row["variance_ratio"] = float(row["pred_variance"] / max(row["target_variance"], 1e-8))
    row.update(metadata)
    return row


def distance_rows(
    *,
    variant: str,
    errors: np.ndarray,
    distance_map: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for distance in range(int(distance_map.max()) + 1):
        mask2d = distance_map == distance
        mask = np.broadcast_to(mask2d, errors.shape)
        values = errors[mask]
        rows.append(
            {
                "variant": variant,
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


def save_gradient_detail_maps(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    pred_grad = gradient_magnitude(prediction)
    target_grad = gradient_magnitude(target)
    pred_detail = high_pass(prediction)
    target_detail = high_pass(target)
    pred_contrast = local_contrast(prediction)
    target_contrast = local_contrast(target)
    grad_vmax = float(max(np.percentile(pred_grad, 99), np.percentile(target_grad, 99)))
    detail_vmax = float(max(np.percentile(np.abs(pred_detail), 99), np.percentile(np.abs(target_detail), 99)))
    contrast_vmax = float(max(np.percentile(pred_contrast, 99), np.percentile(target_contrast, 99)))

    fig, axes = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=True)
    panels = [
        ("prediction gradient", pred_grad, "viridis", 0.0, grad_vmax),
        ("target gradient", target_grad, "viridis", 0.0, grad_vmax),
        ("prediction high-pass", pred_detail, "coolwarm", -detail_vmax, detail_vmax),
        ("target high-pass", target_detail, "coolwarm", -detail_vmax, detail_vmax),
        ("prediction contrast", pred_contrast, "magma", 0.0, contrast_vmax),
        ("target contrast", target_contrast, "magma", 0.0, contrast_vmax),
    ]
    for ax, (label, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_distance_plot(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for variant in sorted({str(row["variant"]) for row in rows}):
        distances = sorted(int(row["distance_from_edge"]) for row in rows if row["variant"] == variant)
        maes = [float(row["mae"]) for row in rows if row["variant"] == variant]
        ax.plot(distances, maes, marker="o", markersize=2.5, linewidth=1.5, label=variant)
    ax.set_xlabel("Distance from nearest image edge (pixels)")
    ax.set_ylabel("Mean absolute error (deg C)")
    ax.set_title("Boundary error profile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_border_mask(shape: Tuple[int, int], border_pixels: int, output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    masks = mask_bundle(shape, border_pixels, center_crop_fraction=0.5)
    fig, ax = plt.subplots(figsize=(4.4, 4), constrained_layout=True)
    im = ax.imshow(masks["border"].astype(float), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"Border mask ({border_pixels}px)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
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


def evaluate_variant(args: argparse.Namespace, variant: str, checkpoint_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device, set_seed
    from models.baselines import upsample_latest_era5

    apply_dataset_version_to_args(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)

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
        raise RuntimeError(f"{variant} checkpoint history {ckpt_history} != {args.history_length}")
    if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
        raise RuntimeError(f"{variant} checkpoint input_set {checkpoint_input_set} != {args.input_set}")
    if val_indices is None:
        raise RuntimeError(f"{variant} checkpoint does not include validation indices")

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
                raise RuntimeError(f"{variant} prediction shape {tuple(pred.shape)} != target {tuple(yb.shape)}")
            predictions.append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
            targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))
            era5_panels.append(era5_up.squeeze().detach().cpu().numpy().astype(np.float64))

    pred_cube = np.stack(predictions, axis=0)
    target_cube = np.stack(targets, axis=0)
    masks = mask_bundle(target_cube.shape[-2:], int(args.border_pixels), float(args.center_crop_fraction))
    metadata = {
        "padding_mode": VARIANTS[variant]["padding_mode"],
        "upsample_mode": VARIANTS[variant]["upsample_mode"],
        "dataset_version": args.dataset_version,
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "target_mode": args.target_mode,
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "num_samples": int(len(eval_indices)),
        "border_pixels": int(args.border_pixels),
        "center_crop_fraction": float(args.center_crop_fraction),
    }
    metrics = compute_metrics(
        variant=variant,
        pred_cube=pred_cube,
        target_cube=target_cube,
        masks=masks,
        metadata=metadata,
    )
    errors = pred_cube - target_cube
    rows = distance_rows(variant=variant, errors=errors, distance_map=distance_from_edge(target_cube.shape[-2:]))

    variant_dir = repo_path(args.output_dir) / variant
    save_prediction_panel(
        era5=era5_panels[0],
        prediction=predictions[0],
        target=targets[0],
        output_path=variant_dir / "prediction_panel.png",
        title=f"{variant} prediction panel",
    )
    save_error_map(
        np.mean(np.abs(errors), axis=0),
        variant_dir / "absolute_error_map.png",
        f"{variant} mean absolute error",
    )
    save_gradient_detail_maps(
        prediction=predictions[0],
        target=targets[0],
        output_path=variant_dir / "gradient_detail_maps.png",
        title=f"{variant} reconstruction detail",
    )
    save_distance_plot(rows, variant_dir / "error_vs_boundary_distance.png")
    (variant_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, rows


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = [str(v) for v in args.variants]
    checkpoints: Dict[str, Path] = {}
    for variant in selected_variants:
        checkpoints[variant] = train_variant(args, variant, output_dir)

    metric_rows: List[Dict[str, Any]] = []
    distance_metric_rows: List[Dict[str, Any]] = []
    for variant in selected_variants:
        metrics, rows = evaluate_variant(args, variant, checkpoints[variant])
        metric_rows.append(metrics)
        distance_metric_rows.extend(rows)
        print(
            f"{variant:>24} | rmse={metrics['rmse']:.4f} border={metrics['border_rmse']:.4f} "
            f"center={metrics['center_rmse']:.4f} ratio={metrics['border_center_rmse_ratio']:.3f}"
        )

    metric_fields = [
        "variant",
        "padding_mode",
        "upsample_mode",
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
        "pred_mean",
        "target_mean",
        "variance_ratio",
        "gradient_ratio",
        "high_frequency_ratio",
        "local_contrast_ratio",
        "gradient_magnitude_rmse",
        "border_variance_ratio",
        "center_variance_ratio",
        "top_variance_ratio",
        "bottom_variance_ratio",
        "left_variance_ratio",
        "right_variance_ratio",
        "dataset_version",
        "input_set",
        "history_length",
        "target_mode",
        "seed",
        "split_seed",
        "num_samples",
        "border_pixels",
    ]
    write_csv(output_dir / "summary.csv", metric_rows, metric_fields)
    write_csv(
        output_dir / "error_by_distance.csv",
        distance_metric_rows,
        ["variant", "distance_from_edge", "rmse", "mae", "n_values"],
    )
    (output_dir / "boundary_ablation_summary.json").write_text(
        json.dumps(
            {
                "dataset_version": args.dataset_version,
                "input_set": args.input_set,
                "history_length": int(args.history_length),
                "target_mode": args.target_mode,
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "variants": selected_variants,
                "metrics": metric_rows,
            },
            indent=2,
        )
        + "\n"
    )
    save_distance_plot(distance_metric_rows, output_dir / "error_vs_boundary_distance.png")
    if metric_rows:
        first_variant = selected_variants[0]
        import torch
        from datasets.dataset_paths import apply_dataset_version_to_args
        from datasets.prism_dataset import ERA5_PRISM_Dataset

        apply_dataset_version_to_args(args)
        dataset = ERA5_PRISM_Dataset(
            era5_path=str(repo_path(args.era5_path)),
            prism_path=str(repo_path(args.prism_path)),
            history_length=int(args.history_length),
            input_set=str(args.input_set),
            verbose=False,
        )
        checkpoint = torch.load(checkpoints[first_variant], map_location="cpu")
        val_indices = checkpoint.get("val_indices") or [0]
        _, y = dataset[int(val_indices[0])]
        save_border_mask(tuple(y.shape[-2:]), int(args.border_pixels), output_dir / "border_mask.png")

    print(f"wrote boundary ablation outputs under {output_dir}")


if __name__ == "__main__":
    main()
