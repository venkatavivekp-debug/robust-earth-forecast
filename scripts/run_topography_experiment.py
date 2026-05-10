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

VARIANTS = (
    ("unet_core4_h3", "core4", None),
    ("unet_core4_elev_h3", "core4_elev", "elevation"),
    ("unet_core4_topo_h3", "core4_topo", "elevation+slope+aspect+terrain_gradient_magnitude"),
)
VARIANT_MAP = {name: (name, input_set, features) for name, input_set, features in VARIANTS}


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled U-Net topography/static-context comparison.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/topography_context")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument(
        "--unet-padding-mode",
        choices=["reflection", "zero", "replicate"],
        default="reflection",
        help="Boundary padding used by U-Net convolution blocks.",
    )
    parser.add_argument(
        "--upsampling-mode",
        "--unet-upsampling-mode",
        dest="unet_upsampling_mode",
        choices=["bilinear", "convtranspose", "pixelshuffle"],
        default="bilinear",
        help="U-Net decoder upsampling path.",
    )
    parser.add_argument("--loss-mode", choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"], default="mse_l1")
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
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
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--high-pass-window", type=int, default=7)
    parser.add_argument("--local-window", type=int, default=7)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_MAP),
        default=[name for name, _, _ in VARIANTS],
        help="Subset of topography variants to run.",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
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


def selected_variants(args: argparse.Namespace) -> Tuple[Tuple[str, str, Optional[str]], ...]:
    return tuple(VARIANT_MAP[name] for name in args.variants)


def train_variants(
    args: argparse.Namespace,
    output_dir: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> Dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "training_logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    commands: Dict[str, List[str]] = {}
    checkpoints: Dict[str, Path] = {}
    static_path = repo_path(args.static_covariate_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")

    for variant_name, input_set, _features in variants:
        checkpoint = checkpoint_dir / f"{variant_name}_best.pt"
        checkpoints[variant_name] = checkpoint
        if args.skip_training and checkpoint.exists():
            continue

        cmd = [
            sys.executable,
            "training/train_downscaler.py",
            "--dataset-version",
            str(args.dataset_version),
            "--model",
            "unet",
            "--input-set",
            input_set,
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
            "--unet-base-channels",
            str(args.unet_base_channels),
            "--loss-mode",
            str(args.loss_mode),
            "--l1-weight",
            str(args.l1_weight),
            "--grad-weight",
            str(args.grad_weight),
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
            str(args.unet_padding_mode),
            "--unet-upsampling-mode",
            str(args.unet_upsampling_mode),
            "--checkpoint-out",
            str(checkpoint),
            "--training-results-dir",
            str(log_dir),
            "--run-name",
            f"{variant_name}_{args.target_mode}_{args.loss_mode}",
        ]
        if input_set != "core4":
            cmd.extend(["--static-covariate-path", str(static_path)])
        if args.era5_path is not None:
            cmd.extend(["--era5-path", str(args.era5_path)])
        if args.prism_path is not None:
            cmd.extend(["--prism-path", str(args.prism_path)])

        commands[variant_name] = cmd
        print(f"training {variant_name}")
        run_cmd(cmd)

    (output_dir / "training_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return checkpoints


def border_mask(shape: Tuple[int, int], width: int) -> np.ndarray:
    h, w = shape
    if width < 1:
        raise ValueError("border-pixels must be >= 1")
    if width * 2 >= min(h, w):
        raise ValueError(f"border-pixels={width} is too large for target shape {(h, w)}")
    mask = np.zeros((h, w), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    if af.size < 2 or bf.size < 2:
        return 0.0
    value = float(np.corrcoef(af, bf)[0, 1])
    return value if np.isfinite(value) else 0.0


def gradient_magnitude(cube: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(cube, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx**2 + gy**2)


def box_mean(cube: np.ndarray, window: int) -> np.ndarray:
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


def high_pass(cube: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(cube, dtype=np.float64) - box_mean(cube, window)


def local_contrast(cube: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    mean = box_mean(arr, window)
    mean_sq = box_mean(arr * arr, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def compute_metrics(
    *,
    model_name: str,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
    border_pixels: int,
    high_pass_window: int,
    local_window: int,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    err = pred_cube - target_cube
    abs_err = np.abs(err)
    mask2d = border_mask(pred_cube.shape[-2:], border_pixels)
    center2d = ~mask2d
    border = np.broadcast_to(mask2d, pred_cube.shape)
    center = np.broadcast_to(center2d, pred_cube.shape)

    pred_grad = gradient_magnitude(pred_cube)
    target_grad = gradient_magnitude(target_cube)
    pred_detail = high_pass(pred_cube, high_pass_window)
    target_detail = high_pass(target_cube, high_pass_window)
    pred_contrast = local_contrast(pred_cube, local_window)
    target_contrast = local_contrast(target_cube, local_window)

    pred_variance = float(np.var(pred_cube))
    target_variance = float(np.var(target_cube))
    pred_grad_mean = float(np.mean(pred_grad))
    target_grad_mean = float(np.mean(target_grad))
    pred_hf = float(np.mean(pred_detail**2))
    target_hf = float(np.mean(target_detail**2))
    pred_contrast_mean = float(np.mean(pred_contrast))
    target_contrast_mean = float(np.mean(target_contrast))

    row: Dict[str, Any] = {
        "model": model_name,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "correlation": corrcoef(pred_cube, target_cube),
        "pred_variance": pred_variance,
        "target_variance": target_variance,
        "variance_ratio": float(pred_variance / max(target_variance, 1e-8)),
        "border_rmse": float(np.sqrt(np.mean(err[border] ** 2))),
        "center_rmse": float(np.sqrt(np.mean(err[center] ** 2))),
        "border_center_rmse_ratio": 0.0,
        "target_gradient_mean": target_grad_mean,
        "prediction_gradient_mean": pred_grad_mean,
        "gradient_ratio": float(pred_grad_mean / max(target_grad_mean, 1e-8)),
        "high_frequency_ratio": float(pred_hf / max(target_hf, 1e-8)),
        "local_contrast_ratio": float(pred_contrast_mean / max(target_contrast_mean, 1e-8)),
        "gradient_magnitude_rmse": float(np.sqrt(np.mean((pred_grad - target_grad) ** 2))),
    }
    row["border_center_rmse_ratio"] = float(row["border_rmse"] / max(row["center_rmse"], 1e-8))
    row.update(metadata)
    return row


def evaluate_variant(
    *,
    args: argparse.Namespace,
    variant_name: str,
    input_set: str,
    checkpoint: Path,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    import torch

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device, set_seed
    from models.baselines import upsample_latest_era5

    apply_dataset_version_to_args(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    static_path = repo_path(args.static_covariate_path)

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=input_set,
        static_covariate_path=str(static_path) if input_set != "core4" else None,
        verbose=False,
    )
    model, ckpt_history, input_norm, checkpoint_input_set, _train_indices, val_indices = load_checkpoint_model(
        "unet", checkpoint, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"{variant_name} checkpoint history {ckpt_history} != {args.history_length}")
    if checkpoint_input_set is not None and checkpoint_input_set != input_set:
        raise RuntimeError(f"{variant_name} checkpoint input_set {checkpoint_input_set} != {input_set}")
    if val_indices is None:
        raise RuntimeError(f"{variant_name} checkpoint is missing validation indices")

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    era5_arrays: List[np.ndarray] = []
    dates: List[str] = []

    with torch.no_grad():
        for sample_idx in val_indices:
            x, y = dataset[int(sample_idx)]
            xb = x.unsqueeze(0).to(device)
            yb = y.unsqueeze(0).to(device)
            era5_up = upsample_latest_era5(xb, target_size=(yb.shape[-2], yb.shape[-1]))
            raw_pred = model(normalize_input_batch(xb, input_norm), target_size=(yb.shape[-2], yb.shape[-1]))
            pred = era5_up + raw_pred if getattr(model, "target_mode", args.target_mode) == "residual" else raw_pred
            if pred.shape != yb.shape:
                raise RuntimeError(f"{variant_name} prediction/target mismatch: {tuple(pred.shape)} vs {tuple(yb.shape)}")
            predictions.append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
            targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))
            era5_arrays.append(era5_up.squeeze().detach().cpu().numpy().astype(np.float64))
            dates.append(dataset.metadata(int(sample_idx)).date.strftime("%Y-%m-%d"))

    pred_cube = np.stack(predictions, axis=0)
    target_cube = np.stack(targets, axis=0)
    era5_cube = np.stack(era5_arrays, axis=0)
    metadata = {
        "dataset_version": args.dataset_version,
        "input_set": input_set,
        "history_length": int(args.history_length),
        "target_mode": args.target_mode,
        "loss_mode": args.loss_mode,
        "padding": args.unet_padding_mode,
        "upsampling": args.unet_upsampling_mode,
        "split_seed": int(args.split_seed),
        "seed": int(args.seed),
        "num_samples": int(len(val_indices)),
        "eval_indices": [int(i) for i in val_indices],
        "eval_dates": dates,
    }
    metrics = compute_metrics(
        model_name=variant_name,
        pred_cube=pred_cube,
        target_cube=target_cube,
        border_pixels=int(args.border_pixels),
        high_pass_window=int(args.high_pass_window),
        local_window=int(args.local_window),
        metadata=metadata,
    )
    return metrics, pred_cube, target_cube, era5_cube, [int(i) for i in val_indices], dates


def compute_persistence_metrics(
    *,
    args: argparse.Namespace,
    target_cube: np.ndarray,
    era5_cube: np.ndarray,
    val_indices: List[int],
    dates: List[str],
) -> Dict[str, Any]:
    metadata = {
        "dataset_version": args.dataset_version,
        "input_set": "core4",
        "history_length": int(args.history_length),
        "target_mode": "direct",
        "padding": "none",
        "upsampling": "bilinear",
        "split_seed": int(args.split_seed),
        "seed": int(args.seed),
        "num_samples": int(len(val_indices)),
        "eval_indices": val_indices,
        "eval_dates": dates,
    }
    return compute_metrics(
        model_name="persistence",
        pred_cube=era5_cube,
        target_cube=target_cube,
        border_pixels=int(args.border_pixels),
        high_pass_window=int(args.high_pass_window),
        local_window=int(args.local_window),
        metadata=metadata,
    )


def save_summary(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "model",
        "input_set",
        "loss_mode",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "variance_ratio",
        "gradient_ratio",
        "high_frequency_ratio",
        "local_contrast_ratio",
        "gradient_magnitude_rmse",
        "num_samples",
        "seed",
        "split_seed",
        "padding",
        "upsampling",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(list(rows), indent=2) + "\n")


def save_prediction_panel(
    predictions: Dict[str, np.ndarray],
    era5: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["ERA5 upsampled", "persistence", *[name for name, _, _ in variants], "PRISM target"]
    arrays = [era5, predictions["persistence"], *[predictions[name] for name, _, _ in variants], target]
    vmin = float(min(np.min(arr) for arr in arrays))
    vmax = float(max(np.max(arr) for arr in arrays))
    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 4), constrained_layout=True)
    im = None
    for ax, label, arr in zip(axes, labels, arrays):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="deg C")
    fig.suptitle("Topography context comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_maps(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["persistence", *[name for name, _, _ in variants]]
    errors = [np.abs(predictions[label] - target) for label in labels]
    vmax = float(max(np.max(err) for err in errors))
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), constrained_layout=True)
    im = None
    for ax, label, err in zip(axes, labels, errors):
        im = ax.imshow(err, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="|error| deg C")
    fig.suptitle("Absolute error maps")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_gradient_plot(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["target", "persistence", *[name for name, _, _ in variants]]
    arrays = [target, predictions["persistence"], *[predictions[name] for name, _, _ in variants]]
    maps = [gradient_magnitude(arr) for arr in arrays]
    vmax = float(max(np.percentile(arr, 99) for arr in maps))
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), constrained_layout=True)
    im = None
    for ax, label, arr in zip(axes, labels, maps):
        im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="gradient magnitude")
    fig.suptitle("Gradient comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_detail_maps(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    high_pass_window: int,
    output_path: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["target", "persistence", *[name for name, _, _ in variants]]
    arrays = [target, predictions["persistence"], *[predictions[name] for name, _, _ in variants]]
    details = [high_pass(arr, high_pass_window) for arr in arrays]
    vmax = float(max(np.percentile(np.abs(arr), 99) for arr in details))
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), constrained_layout=True)
    im = None
    for ax, label, arr in zip(axes, labels, details):
        im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="high-pass deg C")
    fig.suptitle("High-frequency detail comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_local_patch(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    patch_size: int,
    output_path: Path,
    variants: Sequence[Tuple[str, str, Optional[str]]],
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    grad = gradient_magnitude(target)
    pad = patch_size // 2
    score = grad.copy()
    score[:pad, :] = -np.inf
    score[-pad:, :] = -np.inf
    score[:, :pad] = -np.inf
    score[:, -pad:] = -np.inf
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    ys = slice(y - pad, y - pad + patch_size)
    xs = slice(x - pad, x - pad + patch_size)

    labels = ["target", "persistence", *[name for name, _, _ in variants]]
    arrays = [target[ys, xs], predictions["persistence"][ys, xs], *[predictions[name][ys, xs] for name, _, _ in variants]]
    vmin = float(min(np.min(arr) for arr in arrays))
    vmax = float(max(np.max(arr) for arr in arrays))
    fig, axes = plt.subplots(2, len(labels), figsize=(4 * len(labels), 7), constrained_layout=True)
    im = None
    for ax, label, arr in zip(axes[0], labels, arrays):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="deg C")
    err_vmax = float(max(np.max(np.abs(arr - arrays[0])) for arr in arrays[1:]))
    axes[1, 0].imshow(gradient_magnitude(arrays[0]), cmap="viridis")
    axes[1, 0].set_title("target gradient")
    axes[1, 0].axis("off")
    err_im = None
    for ax, label, arr in zip(axes[1, 1:], labels[1:], arrays[1:]):
        err_im = ax.imshow(np.abs(arr - arrays[0]), cmap="magma", vmin=0.0, vmax=err_vmax)
        ax.set_title(f"{label} |error|")
        ax.axis("off")
    fig.colorbar(err_im, ax=axes[1, 1:], shrink=0.85, label="absolute error")
    fig.suptitle(f"Local high-gradient patch y={y}, x={x}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.skip_training and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets.dataset_paths import apply_dataset_version_to_args

    apply_dataset_version_to_args(args)
    variants = selected_variants(args)
    config = vars(args).copy()
    config["variant_details"] = [
        {"name": name, "input_set": input_set, "features": features}
        for name, input_set, features in variants
    ]
    (output_dir / "topography_experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")

    checkpoints = train_variants(args, output_dir, variants)
    rows: List[Dict[str, Any]] = []
    predictions_for_plot: Dict[str, np.ndarray] = {}
    target_for_plot: Optional[np.ndarray] = None
    era5_for_plot: Optional[np.ndarray] = None
    base_target_cube: Optional[np.ndarray] = None
    base_era5_cube: Optional[np.ndarray] = None
    base_val_indices: Optional[List[int]] = None
    base_dates: Optional[List[str]] = None

    for variant_name, input_set, _features in variants:
        metrics, pred_cube, target_cube, era5_cube, val_indices, dates = evaluate_variant(
            args=args,
            variant_name=variant_name,
            input_set=input_set,
            checkpoint=checkpoints[variant_name],
        )
        (output_dir / variant_name).mkdir(parents=True, exist_ok=True)
        (output_dir / variant_name / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        rows.append(metrics)
        predictions_for_plot[variant_name] = pred_cube[0]
        if base_target_cube is None:
            base_target_cube = target_cube
            base_era5_cube = era5_cube
            base_val_indices = val_indices
            base_dates = dates
            target_for_plot = target_cube[0]
            era5_for_plot = era5_cube[0]
        elif base_val_indices != val_indices:
            raise RuntimeError("Topography variants used different validation splits")

    assert base_target_cube is not None and base_era5_cube is not None and base_val_indices is not None and base_dates is not None
    persistence = compute_persistence_metrics(
        args=args,
        target_cube=base_target_cube,
        era5_cube=base_era5_cube,
        val_indices=base_val_indices,
        dates=base_dates,
    )
    rows.insert(0, persistence)
    predictions_for_plot["persistence"] = base_era5_cube[0]

    save_summary(rows, output_dir)
    assert target_for_plot is not None and era5_for_plot is not None
    save_prediction_panel(predictions_for_plot, era5_for_plot, target_for_plot, output_dir / "prediction_panel.png", variants)
    save_error_maps(predictions_for_plot, target_for_plot, output_dir / "absolute_error_maps.png", variants)
    save_gradient_plot(predictions_for_plot, target_for_plot, output_dir / "gradient_comparison.png", variants)
    save_detail_maps(
        predictions_for_plot,
        target_for_plot,
        int(args.high_pass_window),
        output_dir / "high_frequency_detail_maps.png",
        variants,
    )
    save_local_patch(predictions_for_plot, target_for_plot, int(args.patch_size), output_dir / "local_patch_comparison.png", variants)

    print("controlled topography context comparison")
    for row in rows:
        print(
            f"{row['model']:>21} | rmse={row['rmse']:.4f} mae={row['mae']:.4f} "
            f"border={row['border_rmse']:.4f} center={row['center_rmse']:.4f} "
            f"grad_ratio={row['gradient_ratio']:.3f} hf_ratio={row['high_frequency_ratio']:.3f}"
        )
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
