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

LEARNED_MODELS = ("plain_encoder_decoder", "unet")


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled spatial benchmark: persistence vs PlainEncoderDecoder vs U-Net."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/spatial_benchmark")
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
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
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates all validation samples")
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true", help="Reuse checkpoints already in --output-dir")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output-dir before running")
    return parser.parse_args()


def repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cmd(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 5:
            print(f"... {len(lines) - 5} training log lines omitted")
        print("\n".join(lines[-5:]))


def train_models(args: argparse.Namespace, output_dir: Path) -> Dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "training_logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: Dict[str, Path] = {}
    commands: Dict[str, List[str]] = {}

    for model_name in LEARNED_MODELS:
        checkpoint = checkpoint_dir / f"{model_name}_best.pt"
        checkpoints[model_name] = checkpoint
        if args.skip_training and checkpoint.exists():
            continue

        cmd = [
            sys.executable,
            "training/train_downscaler.py",
            "--dataset-version",
            str(args.dataset_version),
            "--model",
            model_name,
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
            "--checkpoint-out",
            str(checkpoint),
            "--training-results-dir",
            str(log_dir),
            "--run-name",
            f"{model_name}_{args.input_set}_h{args.history_length}_{args.target_mode}",
        ]
        if args.era5_path is not None:
            cmd.extend(["--era5-path", str(args.era5_path)])
        if args.prism_path is not None:
            cmd.extend(["--prism-path", str(args.prism_path)])

        commands[model_name] = cmd
        print(f"training {model_name}")
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
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_flat.size < 2 or b_flat.size < 2:
        return 0.0
    value = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return value if np.isfinite(value) else 0.0


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(arr, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx**2 + gy**2)


def compute_metrics(
    *,
    model_name: str,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
    border_pixels: int,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    err = pred_cube - target_cube
    abs_err = np.abs(err)
    mask2d = border_mask(pred_cube.shape[-2:], border_pixels)
    center2d = ~mask2d
    mask = np.broadcast_to(mask2d, pred_cube.shape)
    center = np.broadcast_to(center2d, pred_cube.shape)

    target_grad = gradient_magnitude(target_cube)
    payload: Dict[str, Any] = {
        "model": model_name,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "correlation": corrcoef(pred_cube, target_cube),
        "pred_variance": float(np.var(pred_cube)),
        "target_variance": float(np.var(target_cube)),
        "border_rmse": float(np.sqrt(np.mean(err[mask] ** 2))),
        "center_rmse": float(np.sqrt(np.mean(err[center] ** 2))),
        "border_pixels": int(border_pixels),
        "gradient_error_correlation": corrcoef(target_grad, abs_err),
    }
    payload["variance_ratio"] = float(payload["pred_variance"] / max(payload["target_variance"], 1e-8))
    payload["border_center_rmse_ratio"] = float(
        payload["border_rmse"] / max(payload["center_rmse"], 1e-8)
    )
    payload.update(metadata)
    return payload


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


def save_combined_predictions(
    *,
    era5: np.ndarray,
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["ERA5 upsampled", "persistence", "plain_encoder_decoder", "unet", "PRISM target"]
    arrays = [era5, predictions["persistence"], predictions["plain_encoder_decoder"], predictions["unet"], target]
    vmin = float(min(np.min(a) for a in arrays))
    vmax = float(max(np.max(a) for a in arrays))

    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 4), constrained_layout=True)
    im = None
    for ax, label, arr in zip(axes, labels, arrays):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="deg C")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_maps(
    *,
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    labels = ["persistence", "plain_encoder_decoder", "unet"]
    errors = [np.abs(predictions[label] - target) for label in labels]
    vmax = float(max(np.max(e) for e in errors))

    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), constrained_layout=True)
    im = None
    for ax, label, err in zip(axes, labels, errors):
        im = ax.imshow(err, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="|error| deg C")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_gradient_diagnostic(
    *,
    target_cube: np.ndarray,
    pred_cube: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    grad = gradient_magnitude(target_cube).reshape(-1)
    err = np.abs(pred_cube - target_cube).reshape(-1)
    rng = np.random.default_rng(seed)
    n = min(5000, grad.size)
    take = rng.choice(grad.size, size=n, replace=False)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(grad[take], err[take], s=4, alpha=0.18)
    ax.set_xlabel("PRISM gradient magnitude")
    ax.set_ylabel("|prediction - target|")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_models(args: argparse.Namespace, output_dir: Path, checkpoints: Dict[str, Path]) -> List[Dict[str, Any]]:
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

    loaded: Dict[str, Tuple[Any, Optional[Dict[str, List[float]]], List[int], List[int]]] = {}
    split_signature: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
    for model_name, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        model, ckpt_history, input_norm, checkpoint_input_set, train_indices, val_indices = load_checkpoint_model(
            model_name, checkpoint, device
        )
        if int(ckpt_history) != int(args.history_length):
            raise RuntimeError(f"{model_name} checkpoint history {ckpt_history} != {args.history_length}")
        if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
            raise RuntimeError(f"{model_name} checkpoint input_set {checkpoint_input_set} != {args.input_set}")
        if train_indices is None or val_indices is None:
            raise RuntimeError(f"{model_name} checkpoint is missing split metadata")

        signature = (
            tuple(int(i) for i in train_indices),
            tuple(int(i) for i in val_indices),
        )
        if split_signature is None:
            split_signature = signature
        elif signature != split_signature:
            raise RuntimeError("Spatial benchmark checkpoints use different train/validation splits")
        loaded[model_name] = (model, input_norm, list(signature[0]), list(signature[1]))

    assert split_signature is not None
    eval_indices = list(split_signature[1])
    if args.num_samples > 0:
        eval_indices = eval_indices[: min(int(args.num_samples), len(eval_indices))]
    if not eval_indices:
        raise RuntimeError("No validation samples available")

    sample_predictions: Dict[str, np.ndarray] = {}
    sample_era5: Optional[np.ndarray] = None
    sample_target: Optional[np.ndarray] = None
    sample_date: Optional[str] = None
    metric_rows: List[Dict[str, Any]] = []

    model_names = ("persistence", *LEARNED_MODELS)
    for model_name in model_names:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        era5_arrays: List[np.ndarray] = []
        dates: List[str] = []

        with torch.no_grad():
            for sample_idx in eval_indices:
                x, y = dataset[int(sample_idx)]
                xb = x.unsqueeze(0).to(device)
                yb = y.unsqueeze(0).to(device)
                era5_up = upsample_latest_era5(xb, target_size=(yb.shape[-2], yb.shape[-1]))

                if model_name == "persistence":
                    pred = era5_up
                    target_mode = "direct"
                else:
                    model, input_norm, _, _ = loaded[model_name]
                    raw_pred = model(
                        normalize_input_batch(xb, input_norm),
                        target_size=(yb.shape[-2], yb.shape[-1]),
                    )
                    target_mode = getattr(model, "target_mode", args.target_mode)
                    if target_mode == "residual":
                        pred = era5_up + raw_pred
                    else:
                        pred = raw_pred

                if pred.shape != yb.shape:
                    raise RuntimeError(
                        f"{model_name} prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(yb.shape)}"
                    )

                predictions.append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
                targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))
                era5_arrays.append(era5_up.squeeze().detach().cpu().numpy().astype(np.float64))
                dates.append(dataset.metadata(int(sample_idx)).date.strftime("%Y-%m-%d"))

        pred_cube = np.stack(predictions, axis=0)
        target_cube = np.stack(targets, axis=0)
        era5_cube = np.stack(era5_arrays, axis=0)
        metadata = {
            "dataset_version": args.dataset_version,
            "input_set": args.input_set,
            "history_length": int(args.history_length),
            "target_mode": target_mode,
            "split_seed": int(args.split_seed),
            "seed": int(args.seed),
            "num_samples": int(len(eval_indices)),
            "eval_indices": [int(i) for i in eval_indices],
            "eval_dates": dates,
        }
        metrics = compute_metrics(
            model_name=model_name,
            pred_cube=pred_cube,
            target_cube=target_cube,
            border_pixels=int(args.border_pixels),
            metadata=metadata,
        )
        metric_rows.append(metrics)
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        save_prediction_panel(
            era5=era5_cube[0],
            prediction=pred_cube[0],
            target=target_cube[0],
            output_path=model_dir / "prediction_panel.png",
            title=f"{model_name} | {dates[0]} | {args.input_set}_h{args.history_length}",
        )
        save_gradient_diagnostic(
            target_cube=target_cube,
            pred_cube=pred_cube,
            output_path=model_dir / "gradient_vs_error.png",
            title=f"{model_name}: target gradient vs absolute error",
            seed=int(args.seed),
        )

        if sample_target is None:
            sample_era5 = era5_cube[0]
            sample_target = target_cube[0]
            sample_date = dates[0]
        sample_predictions[model_name] = pred_cube[0]

    assert sample_era5 is not None and sample_target is not None and sample_date is not None
    save_combined_predictions(
        era5=sample_era5,
        predictions=sample_predictions,
        target=sample_target,
        output_path=output_dir / "prediction_panel.png",
        title=f"Controlled spatial benchmark | {sample_date}",
    )
    save_error_maps(
        predictions=sample_predictions,
        target=sample_target,
        output_path=output_dir / "absolute_error_maps.png",
        title=f"Absolute error maps | {sample_date}",
    )
    return metric_rows


def save_summary(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "model",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "pred_variance",
        "target_variance",
        "variance_ratio",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "gradient_error_correlation",
        "num_samples",
        "target_mode",
        "input_set",
        "history_length",
        "split_seed",
        "seed",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / "summary.json").write_text(json.dumps(list(rows), indent=2) + "\n")


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
    config = vars(args).copy()
    (output_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2) + "\n")

    checkpoints = train_models(args, output_dir)
    rows = evaluate_models(args, output_dir, checkpoints)
    save_summary(rows, output_dir)

    print("controlled spatial benchmark")
    for row in rows:
        print(
            f"{row['model']:>21} | rmse={row['rmse']:.4f} mae={row['mae']:.4f} "
            f"border={row['border_rmse']:.4f} center={row['center_rmse']:.4f} "
            f"ratio={row['border_center_rmse_ratio']:.3f}"
        )
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
