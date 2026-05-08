from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

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
    parser = argparse.ArgumentParser(description="Evaluate ERA5->PRISM downscaling baselines and temporal models")
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Select ERA5/PRISM paths from datasets/<version>/paths.json when paths are omitted",
    )
    parser.add_argument("--era5-path", type=str, default=None, help="ERA5 NetCDF (default: from --dataset-version)")
    parser.add_argument("--prism-path", type=str, default=None, help="PRISM directory (default: from --dataset-version)")
    parser.add_argument("--input-set", type=str, choices=["t2m", "core4", "extended"], default="extended")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["persistence", "era5_upsampled", "linear", "cnn", "convlstm"],
        choices=["persistence", "era5_upsampled", "linear", "cnn", "unet", "convlstm"],
        help="Models/baselines to evaluate",
    )
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cnn-checkpoint", type=str, default="checkpoints/cnn_best.pt")
    parser.add_argument("--unet-checkpoint", type=str, default="checkpoints/unet_best.pt")
    parser.add_argument("--convlstm-checkpoint", type=str, default="checkpoints/convlstm_best.pt")
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["direct", "residual"],
        default="direct",
        help="Fallback for checkpoints without target_mode metadata",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--results-dir", type=str, default="results/evaluation")
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="If set, raise an error when CNN/ConvLSTM do not improve over persistence RMSE",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> Any:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(n_total: int, val_fraction: float, split_seed: int) -> Tuple[List[int], List[int]]:
    if n_total < 2:
        raise ValueError("At least 2 samples are required to build train/validation splits")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    val_size = max(1, int(round(n_total * val_fraction)))
    if val_size >= n_total:
        val_size = n_total - 1

    all_indices = np.arange(n_total)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(all_indices)
    val_indices = sorted(all_indices[:val_size].tolist())
    train_indices = sorted(all_indices[val_size:].tolist())
    return train_indices, val_indices


def recommended_prism_days(history_length: int, min_samples: int) -> int:
    minimum_dates = history_length + min_samples - 1
    return max(10, minimum_dates + 2)


def build_insufficient_samples_message(
    *,
    history_length: int,
    usable_samples: int,
    min_required: int,
    candidate_dates: int,
) -> str:
    suggested_days = recommended_prism_days(history_length, min_required)
    return (
        f"History length {history_length} produced only {usable_samples} usable sample(s). "
        f"At least {min_required} are required for train/validation split. "
        f"Candidate PRISM dates scanned: {candidate_dates}. "
        "Download more aligned PRISM/ERA5 dates, for example:\n"
        f"  python data_pipeline/download_prism.py --start-date 20230101 --days {suggested_days} --variable tmean\n"
        "  python data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite"
    )


def load_checkpoint_model(
    model_name: str, checkpoint_path: Path, device: Any
) -> Tuple[Any, int, Optional[Dict[str, List[float]]], Optional[str], Optional[List[int]], Optional[List[int]]]:
    from models.cnn_downscaler import CNNDownscaler
    from models.convlstm_downscaler import ConvLSTMDownscaler
    from models.unet_downscaler import UNetDownscaler

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format for {checkpoint_path}")

    model_config = checkpoint.get("model_config", {})
    history_length = int(checkpoint.get("history_length", model_config.get("in_channels", 1)))

    if model_name == "cnn":
        model = CNNDownscaler(
            in_channels=int(model_config.get("in_channels", history_length)),
            out_channels=int(model_config.get("out_channels", 1)),
            base_channels=int(model_config.get("base_channels", 32)),
        )
    elif model_name == "unet":
        model = UNetDownscaler(
            in_channels=int(model_config.get("in_channels", history_length)),
            out_channels=int(model_config.get("out_channels", 1)),
            base_channels=int(model_config.get("base_channels", 24)),
        )
    elif model_name == "convlstm":
        model = ConvLSTMDownscaler(
            input_channels=int(model_config.get("input_channels", 1)),
            hidden_channels=int(model_config.get("hidden_channels", 32)),
            out_channels=int(model_config.get("out_channels", 1)),
            kernel_size=int(model_config.get("kernel_size", 3)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    target_mode = checkpoint.get("target_mode", checkpoint.get("args", {}).get("target_mode"))
    if target_mode is not None:
        model.target_mode = str(target_mode)
    model.to(device)
    model.eval()
    input_norm = checkpoint.get("input_norm")
    checkpoint_input_set = checkpoint.get("args", {}).get("input_set")
    train_indices = checkpoint.get("train_indices")
    val_indices = checkpoint.get("val_indices")
    return model, history_length, input_norm, checkpoint_input_set, train_indices, val_indices


def normalize_input_batch(
    x: torch.Tensor,
    input_norm: Optional[Dict[str, List[float]]],
) -> torch.Tensor:
    if not input_norm:
        return x

    mean_vals = input_norm.get("mean")
    std_vals = input_norm.get("std")
    if not mean_vals or not std_vals:
        return x

    mean = torch.tensor(mean_vals, device=x.device, dtype=x.dtype).clamp(min=-1e6, max=1e6)
    std = torch.tensor(std_vals, device=x.device, dtype=x.dtype).clamp(min=1e-6, max=1e6)

    if x.dim() == 5:
        return (x - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)
    if x.dim() == 4:
        return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    return x


def compute_metrics(errors: Sequence[Dict[str, float]]) -> Dict[str, float]:
    rmse = float(np.mean([row["rmse"] for row in errors]))
    mae = float(np.mean([row["mae"] for row in errors]))
    bias = float(np.mean([row["bias"] for row in errors]))
    correlation = float(np.mean([row["correlation"] for row in errors]))
    pred_variance = float(np.mean([row["pred_variance"] for row in errors]))
    target_variance = float(np.mean([row["target_variance"] for row in errors]))
    variance_ratio = float(pred_variance / max(target_variance, 1e-8))
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": correlation,
        "pred_variance": pred_variance,
        "target_variance": target_variance,
        "variance_ratio": variance_ratio,
    }


def validate_metrics_consistency(results_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "rmse",
        "mae",
        "bias",
        "correlation",
        "pred_variance",
        "target_variance",
        "variance_ratio",
        "num_samples",
        "history_length",
    ]
    tol = 1e-6

    for row in rows:
        model_name = str(row.get("model", ""))
        if not model_name:
            raise ValueError("Missing model name in baselines_summary.csv row")
        metrics_path = results_root / model_name / "metrics.json"
        if not metrics_path.exists():
            raise ValueError(f"Missing metrics.json for model '{model_name}' at {metrics_path}")

        metrics = json.loads(metrics_path.read_text())
        for key in fields:
            if key not in row or key not in metrics:
                raise ValueError(f"Missing '{key}' in metrics for model '{model_name}'")

            a = row[key]
            b = metrics[key]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not np.isfinite(float(a)) or not np.isfinite(float(b)):
                    raise ValueError(f"Non-finite '{key}' for model '{model_name}'")
                if abs(float(a) - float(b)) > tol:
                    raise ValueError("Metrics mismatch between JSON and CSV")
            else:
                if str(a) != str(b):
                    raise ValueError("Metrics mismatch between JSON and CSV")


def save_comparison_plot(
    era5_up: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    abs_error = np.abs(prediction - target)

    main_vmin = float(min(np.min(era5_up), np.min(prediction), np.min(target)))
    main_vmax = float(max(np.max(era5_up), np.max(prediction), np.max(target)))
    err_vmin = 0.0
    err_vmax = float(np.max(abs_error))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(era5_up, cmap="coolwarm", vmin=main_vmin, vmax=main_vmax)
    axes[0, 0].set_title("ERA5 Input (Upsampled)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prediction, cmap="coolwarm", vmin=main_vmin, vmax=main_vmax)
    axes[0, 1].set_title("Model Prediction")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(target, cmap="coolwarm", vmin=main_vmin, vmax=main_vmax)
    axes[1, 0].set_title("PRISM Target")
    axes[1, 0].axis("off")

    im3 = axes[1, 1].imshow(abs_error, cmap="magma", vmin=err_vmin, vmax=err_vmax)
    axes[1, 1].set_title("Absolute Error")
    axes[1, 1].axis("off")

    fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1], axes[1, 0]], shrink=0.82, label="Temperature (deg C)")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.82, label="|Error| (deg C)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_comparison(
    era5_up: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    ordered_models = sorted(model_predictions.keys())
    panels = ["ERA5 (Upsampled)"] + [f"{m}" for m in ordered_models] + ["PRISM Target"]
    arrays = [era5_up] + [model_predictions[m] for m in ordered_models] + [target]

    vmin = float(min(np.min(a) for a in arrays))
    vmax = float(max(np.max(a) for a in arrays))

    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 4), constrained_layout=True)
    if len(arrays) == 1:
        axes = [axes]

    im = None
    for ax, label, arr in zip(axes, panels, arrays):
        im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")

    fig.colorbar(im, ax=axes, shrink=0.85, label="Temperature (deg C)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_visual_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    output_dir: Path,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    abs_error = np.abs(prediction - target)

    vmin = float(min(np.min(prediction), np.min(target)))
    vmax = float(max(np.max(prediction), np.max(target)))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("PRISM Target")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    im = axes[2].imshow(abs_error, cmap="magma")
    axes[2].set_title("Absolute Error")
    axes[2].axis("off")

    fig.colorbar(im, ax=axes[2], shrink=0.85)
    fig.savefig(output_dir / "sample_prediction.png", dpi=150)
    plt.close(fig)

    fig_err, ax_err = plt.subplots(1, 1, figsize=(4.5, 4), constrained_layout=True)
    im_err = ax_err.imshow(abs_error, cmap="magma")
    ax_err.set_title("Absolute Error Map")
    ax_err.axis("off")
    fig_err.colorbar(im_err, ax=ax_err, shrink=0.85)
    fig_err.savefig(output_dir / "error_map.png", dpi=150)
    plt.close(fig_err)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run evaluation. Install dependencies with: pip install -r requirements.txt"
        )

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset

    apply_dataset_version_to_args(args)
    from models.baselines import fit_global_linear_baseline, upsample_latest_era5

    era5_path = Path(args.era5_path)
    prism_path = Path(args.prism_path)

    if not era5_path.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {era5_path}. Run data_pipeline/download_era5_georgia.py first."
        )
    if not prism_path.exists():
        raise FileNotFoundError(
            f"PRISM path not found: {prism_path}. Run data_pipeline/download_prism.py first."
        )

    set_seed(int(args.split_seed))
    device = resolve_device(args.device)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Loading ERA5 and PRISM data")
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=args.history_length,
        input_set=args.input_set,
        verbose=False,
    )
    stats = getattr(dataset, "summary_stats", {})
    candidate_dates = int(stats.get("candidate_dates", len(dataset)))
    usable_samples = len(dataset)
    print(
        "Dataset alignment summary: "
        f"candidate_dates={candidate_dates} usable_samples={usable_samples} "
        f"history_length={args.history_length}"
    )
    if usable_samples < 2:
        raise RuntimeError(
            build_insufficient_samples_message(
                history_length=args.history_length,
                usable_samples=usable_samples,
                min_required=2,
                candidate_dates=candidate_dates,
            )
        )

    linear_model = None
    if "linear" in args.models:
        # train_indices may come from checkpoint metadata; finalize after checkpoint load.
        linear_model = "__PENDING__"

    learned_models: Dict[str, Tuple[Any, Optional[Dict[str, List[float]]]]] = {}
    learned_split_signatures: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    checkpoint_paths = {
        "cnn": Path(args.cnn_checkpoint),
        "unet": Path(args.unet_checkpoint),
        "convlstm": Path(args.convlstm_checkpoint),
    }

    for model_name in [m for m in args.models if m in ("cnn", "unet", "convlstm")]:
        ckpt_path = checkpoint_paths[model_name]
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: checkpoint not found at {ckpt_path}")
            continue

        model, ckpt_history, input_norm, checkpoint_input_set, ckpt_train_indices, ckpt_val_indices = load_checkpoint_model(
            model_name, ckpt_path, device
        )
        if ckpt_history != args.history_length:
            print(
                f"Skipping {model_name}: checkpoint history_length={ckpt_history} does not match requested {args.history_length}"
            )
            continue
        if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
            print(
                f"Skipping {model_name}: checkpoint input_set={checkpoint_input_set} does not match requested {args.input_set}"
            )
            continue
        learned_models[model_name] = (model, input_norm)
        if ckpt_train_indices is not None and ckpt_val_indices is not None:
            signature = (
                tuple(int(i) for i in ckpt_train_indices),
                tuple(int(i) for i in ckpt_val_indices),
            )
            learned_split_signatures.append(signature)

    if learned_split_signatures:
        base_signature = learned_split_signatures[0]
        if any(sig != base_signature for sig in learned_split_signatures[1:]):
            raise RuntimeError("Loaded checkpoints use different train/validation splits")
        train_indices = list(base_signature[0])
        val_indices = list(base_signature[1])
        if not train_indices or not val_indices:
            raise RuntimeError("Checkpoint split metadata is empty")
        if max(train_indices + val_indices) >= len(dataset) or min(train_indices + val_indices) < 0:
            raise RuntimeError("Checkpoint split metadata is out of bounds for the current dataset")
        print("Using train/validation split stored in checkpoint metadata")
    else:
        # Research-grade consistency: if evaluating learned models, require split metadata
        # from checkpoints rather than implicitly generating a split.
        requested_learned = any(m in ("cnn", "unet", "convlstm") for m in args.models)
        if requested_learned and learned_models:
            raise RuntimeError(
                "Learned model evaluation requires train/val split metadata in the checkpoint(s). "
                "Re-train with training/train_downscaler.py and ensure checkpoints include train_indices/val_indices."
            )
        train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.split_seed)
        print("Using split generated from --val-fraction and --split-seed (baselines only)")

    eval_indices = val_indices[: max(1, min(args.num_samples, len(val_indices)))]
    if linear_model == "__PENDING__":
        print("Fitting linear baseline on training split")
        linear_model = fit_global_linear_baseline(dataset, train_indices)
        print(f"Linear baseline coefficients: slope={linear_model.slope:.6f}, intercept={linear_model.intercept:.6f}")

    model_metrics_rows: List[Dict[str, Any]] = []
    comparison_predictions: Dict[str, np.ndarray] = {}
    comparison_era5: Optional[np.ndarray] = None
    comparison_target: Optional[np.ndarray] = None
    comparison_date: Optional[str] = None

    for model_name in args.models:
        if model_name in ("cnn", "unet", "convlstm") and model_name not in learned_models:
            continue

        errors: List[Dict[str, float]] = []
        saved_plot = False

        model_dir = results_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for stale_plot in model_dir.glob("comparison_*.png"):
            stale_plot.unlink(missing_ok=True)

        with torch.no_grad():
            for sample_idx in eval_indices:
                x, y = dataset[sample_idx]
                x = x.unsqueeze(0).to(device)
                y = y.unsqueeze(0).to(device)

                if model_name == "persistence":
                    pred = upsample_latest_era5(x, target_size=(y.shape[-2], y.shape[-1]))
                elif model_name == "era5_upsampled":
                    # Explicit baseline name for "upsampled ERA5" comparison.
                    pred = upsample_latest_era5(x, target_size=(y.shape[-2], y.shape[-1]))
                elif model_name == "linear":
                    if linear_model is None:
                        raise RuntimeError("Linear baseline requested but not fitted")
                    pred = linear_model.predict(x, target_size=(y.shape[-2], y.shape[-1]))
                else:
                    model, input_norm = learned_models[model_name]
                    raw_pred = model(
                        normalize_input_batch(x, input_norm),
                        target_size=(y.shape[-2], y.shape[-1]),
                    )
                    target_mode = getattr(model, "target_mode", args.target_mode)
                    if target_mode == "residual":
                        if model_name == "convlstm":
                            raise RuntimeError("Residual target mode is not supported for ConvLSTM checkpoints")
                        pred = upsample_latest_era5(x, target_size=(y.shape[-2], y.shape[-1])) + raw_pred
                    else:
                        pred = raw_pred

                if pred.shape != y.shape:
                    raise RuntimeError(
                        f"Prediction/target shape mismatch: pred={tuple(pred.shape)} y={tuple(y.shape)}"
                    )
                err = pred - y
                rmse = float(torch.sqrt(torch.mean(err ** 2)).item())
                mae = float(torch.mean(torch.abs(err)).item())
                bias = float(torch.mean(err).item())
                pred_flat = pred.reshape(-1).detach().cpu().numpy()
                target_flat = y.reshape(-1).detach().cpu().numpy()
                correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1]) if pred_flat.size > 1 else 0.0
                if not np.isfinite(correlation):
                    correlation = 0.0
                pred_variance = float(np.var(pred_flat))
                target_variance = float(np.var(target_flat))
                errors.append(
                    {
                        "rmse": rmse,
                        "mae": mae,
                        "bias": bias,
                        "correlation": correlation,
                        "pred_variance": pred_variance,
                        "target_variance": target_variance,
                    }
                )

                if not saved_plot:
                    date = dataset.metadata(sample_idx).date.strftime("%Y%m%d")
                    era5_up = upsample_latest_era5(x, target_size=(y.shape[-2], y.shape[-1])).squeeze().cpu().numpy()
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y.squeeze().cpu().numpy()

                    plot_path = model_dir / f"comparison_1_{date}.png"
                    save_comparison_plot(
                        era5_up=era5_up,
                        prediction=pred_np,
                        target=target_np,
                        output_path=plot_path,
                        title=f"{model_name.upper()} | ERA5->PRISM | {date} | history={args.history_length}",
                    )
                    saved_plot = True

                    comparison_predictions[model_name] = pred_np
                    comparison_era5 = era5_up
                    comparison_target = target_np
                    comparison_date = date

        if not errors:
            continue

        summary = compute_metrics(errors)
        summary.update(
            {
                "model": model_name,
                "num_samples": len(errors),
                "history_length": int(args.history_length),
            }
        )
        model_metrics_rows.append(summary)

        metrics_path = model_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2))
        print(
            f"{model_name:>11} | RMSE={summary['rmse']:.4f} "
            f"MAE={summary['mae']:.4f} BIAS={summary['bias']:.4f} CORR={summary['correlation']:.4f} "
            f"VAR_RATIO={summary['variance_ratio']:.4f}"
        )
        if summary["variance_ratio"] < 0.15:
            print(f"Warning: {model_name} predictions show very low spatial variance relative to target.")

    if not model_metrics_rows:
        raise RuntimeError("No models were evaluated. Provide valid checkpoints or baseline selections.")

    summary_csv = results_root / "baselines_summary.csv"
    with summary_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "model",
                "rmse",
                "mae",
                "bias",
                "correlation",
                "pred_variance",
                "target_variance",
                "variance_ratio",
                "num_samples",
                "history_length",
            ],
        )
        writer.writeheader()
        writer.writerows(model_metrics_rows)

    validate_metrics_consistency(results_root, model_metrics_rows)

    # Baseline comparison check: learned models should improve over persistence baseline.
    rmse_by_model = {str(r["model"]): float(r["rmse"]) for r in model_metrics_rows if "model" in r and "rmse" in r}
    if "persistence" in rmse_by_model:
        base = rmse_by_model["persistence"]
        for learned in ("cnn", "unet", "convlstm"):
            if learned in rmse_by_model and not (rmse_by_model[learned] < base):
                msg = (
                    f"{learned} did not improve over persistence baseline: "
                    f"rmse={rmse_by_model[learned]:.4f} baseline={base:.4f}"
                )
                if args.require_improvement:
                    raise RuntimeError(msg)
                print(f"Warning: {msg}")

    if comparison_era5 is not None and comparison_target is not None and comparison_predictions:
        comparison_plot = results_root / "model_comparison.png"
        save_model_comparison(
            era5_up=comparison_era5,
            model_predictions=comparison_predictions,
            target=comparison_target,
            output_path=comparison_plot,
            title=f"Model Comparison | ERA5->PRISM | {comparison_date}",
        )

        preferred_model = "convlstm" if "convlstm" in comparison_predictions else sorted(comparison_predictions.keys())[0]
        save_visual_diagnostics(
            prediction=comparison_predictions[preferred_model],
            target=comparison_target,
            output_dir=results_root,
        )

    print(f"Saved evaluation summary CSV: {summary_csv}")
    print(f"Saved evaluation outputs under: {results_root}")


if __name__ == "__main__":
    main()
