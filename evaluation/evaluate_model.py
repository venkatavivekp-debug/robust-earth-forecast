from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    parser.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    parser.add_argument("--prism-path", type=str, default="data_raw/prism")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["persistence", "linear", "cnn", "convlstm"],
        choices=["persistence", "linear", "cnn", "convlstm"],
        help="Models/baselines to evaluate",
    )
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cnn-checkpoint", type=str, default="checkpoints/cnn_best.pt")
    parser.add_argument("--convlstm-checkpoint", type=str, default="checkpoints/convlstm_best.pt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--results-dir", type=str, default="results/evaluation")
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


def split_indices(n_total: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    if n_total < 2:
        raise ValueError("At least 2 samples are required to build train/validation splits")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    val_size = max(1, int(round(n_total * val_fraction)))
    if val_size >= n_total:
        val_size = n_total - 1
    train_size = n_total - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n_total))
    return train_indices, val_indices


def load_checkpoint_model(model_name: str, checkpoint_path: Path, device: Any) -> Tuple[Any, int]:
    from models.cnn_downscaler import CNNDownscaler
    from models.convlstm_downscaler import ConvLSTMDownscaler

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
    model.to(device)
    model.eval()
    return model, history_length


def compute_metrics(errors: Sequence[Dict[str, float]]) -> Dict[str, float]:
    rmse = float(np.mean([row["rmse"] for row in errors]))
    mae = float(np.mean([row["mae"] for row in errors]))
    bias = float(np.mean([row["bias"] for row in errors]))
    return {"rmse": rmse, "mae": mae, "bias": bias}


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


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run evaluation. Install dependencies with: pip install -r requirements.txt"
        )

    from datasets.prism_dataset import ERA5_PRISM_Dataset
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

    device = resolve_device(args.device)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Loading ERA5 and PRISM data")
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=args.history_length,
    )

    train_indices, val_indices = split_indices(len(dataset), args.val_fraction)
    eval_indices = val_indices[: max(1, min(args.num_samples, len(val_indices)))]

    linear_model = None
    if "linear" in args.models:
        print("Fitting linear baseline on training split")
        linear_model = fit_global_linear_baseline(dataset, train_indices)
        print(f"Linear baseline coefficients: slope={linear_model.slope:.6f}, intercept={linear_model.intercept:.6f}")

    learned_models: Dict[str, Any] = {}
    checkpoint_paths = {
        "cnn": Path(args.cnn_checkpoint),
        "convlstm": Path(args.convlstm_checkpoint),
    }

    for model_name in [m for m in args.models if m in ("cnn", "convlstm")]:
        ckpt_path = checkpoint_paths[model_name]
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: checkpoint not found at {ckpt_path}")
            continue

        model, ckpt_history = load_checkpoint_model(model_name, ckpt_path, device)
        if ckpt_history != args.history_length:
            print(
                f"Skipping {model_name}: checkpoint history_length={ckpt_history} does not match requested {args.history_length}"
            )
            continue
        learned_models[model_name] = model

    model_metrics_rows: List[Dict[str, Any]] = []
    comparison_predictions: Dict[str, np.ndarray] = {}
    comparison_era5: Optional[np.ndarray] = None
    comparison_target: Optional[np.ndarray] = None
    comparison_date: Optional[str] = None

    for model_name in args.models:
        if model_name in ("cnn", "convlstm") and model_name not in learned_models:
            continue

        errors: List[Dict[str, float]] = []
        saved_plot = False

        model_dir = results_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for sample_idx in eval_indices:
                x, y = dataset[sample_idx]
                x = x.unsqueeze(0).to(device)
                y = y.unsqueeze(0).to(device)

                if model_name == "persistence":
                    pred = upsample_latest_era5(x, target_size=(y.shape[-2], y.shape[-1]))
                elif model_name == "linear":
                    if linear_model is None:
                        raise RuntimeError("Linear baseline requested but not fitted")
                    pred = linear_model.predict(x, target_size=(y.shape[-2], y.shape[-1]))
                else:
                    pred = learned_models[model_name](x, target_size=(y.shape[-2], y.shape[-1]))

                err = pred - y
                rmse = float(torch.sqrt(torch.mean(err ** 2)).item())
                mae = float(torch.mean(torch.abs(err)).item())
                bias = float(torch.mean(err).item())
                errors.append({"rmse": rmse, "mae": mae, "bias": bias})

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
            f"MAE={summary['mae']:.4f} BIAS={summary['bias']:.4f}"
        )

    if not model_metrics_rows:
        raise RuntimeError("No models were evaluated. Provide valid checkpoints or baseline selections.")

    summary_csv = results_root / "metrics_summary.csv"
    with summary_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model", "rmse", "mae", "bias", "num_samples", "history_length"])
        writer.writeheader()
        writer.writerows(model_metrics_rows)

    if comparison_era5 is not None and comparison_target is not None and comparison_predictions:
        comparison_plot = results_root / "model_comparison.png"
        save_model_comparison(
            era5_up=comparison_era5,
            model_predictions=comparison_predictions,
            target=comparison_target,
            output_path=comparison_plot,
            title=f"Model Comparison | ERA5->PRISM | {comparison_date}",
        )

    print(f"Saved evaluation summary CSV: {summary_csv}")
    print(f"Saved evaluation outputs under: {results_root}")


if __name__ == "__main__":
    main()
