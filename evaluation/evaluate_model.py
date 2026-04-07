from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Force headless-safe plotting on macOS/Linux terminals and CI.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal ERA5->PRISM CNN downscaler")
    parser.add_argument(
        "--era5-path",
        type=str,
        default="data_raw/era5_georgia_temp.nc",
        help="Path to ERA5 NetCDF file",
    )
    parser.add_argument(
        "--prism-path",
        type=str,
        default="data_raw/prism",
        help="Path to PRISM raster file or directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/cnn_downscaler_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Temporal history length; if omitted, inferred from checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=8, help="Max samples to evaluate")
    parser.add_argument("--num-plots", type=int, default=1, help="Number of comparison plots to save")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/evaluation",
        help="Directory for metrics and plots",
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


def load_model(checkpoint_path: Path, device: Any) -> Tuple[Any, int]:
    from models.cnn_downscaler import CNNDownscaler

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", {})
        in_channels = int(model_config.get("in_channels", 1))
        model = CNNDownscaler(
            in_channels=in_channels,
            out_channels=int(model_config.get("out_channels", 1)),
            base_channels=int(model_config.get("base_channels", 32)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CNNDownscaler()
        model.load_state_dict(checkpoint)
        in_channels = 1

    model.to(device)
    model.eval()
    return model, in_channels


def save_comparison_plot(
    era5_input: Any,
    prediction: Any,
    target: Any,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    # Temporal baseline visualization uses the most recent ERA5 frame at t.
    era5_last = era5_input[-1].unsqueeze(0).unsqueeze(0)
    era5_up = F.interpolate(
        era5_last,
        size=(target.shape[-2], target.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    era5_np = era5_up.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    vmin = float(min(np.min(era5_np), np.min(pred_np), np.min(target_np)))
    vmax = float(max(np.max(era5_np), np.max(pred_np), np.max(target_np)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(era5_np, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("ERA5 t (Upsampled)")
    axes[0].axis("off")

    axes[1].imshow(pred_np, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("CNN Prediction")
    axes[1].axis("off")

    axes[2].imshow(target_np, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[2].set_title("PRISM Target")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.colorbar(im0, ax=axes, shrink=0.8, label="Temperature (deg C)")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if torch is None or F is None or DataLoader is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run evaluation. Install dependencies with: pip install -r requirements.txt"
        )

    from datasets.prism_dataset import ERA5_PRISM_Dataset

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Loading ERA5 and PRISM data for evaluation")

    device = resolve_device(args.device)
    era5_path = Path(args.era5_path)
    prism_path = Path(args.prism_path)
    checkpoint_path = Path(args.checkpoint_path)

    if not era5_path.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {era5_path}. Run data_pipeline/download_era5_georgia.py first."
        )
    if not prism_path.exists():
        raise FileNotFoundError(
            f"PRISM path not found: {prism_path}. Run data_pipeline/download_prism.py first."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training/train_downscaler.py first."
        )

    model, checkpoint_history = load_model(checkpoint_path, device)
    history_length = args.history_length if args.history_length is not None else checkpoint_history

    dataset = ERA5_PRISM_Dataset(
        str(era5_path),
        str(prism_path),
        history_length=history_length,
    )

    if len(dataset) < 1:
        raise RuntimeError("No samples available for evaluation")

    sample_x, _ = dataset[0]
    if sample_x.shape[0] != checkpoint_history:
        raise RuntimeError(
            f"History mismatch: dataset uses {sample_x.shape[0]} channels but checkpoint expects {checkpoint_history}. "
            "Use --history-length to match training configuration."
        )

    n_eval = min(args.num_samples, len(dataset))
    eval_indices = list(range(n_eval))
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)
    loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False)

    rmse_values: List[float] = []
    mae_values: List[float] = []
    plots_saved = 0
    global_index = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x, target_size=(y.shape[-2], y.shape[-1]))

            sample_rmse = torch.sqrt(torch.mean((pred - y) ** 2, dim=(1, 2, 3)))
            sample_mae = torch.mean(torch.abs(pred - y), dim=(1, 2, 3))

            rmse_values.extend(sample_rmse.cpu().tolist())
            mae_values.extend(sample_mae.cpu().tolist())

            for i in range(x.shape[0]):
                if global_index >= n_eval:
                    break

                if plots_saved < args.num_plots:
                    date = dataset.metadata(eval_indices[global_index]).date
                    plot_path = results_dir / f"comparison_{plots_saved + 1}_{date.strftime('%Y%m%d')}.png"

                    save_comparison_plot(
                        era5_input=x[i].cpu(),
                        prediction=pred[i].cpu(),
                        target=y[i].cpu(),
                        output_path=plot_path,
                        title=f"Temporal ERA5->PRISM | {date.date()} | history={history_length}",
                    )
                    plots_saved += 1

                global_index += 1

    metrics: Dict[str, object] = {
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "num_samples": int(n_eval),
        "history_length": int(history_length),
    }

    print("Saving evaluation results")
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Evaluated samples: {n_eval}")
    print(f"History length: {history_length}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    if plots_saved:
        print(f"Saved {plots_saved} comparison plot(s) under: {results_dir}")


if __name__ == "__main__":
    main()
