from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.cnn_downscaler import CNNDownscaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ERA5->PRISM CNN downscaler")
    parser.add_argument("--era5-path", type=str, required=True, help="Path to ERA5 NetCDF file")
    parser.add_argument(
        "--prism-path",
        type=str,
        required=True,
        help="Path to PRISM raster file or directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
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


def resolve_device(device_arg: str) -> torch.device:
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


def load_model(checkpoint_path: Path, device: torch.device) -> CNNDownscaler:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", {})
        model = CNNDownscaler(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 1),
            base_channels=model_config.get("base_channels", 32),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CNNDownscaler()
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def save_comparison_plot(
    era5_input: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    era5_up = F.interpolate(
        era5_input.unsqueeze(0),
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
    axes[0].set_title("ERA5 Input (Upsampled)")
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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = ERA5_PRISM_Dataset(args.era5_path, args.prism_path)
    n_eval = min(args.num_samples, len(dataset))

    if n_eval < 1:
        raise RuntimeError("No samples available for evaluation")

    eval_indices = list(range(n_eval))
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)
    loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False)

    model = load_model(checkpoint_path, device)

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
                    plot_path = (
                        results_dir
                        / f"comparison_{plots_saved + 1}_{date.strftime('%Y%m%d')}.png"
                    )
                    save_comparison_plot(
                        era5_input=x[i].cpu(),
                        prediction=pred[i].cpu(),
                        target=y[i].cpu(),
                        output_path=plot_path,
                        title=f"ERA5->PRISM Downscaling | {date.date()}",
                    )
                    plots_saved += 1

                global_index += 1

    metrics: Dict[str, object] = {
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "num_samples": int(n_eval),
    }

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Evaluated samples: {n_eval}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    if plots_saved:
        print(f"Saved {plots_saved} comparison plot(s) under: {results_dir}")


if __name__ == "__main__":
    main()
