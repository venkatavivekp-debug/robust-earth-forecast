from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Any, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    Subset = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal ERA5->PRISM CNN downscaler baseline")
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
        help="Path to PRISM raster file or directory of daily rasters",
    )
    parser.add_argument("--history-length", type=int, default=3, help="Number of ERA5 timesteps [t-k+1 ... t]")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="checkpoints/cnn_downscaler_best.pt",
        help="Output path for best model checkpoint",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def split_dataset(dataset: Any, val_fraction: float) -> Tuple[Any, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("At least 2 aligned samples are required for train/validation split")

    val_size = max(1, int(round(n_total * val_fraction)))
    if val_size >= n_total:
        val_size = n_total - 1

    train_size = n_total - val_size
    # Chronological split keeps this baseline deterministic and simple.
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n_total))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    train: bool,
) -> float:
    model.train(mode=train)
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            preds = model(x, target_size=(y.shape[-2], y.shape[-1]))
            loss = criterion(preds, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(loader))


def main() -> None:
    args = parse_args()
    if torch is None or nn is None or DataLoader is None or Subset is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run training. Install dependencies with: pip install -r requirements.txt"
        )

    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from models.cnn_downscaler import CNNDownscaler

    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Training started")

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

    print("Loading ERA5 and PRISM data")
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=args.history_length,
    )
    train_set, val_set = split_dataset(dataset, args.val_fraction)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample_x, sample_y = dataset[0]
    model = CNNDownscaler(in_channels=sample_x.shape[0], out_channels=sample_y.shape[0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(
        f"Dataset: total={len(dataset)} train={len(train_set)} val={len(val_set)} "
        f"input_shape={tuple(sample_x.shape)} target_shape={tuple(sample_y.shape)}"
    )
    print(f"History length: {args.history_length}")
    print(f"Device: {device.type} | Seed: {args.seed}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            # Keep one best checkpoint for reproducible evaluation.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_channels": sample_x.shape[0],
                        "out_channels": sample_y.shape[0],
                        "base_channels": 32,
                    },
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )

        marker = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} {marker}"
        )

    print(f"Best checkpoint saved to: {checkpoint_path} (val_loss={best_val_loss:.6f})")


if __name__ == "__main__":
    main()
