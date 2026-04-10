from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys
from typing import Any, List, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Train ERA5->PRISM downscaling models (CNN or ConvLSTM)")
    parser.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    parser.add_argument("--prism-path", type=str, default="data_raw/prism")
    parser.add_argument("--model", type=str, choices=["cnn", "convlstm"], default="convlstm")
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=32, help="Hidden channels for ConvLSTM")
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
        default=None,
        help="Optional checkpoint output path. Defaults to checkpoints/<model>_best.pt",
    )
    parser.add_argument(
        "--training-results-dir",
        type=str,
        default="results/training",
        help="Directory for training curve CSV",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def split_dataset(dataset: Any, val_fraction: float) -> Tuple[Any, Any, List[int], List[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("At least 2 aligned samples are required for train/validation split")

    val_size = max(1, int(round(n_total * val_fraction)))
    if val_size >= n_total:
        val_size = n_total - 1

    train_size = n_total - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n_total))
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        train_indices,
        val_indices,
    )


def recommended_prism_days(history_length: int, min_samples: int) -> int:
    # Usable samples are roughly: n_dates - history_length + 1
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


def build_model(args: argparse.Namespace, sample_x: Any, sample_y: Any) -> Tuple[Any, dict]:
    from models.cnn_downscaler import CNNDownscaler
    from models.convlstm_downscaler import ConvLSTMDownscaler

    if args.model == "cnn":
        in_channels = int(sample_x.shape[0] * sample_x.shape[1])
        model = CNNDownscaler(in_channels=in_channels, out_channels=int(sample_y.shape[0]))
        model_config = {
            "in_channels": in_channels,
            "out_channels": int(sample_y.shape[0]),
            "base_channels": 32,
        }
        return model, model_config

    model = ConvLSTMDownscaler(
        input_channels=int(sample_x.shape[1]),
        hidden_channels=int(args.hidden_channels),
        out_channels=int(sample_y.shape[0]),
    )
    model_config = {
        "input_channels": int(sample_x.shape[1]),
        "hidden_channels": int(args.hidden_channels),
        "out_channels": int(sample_y.shape[0]),
        "kernel_size": 3,
    }
    return model, model_config


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


def save_training_curve(curve_rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "is_best"]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)


def main() -> None:
    args = parse_args()
    if torch is None or nn is None or DataLoader is None or Subset is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run training. Install dependencies with: pip install -r requirements.txt"
        )

    from datasets.prism_dataset import ERA5_PRISM_Dataset

    set_seed(args.seed)
    device = resolve_device(args.device)

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

    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else Path(f"checkpoints/{args.model}_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training started")
    print(f"Model: {args.model}")
    print(f"History length: {args.history_length}")
    print(f"Device: {device.type} | Seed: {args.seed}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=args.history_length,
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

    train_set, val_set, train_indices, val_indices = split_dataset(dataset, args.val_fraction)

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
    model, model_config = build_model(args, sample_x, sample_y)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    print(
        f"Dataset: total={len(dataset)} train={len(train_set)} val={len(val_set)} "
        f"input_shape={tuple(sample_x.shape)} target_shape={tuple(sample_y.shape)}"
    )

    best_val_loss = float("inf")
    curve_rows: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "history_length": int(args.history_length),
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "args": vars(args),
                },
                checkpoint_path,
            )

        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "is_best": int(improved),
            }
        )

        marker = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} {marker}"
        )

    curves_dir = Path(args.training_results_dir)
    curve_path = curves_dir / f"{args.model}_training_curve.csv"
    save_training_curve(curve_rows, curve_path)

    print(f"Best checkpoint saved to: {checkpoint_path} (val_loss={best_val_loss:.6f})")
    print(f"Training curve saved to: {curve_path}")


if __name__ == "__main__":
    main()
