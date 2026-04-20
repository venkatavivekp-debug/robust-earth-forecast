from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Subset = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERA5->PRISM downscaling models (CNN or ConvLSTM)")
    parser.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    parser.add_argument("--prism-path", type=str, default="data_raw/prism")
    parser.add_argument("--input-set", type=str, choices=["t2m", "core4", "extended"], default="extended")
    parser.add_argument("--model", type=str, choices=["cnn", "convlstm"], default="convlstm")
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
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
        default="results/training_logs",
        help="Directory for training logs and curves",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run label for log artifacts")
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


def split_dataset(dataset: Any, val_fraction: float, split_seed: int) -> Tuple[Any, Any, List[int], List[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("At least 2 aligned samples are required for train/validation split")

    val_size = max(1, int(round(n_total * val_fraction)))
    if val_size >= n_total:
        val_size = n_total - 1

    all_indices = np.arange(n_total)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(all_indices)

    val_indices = sorted(all_indices[:val_size].tolist())
    train_indices = sorted(all_indices[val_size:].tolist())
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


def compute_input_stats(dataset: Any, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not indices:
        raise ValueError("Cannot compute input normalization stats with empty indices")

    sample_x, _ = dataset[int(indices[0])]
    if sample_x.dim() != 4:
        raise ValueError(f"Expected sample input shape [T, C, H, W], got {tuple(sample_x.shape)}")

    channel_count = int(sample_x.shape[1])
    sum_x = np.zeros(channel_count, dtype=np.float64)
    sum_x2 = np.zeros(channel_count, dtype=np.float64)
    n_values = 0

    for idx in indices:
        x, _ = dataset[int(idx)]
        x_np = x.numpy().astype(np.float64)
        x_flat = np.transpose(x_np, (1, 0, 2, 3)).reshape(channel_count, -1)
        sum_x += x_flat.sum(axis=1)
        sum_x2 += (x_flat ** 2).sum(axis=1)
        n_values += x_flat.shape[1]

    mean = sum_x / max(1, n_values)
    var = np.maximum((sum_x2 / max(1, n_values)) - (mean ** 2), 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_input_batch(
    x: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mean is None or std is None:
        return x

    if x.dim() == 5:
        return (x - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)
    if x.dim() == 4:
        return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    raise ValueError(f"Unsupported input tensor shape for normalization: {tuple(x.shape)}")


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    l1_weight: float,
    optimizer: Any,
    device: Any,
    train: bool,
    input_mean: Optional[torch.Tensor],
    input_std: Optional[torch.Tensor],
    grad_clip: Optional[float],
) -> Tuple[float, Optional[float], Optional[float]]:
    model.train(mode=train)
    running_loss = 0.0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_steps = 0

    for x, y in loader:
        if x.dim() != 5 or y.dim() != 4:
            raise RuntimeError(
                f"Unexpected batch shapes: input={tuple(x.shape)} target={tuple(y.shape)}"
            )
        x = x.to(device)
        y = y.to(device)
        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite values detected in training inputs")
        if not torch.isfinite(y).all():
            raise RuntimeError("Non-finite values detected in training targets")
        x_model = normalize_input_batch(x, input_mean, input_std)

        with torch.set_grad_enabled(train):
            preds = model(x_model, target_size=(y.shape[-2], y.shape[-1]))
            if not torch.isfinite(preds).all():
                raise RuntimeError("Model produced non-finite predictions")
            mse_loss = criterion(preds, y)
            if l1_weight > 0.0:
                l1_loss = F.l1_loss(preds, y)
                loss = mse_loss + float(l1_weight) * l1_loss
            else:
                loss = mse_loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite loss encountered during training")

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                    )
                else:
                    sq_norm = torch.zeros(1, device=device)
                    for param in model.parameters():
                        if param.grad is not None:
                            sq_norm = sq_norm + torch.sum(param.grad.detach() ** 2)
                    grad_norm = float(torch.sqrt(sq_norm).item())
                if not np.isfinite(grad_norm):
                    raise RuntimeError("Non-finite gradient norm encountered")
                grad_norm_sum += grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
                grad_norm_steps += 1
                optimizer.step()

        running_loss += float(loss.item())

    mean_loss = running_loss / max(1, len(loader))
    if not train:
        return mean_loss, None, None

    grad_norm_mean = grad_norm_sum / max(1, grad_norm_steps)
    return mean_loss, grad_norm_mean, grad_norm_max


def save_training_curve(curve_rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "train_grad_norm_mean",
        "train_grad_norm_max",
        "is_best",
    ]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)


def save_training_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def save_loss_curve_plot(curve_rows: Sequence[dict], output_path: Path) -> None:
    if not curve_rows:
        return

    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

    import matplotlib
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in curve_rows]
    train_losses = [float(row["train_loss"]) for row in curve_rows]
    val_losses = [float(row["val_loss"]) for row in curve_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_losses, label="train_loss", linewidth=1.8)
    ax.plot(epochs, val_losses, label="val_loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=args.history_length,
        input_set=args.input_set,
    )
    stats = getattr(dataset, "summary_stats", {})
    candidate_dates = int(stats.get("candidate_dates", len(dataset)))
    usable_samples = len(dataset)
    if usable_samples < 2:
        raise RuntimeError(
            build_insufficient_samples_message(
                history_length=args.history_length,
                usable_samples=usable_samples,
                min_required=2,
                candidate_dates=candidate_dates,
            )
        )

    train_set, val_set, train_indices, val_indices = split_dataset(dataset, args.val_fraction, args.split_seed)

    input_mean_np, input_std_np = compute_input_stats(dataset, train_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.scheduler_factor),
            patience=int(args.scheduler_patience),
            min_lr=1e-6,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=1e-6,
        )
    else:
        scheduler = None

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    curve_rows: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_grad_norm_mean, train_grad_norm_max = run_epoch(
            model,
            train_loader,
            criterion,
            args.l1_weight,
            optimizer,
            device,
            train=True,
            input_mean=input_mean,
            input_std=input_std,
            grad_clip=args.grad_clip,
        )
        val_loss, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            args.l1_weight,
            optimizer,
            device,
            train=False,
            input_mean=input_mean,
            input_std=input_std,
            grad_clip=args.grad_clip,
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
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
                    "input_norm": {
                        "mean": input_mean_np.tolist(),
                        "std": input_std_np.tolist(),
                    },
                    "args": vars(args),
                },
                checkpoint_path,
            )

        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "lr": f"{current_lr:.8f}",
                "train_grad_norm_mean": f"{(train_grad_norm_mean or 0.0):.8f}",
                "train_grad_norm_max": f"{(train_grad_norm_max or 0.0):.8f}",
                "is_best": int(improved),
            }
        )

        if train_grad_norm_mean is not None and train_grad_norm_max is not None:
            print(
                f"grad_norm_mean={float(train_grad_norm_mean):.6f} "
                f"grad_norm_max={float(train_grad_norm_max):.6f}"
            )

    artifact_dir = Path(args.training_results_dir)
    run_name = args.run_name or args.model
    curve_path = artifact_dir / f"{run_name}_training_log.csv"
    summary_path = artifact_dir / f"{run_name}_training_log.json"
    plot_path = artifact_dir / f"{run_name}_loss_curve.png"

    save_training_curve(curve_rows, curve_path)
    save_loss_curve_plot(curve_rows, plot_path)
    save_training_summary(
        {
            "run_name": run_name,
            "model": args.model,
            "input_set": args.input_set,
            "history_length": int(args.history_length),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "l1_weight": float(args.l1_weight),
            "grad_clip": float(args.grad_clip),
            "scheduler": args.scheduler,
            "split_seed": int(args.split_seed),
            "seed": int(args.seed),
            "checkpoint_path": str(checkpoint_path),
        },
        summary_path,
    )

    # Training artifacts are written to disk; avoid verbose stdout.


if __name__ == "__main__":
    main()
