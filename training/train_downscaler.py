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

PLAIN_ENCODER_DECODER_ALIASES = {"cnn", "plain_encoder_decoder"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERA5->PRISM downscaling models")
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Select ERA5/PRISM paths from datasets/<version>/paths.json when paths are omitted",
    )
    parser.add_argument(
        "--era5-path",
        type=str,
        default=None,
        help="ERA5 NetCDF path (default: from --dataset-version)",
    )
    parser.add_argument(
        "--prism-path",
        type=str,
        default=None,
        help="PRISM directory path (default: from --dataset-version)",
    )
    parser.add_argument(
        "--input-set",
        type=str,
        choices=["t2m", "core4", "extended", "core4_elev", "core4_topo"],
        default="extended",
    )
    parser.add_argument(
        "--static-covariate-path",
        type=str,
        default=None,
        help="Optional DEM-derived static covariate NetCDF for core4_elev/core4_topo input sets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "plain_encoder_decoder", "unet", "convlstm"],
        default="convlstm",
    )
    parser.add_argument("--target-mode", type=str, choices=["direct", "residual"], default="direct")
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["mse", "mse_l1", "mse_grad", "mse_l1_grad"],
        default="mse_l1",
        help="Training objective. Default preserves the existing MSE + small L1 setup.",
    )
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if val_loss does not improve for N epochs (0 disables).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=32, help="Hidden channels for ConvLSTM")
    parser.add_argument("--unet-base-channels", type=int, default=24, help="Base channels for U-Net diagnostics")
    parser.add_argument(
        "--unet-padding-mode",
        type=str,
        choices=["reflection", "zero", "replicate"],
        default="reflection",
        help="Boundary padding used by the U-Net convolution blocks",
    )
    parser.add_argument(
        "--unet-upsampling-mode",
        type=str,
        choices=["bilinear", "convtranspose"],
        default="bilinear",
        help="U-Net decoder upsampling path",
    )
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    from models.unet_downscaler import UNetDownscaler

    if args.target_mode == "residual" and args.model == "convlstm":
        raise ValueError("target-mode residual is only supported for plain_encoder_decoder/cnn/unet")

    if args.model in (*PLAIN_ENCODER_DECODER_ALIASES, "unet"):
        in_channels = int(sample_x.shape[0] * sample_x.shape[1])
        if args.model in PLAIN_ENCODER_DECODER_ALIASES:
            base_channels = 32
            model = CNNDownscaler(in_channels=in_channels, out_channels=int(sample_y.shape[0]))
        else:
            base_channels = int(args.unet_base_channels)
            model = UNetDownscaler(
                in_channels=in_channels,
                out_channels=int(sample_y.shape[0]),
                base_channels=base_channels,
                padding_mode=str(args.unet_padding_mode),
                upsample_mode=str(args.unet_upsampling_mode),
            )
        model_config = {
            "in_channels": in_channels,
            "out_channels": int(sample_y.shape[0]),
            "base_channels": base_channels,
        }
        if args.model == "unet":
            model_config["padding_mode"] = str(args.unet_padding_mode)
            model_config["upsample_mode"] = str(args.unet_upsampling_mode)
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
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Non-finite mean/std encountered while computing input normalization stats")
    if (std <= 0).any():
        raise RuntimeError("Non-positive std encountered while computing input normalization stats")
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_input_batch(
    x: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mean is None or std is None:
        return x

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("Non-finite mean/std provided for normalization")
    std = std.clamp(min=1e-6)

    if x.dim() == 5:
        out = (x - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)
    elif x.dim() == 4:
        out = (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Unsupported input tensor shape for normalization: {tuple(x.shape)}")

    if not torch.isfinite(out).all():
        raise RuntimeError("Non-finite values produced by input normalization")
    return out


def residual_base(x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    if x.dim() == 5:
        latest = x[:, -1, 0:1, :, :]
    elif x.dim() == 4:
        latest = x[:, 0:1, :, :]
    else:
        raise ValueError(f"Unsupported ERA5 tensor shape for residual base: {tuple(x.shape)}")
    return F.interpolate(latest, size=target_size, mode="bilinear", align_corners=False)


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise RuntimeError(f"Gradient loss shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    return F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)


def compute_training_loss(
    *,
    raw_preds: torch.Tensor,
    loss_target: torch.Tensor,
    final_preds: torch.Tensor,
    y: torch.Tensor,
    loss_mode: str,
    l1_weight: float,
    grad_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mse_loss = F.mse_loss(raw_preds, loss_target)
    loss = mse_loss
    if loss_mode in {"mse_l1", "mse_l1_grad"} and l1_weight > 0.0:
        loss = loss + float(l1_weight) * F.l1_loss(raw_preds, loss_target)
    if loss_mode in {"mse_grad", "mse_l1_grad"} and grad_weight > 0.0:
        loss = loss + float(grad_weight) * spatial_gradient_loss(final_preds, y)
    return loss, mse_loss


def run_epoch(
    model: Any,
    loader: Any,
    loss_mode: str,
    l1_weight: float,
    grad_weight: float,
    optimizer: Any,
    device: Any,
    train: bool,
    input_mean: Optional[torch.Tensor],
    input_std: Optional[torch.Tensor],
    grad_clip: Optional[float],
    target_mode: str,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    model.train(mode=train)
    running_loss = 0.0
    running_mse = 0.0
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
            raw_preds = model(x_model, target_size=(y.shape[-2], y.shape[-1]))
            if not torch.isfinite(raw_preds).all():
                raise RuntimeError("Model produced non-finite predictions")
            if raw_preds.shape != y.shape:
                raise RuntimeError(f"Prediction/target shape mismatch: pred={tuple(raw_preds.shape)} y={tuple(y.shape)}")

            if target_mode == "residual":
                base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
                loss_target = y - base
                final_preds = base + raw_preds
            else:
                loss_target = y
                final_preds = raw_preds

            loss, _mse_loss = compute_training_loss(
                raw_preds=raw_preds,
                loss_target=loss_target,
                final_preds=final_preds,
                y=y,
                loss_mode=loss_mode,
                l1_weight=l1_weight,
                grad_weight=grad_weight,
            )
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
        running_mse += float(F.mse_loss(final_preds, y).item())

    mean_loss = running_loss / max(1, len(loader))
    mean_mse = running_mse / max(1, len(loader))
    rmse = float(np.sqrt(max(mean_mse, 0.0)))
    if not np.isfinite(mean_loss) or not np.isfinite(rmse):
        raise RuntimeError("Non-finite epoch metrics encountered")
    if not train:
        return mean_loss, rmse, None, None

    grad_norm_mean = grad_norm_sum / max(1, grad_norm_steps)
    return mean_loss, rmse, grad_norm_mean, grad_norm_max


def save_training_curve(curve_rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_rmse",
        "val_rmse",
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

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset

    apply_dataset_version_to_args(args)

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
        static_covariate_path=args.static_covariate_path,
        verbose=False,
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

    best_val_loss = float("inf")
    best_epoch = 0
    curve_rows: List[dict] = []
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_rmse, train_grad_norm_mean, train_grad_norm_max = run_epoch(
            model,
            train_loader,
            args.loss_mode,
            args.l1_weight,
            args.grad_weight,
            optimizer,
            device,
            train=True,
            input_mean=input_mean,
            input_std=input_std,
            grad_clip=args.grad_clip,
            target_mode=args.target_mode,
        )
        val_loss, val_rmse, _, _ = run_epoch(
            model,
            val_loader,
            args.loss_mode,
            args.l1_weight,
            args.grad_weight,
            optimizer,
            device,
            train=False,
            input_mean=input_mean,
            input_std=input_std,
            grad_clip=args.grad_clip,
            target_mode=args.target_mode,
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
            epochs_since_improve = 0
            torch.save(
                {
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "history_length": int(args.history_length),
                    "target_mode": args.target_mode,
                    "loss_mode": args.loss_mode,
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
        else:
            epochs_since_improve += 1

        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "train_rmse": f"{train_rmse:.8f}",
                "val_rmse": f"{val_rmse:.8f}",
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

        if int(args.early_stopping_patience) > 0 and epochs_since_improve >= int(args.early_stopping_patience):
            break

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
            "target_mode": args.target_mode,
            "loss_mode": args.loss_mode,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "l1_weight": float(args.l1_weight),
            "grad_weight": float(args.grad_weight),
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
