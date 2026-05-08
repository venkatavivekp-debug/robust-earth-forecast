from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLAIN_ENCODER_DECODER_ALIASES = {"cnn", "plain_encoder_decoder"}


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check border-vs-center errors for an ERA5->PRISM checkpoint."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="small")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--experiment-dir", type=str, default="results/experiments/core4_h3")
    parser.add_argument("--model", choices=["cnn", "plain_encoder_decoder", "unet", "convlstm"], default="convlstm")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input-set", choices=["t2m", "core4", "extended"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all validation samples")
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-panel",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    run_name = Path(args.experiment_dir).name or "experiment"
    if args.output_json is None:
        args.output_json = f"results/diagnostics/spatial_artifacts_{run_name}_{args.model}.json"
    if args.output_panel is None:
        args.output_panel = f"results/diagnostics/spatial_artifacts_{run_name}_{args.model}.png"
    return args


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


def stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def save_panel(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    abs_error: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(min(np.min(prediction), np.min(target)))
    vmax = float(max(np.max(prediction), np.max(target)))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    im0 = axes[0].imshow(prediction, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(target, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("PRISM target")
    axes[1].axis("off")

    im2 = axes[2].imshow(abs_error, cmap="magma")
    axes[2].set_title("Absolute error")
    axes[2].axis("off")

    axes[3].imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Border mask")
    axes[3].axis("off")

    fig.colorbar(im0, ax=[axes[0], axes[1]], shrink=0.85, label="deg C")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="deg C")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import torch

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device, set_seed
    from models.baselines import upsample_latest_era5

    args = parse_args()
    apply_dataset_version_to_args(args)
    set_seed(int(args.split_seed))
    device = resolve_device(args.device)

    experiment_dir = PROJECT_ROOT / args.experiment_dir
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint_name = "cnn_best.pt" if args.model in PLAIN_ENCODER_DECODER_ALIASES else f"{args.model}_best.pt"
        checkpoint = experiment_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(PROJECT_ROOT / args.era5_path if not Path(args.era5_path).is_absolute() else args.era5_path),
        prism_path=str(PROJECT_ROOT / args.prism_path if not Path(args.prism_path).is_absolute() else args.prism_path),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        verbose=False,
    )

    model, ckpt_history, input_norm, checkpoint_input_set, _, ckpt_val_indices = load_checkpoint_model(
        args.model, checkpoint, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"Checkpoint history {ckpt_history} != requested {args.history_length}")
    if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
        raise RuntimeError(f"Checkpoint input_set {checkpoint_input_set} != requested {args.input_set}")
    if ckpt_val_indices is None:
        raise RuntimeError("Checkpoint does not include validation split metadata")

    eval_indices = [int(i) for i in ckpt_val_indices]
    if args.num_samples > 0:
        eval_indices = eval_indices[: min(int(args.num_samples), len(eval_indices))]
    if not eval_indices:
        raise RuntimeError("No validation samples available for diagnostic")

    errors: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    dates: List[str] = []

    with torch.no_grad():
        for sample_idx in eval_indices:
            x, y = dataset[sample_idx]
            xb = x.unsqueeze(0).to(device)
            yb = y.unsqueeze(0).to(device)
            raw_pred = model(
                normalize_input_batch(xb, input_norm),
                target_size=(yb.shape[-2], yb.shape[-1]),
            )
            if getattr(model, "target_mode", "direct") == "residual":
                if args.model == "convlstm":
                    raise RuntimeError("Residual target mode is not supported for ConvLSTM checkpoints")
                pred = upsample_latest_era5(xb, target_size=(yb.shape[-2], yb.shape[-1])) + raw_pred
            else:
                pred = raw_pred
            if pred.shape != yb.shape:
                raise RuntimeError(f"Prediction/target shape mismatch: pred={tuple(pred.shape)} y={tuple(yb.shape)}")

            pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float64)
            target_np = yb.squeeze().detach().cpu().numpy().astype(np.float64)
            predictions.append(pred_np)
            targets.append(target_np)
            errors.append(pred_np - target_np)
            dates.append(dataset.metadata(sample_idx).date.strftime("%Y-%m-%d"))

    pred_cube = np.stack(predictions, axis=0)
    target_cube = np.stack(targets, axis=0)
    err_cube = np.stack(errors, axis=0)
    abs_err_cube = np.abs(err_cube)

    mask2d = border_mask(pred_cube.shape[-2:], int(args.border_pixels))
    center2d = ~mask2d
    mask = np.broadcast_to(mask2d, pred_cube.shape)
    center = np.broadcast_to(center2d, pred_cube.shape)

    overall_rmse = float(np.sqrt(np.mean(err_cube ** 2)))
    mae = float(np.mean(abs_err_cube))
    border_rmse = float(np.sqrt(np.mean(err_cube[mask] ** 2)))
    center_rmse = float(np.sqrt(np.mean(err_cube[center] ** 2)))
    ratio = float(border_rmse / center_rmse) if center_rmse > 0 else float("inf")

    sample_border_rmse = [
        float(np.sqrt(np.mean(err[mask2d] ** 2)))
        for err in err_cube
    ]
    panel_idx = int(np.argmax(sample_border_rmse))

    payload: Dict[str, Any] = {
        "model": args.model,
        "dataset_version": args.dataset_version,
        "experiment_dir": args.experiment_dir,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint),
        "target_mode": getattr(model, "target_mode", "direct"),
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "border_pixels": int(args.border_pixels),
        "num_samples": len(eval_indices),
        "eval_indices": eval_indices,
        "eval_dates": dates,
        "overall_rmse": overall_rmse,
        "mae": mae,
        "border_rmse": border_rmse,
        "center_rmse": center_rmse,
        "border_center_rmse_ratio": ratio,
        "border_prediction": stats(pred_cube[mask]),
        "center_prediction": stats(pred_cube[center]),
        "target_border_mean": float(np.mean(target_cube[mask])),
        "target_center_mean": float(np.mean(target_cube[center])),
        "panel_sample_index": int(eval_indices[panel_idx]),
        "panel_sample_date": dates[panel_idx],
    }

    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    out_panel = PROJECT_ROOT / args.output_panel
    save_panel(
        prediction=pred_cube[panel_idx],
        target=target_cube[panel_idx],
        abs_error=abs_err_cube[panel_idx],
        mask=mask2d,
        output_path=out_panel,
        title=f"{args.model} border diagnostic | {dates[panel_idx]} | border={args.border_pixels}px",
    )

    print("spatial artifact diagnostic")
    print(f"model={args.model} checkpoint={payload['checkpoint']}")
    print(f"samples={len(eval_indices)} border_pixels={args.border_pixels}")
    print(f"overall_rmse={overall_rmse:.6f}")
    print(f"mae={mae:.6f}")
    print(f"border_rmse={border_rmse:.6f}")
    print(f"center_rmse={center_rmse:.6f}")
    print(f"border_center_rmse_ratio={ratio:.6f}")
    print(
        "border_prediction "
        f"mean={payload['border_prediction']['mean']:.6f} "
        f"min={payload['border_prediction']['min']:.6f} "
        f"max={payload['border_prediction']['max']:.6f}"
    )
    print(
        "center_prediction "
        f"mean={payload['center_prediction']['mean']:.6f} "
        f"min={payload['center_prediction']['min']:.6f} "
        f"max={payload['center_prediction']['max']:.6f}"
    )
    print(f"target_border_mean={payload['target_border_mean']:.6f}")
    print(f"target_center_mean={payload['target_center_mean']:.6f}")
    print(f"wrote_json={out_json}")
    print(f"wrote_panel={out_panel}")


if __name__ == "__main__":
    main()
