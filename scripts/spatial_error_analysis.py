"""
Spatial error vs PRISM gradient analysis (best ConvLSTM: core4, history=3).

Computes pixel-wise absolute error, mean maps over the evaluation split,
PRISM gradient magnitude via simple finite differences, and the correlation
between mean gradient and mean absolute error per pixel.

Does not change model training or architectures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def spatial_gradient_magnitude(z: np.ndarray) -> np.ndarray:
    """Finite-difference gradient magnitude on a 2D field (same shape as z)."""
    z = np.asarray(z, dtype=np.float64)
    gx = np.zeros_like(z)
    gy = np.zeros_like(z)
    gx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) * 0.5
    gx[:, 0] = z[:, 1] - z[:, 0]
    gx[:, -1] = z[:, -1] - z[:, -2]
    gy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
    gy[0, :] = z[1, :] - z[0, :]
    gy[-1, :] = z[-1, :] - z[-2, :]
    return np.sqrt(gx * gx + gy * gy)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial error vs PRISM gradient (ConvLSTM core4)")
    p.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_multi.nc")
    p.add_argument("--prism-path", type=str, default="data_raw/prism")
    p.add_argument("--input-set", type=str, default="core4", choices=["t2m", "core4", "extended"])
    p.add_argument("--history-length", type=int, default=3)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--convlstm-checkpoint",
        type=str,
        default="results/experiments/core4_h3/checkpoints/convlstm_best.pt",
    )
    p.add_argument("--num-samples", type=int, default=8, help="Max validation samples (same logic as evaluate_model)")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output-json", type=str, default="docs/experiments/error_analysis.json")
    p.add_argument("--images-dir", type=str, default="docs/images")
    return p.parse_args()


def main() -> None:
    import torch

    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import (
        load_checkpoint_model,
        normalize_input_batch,
        resolve_device,
        set_seed,
        split_indices,
    )

    args = parse_args()
    set_seed(int(args.split_seed))
    device = resolve_device(args.device)

    ckpt_path = PROJECT_ROOT / args.convlstm_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    era5_path = PROJECT_ROOT / args.era5_path
    prism_path = PROJECT_ROOT / args.prism_path
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_path}")
    if not prism_path.exists():
        raise FileNotFoundError(f"PRISM path not found: {prism_path}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(era5_path),
        prism_path=str(prism_path),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        verbose=False,
    )
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small for analysis")

    model, ckpt_history, input_norm, checkpoint_input_set, ckpt_train_indices, ckpt_val_indices = load_checkpoint_model(
        "convlstm", ckpt_path, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"Checkpoint history {ckpt_history} != {args.history_length}")
    if checkpoint_input_set is not None and checkpoint_input_set != args.input_set:
        raise RuntimeError(f"Checkpoint input_set {checkpoint_input_set} != {args.input_set}")

    if ckpt_train_indices is not None and ckpt_val_indices is not None:
        train_indices = [int(i) for i in ckpt_train_indices]
        val_indices = [int(i) for i in ckpt_val_indices]
    else:
        train_indices, val_indices = split_indices(len(dataset), 0.2, int(args.split_seed))

    eval_indices = val_indices[: max(1, min(int(args.num_samples), len(val_indices)))]

    abs_err_stack: List[np.ndarray] = []
    grad_stack: List[np.ndarray] = []

    with torch.no_grad():
        for sample_idx in eval_indices:
            x, y = dataset[sample_idx]
            xb = x.unsqueeze(0).to(device)
            yb = y.unsqueeze(0).to(device)
            pred = model(
                normalize_input_batch(xb, input_norm),
                target_size=(yb.shape[-2], yb.shape[-1]),
            )
            if pred.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch pred={tuple(pred.shape)} y={tuple(yb.shape)}")

            target_np = yb.squeeze().detach().cpu().numpy()
            pred_np = pred.squeeze().detach().cpu().numpy()
            abs_err = np.abs(pred_np - target_np)
            grad_mag = spatial_gradient_magnitude(target_np)

            abs_err_stack.append(abs_err.astype(np.float64))
            grad_stack.append(grad_mag.astype(np.float64))

    err_cube = np.stack(abs_err_stack, axis=0)  # [N, H, W]
    grad_cube = np.stack(grad_stack, axis=0)
    mean_abs_err = np.mean(err_cube, axis=0)
    mean_grad = np.mean(grad_cube, axis=0)

    flat_g = mean_grad.reshape(-1)
    flat_e = mean_abs_err.reshape(-1)
    mask = np.isfinite(flat_g) & np.isfinite(flat_e)
    flat_g = flat_g[mask]
    flat_e = flat_e[mask]
    if flat_g.size < 3:
        raise RuntimeError("Not enough finite pixels for correlation")

    corr_matrix = np.corrcoef(flat_g, flat_e)
    correlation_spatial_mean_maps = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else float("nan")

    gp = np.concatenate([g.reshape(-1) for g in grad_stack])
    ep = np.concatenate([e.reshape(-1) for e in abs_err_stack])
    m2 = np.isfinite(gp) & np.isfinite(ep)
    gp = gp[m2]
    ep = ep[m2]
    if gp.size < 3:
        raise RuntimeError("Not enough finite pooled samples for correlation")
    corr_pooled = np.corrcoef(gp, ep)
    correlation_pooled_pixels = float(corr_pooled[0, 1]) if np.isfinite(corr_pooled[0, 1]) else float("nan")

    # Primary metric for reporting: spatial mean maps (one gradient / one error per pixel, averaged over time).
    correlation = correlation_spatial_mean_maps

    mean_error = float(np.mean(err_cube))
    max_error = float(np.max(err_cube))

    thr = float(np.percentile(mean_abs_err, 90.0))
    high_error_mask = mean_abs_err >= thr

    if correlation > 0.15:
        conclusion = (
            f"Mean absolute error is clearly positively correlated with mean PRISM gradient magnitude "
            f"(Pearson r={correlation:.3f} on per-pixel mean maps). "
            "Larger errors concentrate where the target field changes rapidly in space, consistent with limited "
            "ability to reproduce fine-scale spatial variability and sharp transitions."
        )
    elif correlation > 0.05:
        conclusion = (
            f"There is a small positive association between mean gradient and mean absolute error "
            f"(r={correlation:.3f} on per-pixel mean maps; pooled over all pixels and samples r={correlation_pooled_pixels:.3f}). "
            "High-gradient areas contribute modestly but do not explain most error variance; other factors "
            "(temporal context, systematic bias, or small misalignments) remain important."
        )
    elif correlation < -0.1:
        conclusion = (
            f"Correlation is negative (r={correlation:.3f}), so high-gradient regions are not the dominant "
            "driver of mean absolute error in this aggregate; other effects matter more."
        )
    else:
        conclusion = (
            f"Correlation between mean gradient and mean absolute error is near zero (r={correlation:.3f}). "
            "A simple gradient proxy does not strongly predict where the model fails; inspect spatial maps for structure."
        )

    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "model": "convlstm",
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "checkpoint": str(args.convlstm_checkpoint),
        "num_samples_used": len(eval_indices),
        "eval_indices": eval_indices,
        "correlation_gradient_error": correlation,
        "correlation_gradient_error_spatial_mean_maps": correlation_spatial_mean_maps,
        "correlation_gradient_error_pooled_pixels": correlation_pooled_pixels,
        "mean_error": mean_error,
        "max_error": max_error,
        "high_error_percentile_threshold": thr,
        "conclusion_text": conclusion,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    _configure_plot_cache()
    import matplotlib.pyplot as plt

    img_dir = PROJECT_ROOT / args.images_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    im0 = axes[0].imshow(mean_abs_err, cmap="magma")
    axes[0].set_title("Mean |pred − target|")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="deg C")

    im1 = axes[1].imshow(mean_grad, cmap="viridis")
    axes[1].set_title("Mean PRISM ||∇target||")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="deg C / pixel")
    fig.savefig(img_dir / "mean_abs_error_and_gradient.png", dpi=150)
    plt.close(fig)

    fig_m, ax_m = plt.subplots(1, 1, figsize=(4.2, 4), constrained_layout=True)
    ax_m.imshow(high_error_mask.astype(float), cmap="Reds", vmin=0, vmax=1)
    ax_m.set_title("High-error mask (top 10% mean |error|)")
    ax_m.axis("off")
    fig_m.savefig(img_dir / "high_error_mask.png", dpi=150)
    plt.close(fig_m)

    fig_s, ax_s = plt.subplots(1, 1, figsize=(5, 4.2), constrained_layout=True)
    rng = np.random.default_rng(42)
    n = flat_g.size
    cap = 12000
    if n > cap:
        idx = rng.choice(n, size=cap, replace=False)
        sg, se = flat_g[idx], flat_e[idx]
    else:
        sg, se = flat_g, flat_e
    ax_s.scatter(sg, se, s=4, alpha=0.25, c="0.2")
    ax_s.set_xlabel("Mean PRISM gradient magnitude")
    ax_s.set_ylabel("Mean absolute error (|pred − target|)")
    ax_s.set_title(f"Pixel-wise (mean over samples)\nPearson r = {correlation:.3f}")
    ax_s.grid(alpha=0.3)
    fig_s.savefig(img_dir / "scatter_gradient_vs_error.png", dpi=150)
    plt.close(fig_s)

    print(f"Wrote {out_json}")
    print(f"Wrote plots under {img_dir}")
    print(f"correlation_gradient_error={correlation:.6f} mean_error={mean_error:.6f} max_error={max_error:.6f}")


if __name__ == "__main__":
    main()
