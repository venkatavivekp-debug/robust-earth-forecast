from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device
from training.train_downscaler import residual_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a residual U-Net collapses toward the bilinear ERA5 baseline."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--input-set", default="core4_topo", choices=["core4", "core4_elev", "core4_topo"])
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument(
        "--checkpoint",
        default="results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt",
    )
    parser.add_argument("--output-dir", default="results/residual_collapse_diagnosis")
    parser.add_argument("--image-out", default="docs/images/residual_collapse_diagnosis.png")
    parser.add_argument("--doc-out", default="docs/research/residual_collapse_findings.md")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return 0.0
    r = float(np.corrcoef(aa, bb)[0, 1])
    return r if np.isfinite(r) else 0.0


def save_residual_maps(
    *,
    pred_mean: np.ndarray,
    target_mean: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(max(np.max(np.abs(pred_mean)), np.max(np.abs(target_mean)), 1e-6))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    im0 = axes[0].imshow(pred_mean, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Mean predicted residual")
    axes[0].axis("off")
    im1 = axes[1].imshow(target_mean, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Mean target residual")
    axes[1].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.82, label="deg C")
    fig.colorbar(im1, ax=axes[1], shrink=0.82, label="deg C")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_findings_doc(
    *,
    output_path: Path,
    summary: Dict[str, Any],
    image_path: Path,
    checkpoint: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collapsed = bool(summary["collapse_confirmed"])
    interpretation = (
        "The residual branch is effectively returning the bilinear ERA5 field."
        if collapsed
        else "The residual branch is not fully collapsed by the 0.2 magnitude-ratio rule."
    )
    output_path.write_text(
        "\n".join(
            [
                "# Residual Collapse Findings",
                "",
                "This diagnostic checks whether the trained residual U-Net is learning a meaningful",
                "`PRISM - ERA5_bilinear` correction or mostly returning the bilinear ERA5 baseline.",
                "",
                f"- Checkpoint: `{checkpoint}`",
                f"- Validation samples: `{summary['num_samples']}`",
                f"- Mean |predicted residual|: `{summary['predicted_residual_magnitude_mean']:.4f}` deg C",
                f"- Mean |target residual|: `{summary['target_residual_magnitude_mean']:.4f}` deg C",
                f"- Residual magnitude ratio: `{summary['residual_magnitude_ratio']:.4f}`",
                f"- Pearson r between mean predicted and target residual maps: `{summary['mean_map_pearson_r']:.4f}`",
                f"- Collapse rule (`ratio < 0.2`): `{'confirmed' if collapsed else 'not confirmed'}`",
                "",
                interpretation,
                "",
                "The image compares the mean predicted residual map with the mean target residual map:",
                f"`{image_path}`.",
                "",
                "This does not prove that the model is incorrectly implemented. If the daily residual",
                "is weakly predictable from ERA5/topography, MSE training should move the predicted",
                "residual toward its conditional mean, which can be close to zero.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)

    checkpoint_path = repo_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    static_path = repo_path(args.static_covariate_path)
    if args.input_set in {"core4_elev", "core4_topo"} and not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        static_covariate_path=str(static_path) if args.input_set in {"core4_elev", "core4_topo"} else None,
        verbose=False,
    )

    device = resolve_device(str(args.device))
    model, ckpt_history, input_norm, ckpt_input_set, _train_indices, val_indices = load_checkpoint_model(
        "unet", checkpoint_path, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"Checkpoint history_length={ckpt_history} does not match {args.history_length}")
    if ckpt_input_set is not None and ckpt_input_set != args.input_set:
        raise RuntimeError(f"Checkpoint input_set={ckpt_input_set} does not match {args.input_set}")
    if not val_indices:
        raise RuntimeError("Checkpoint does not contain validation indices")

    rows: List[Dict[str, Any]] = []
    pred_residual_maps: List[np.ndarray] = []
    target_residual_maps: List[np.ndarray] = []
    with torch.no_grad():
        for sample_idx in [int(i) for i in val_indices]:
            x, y = dataset[sample_idx]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
            raw = model(normalize_input_batch(x, input_norm), target_size=(y.shape[-2], y.shape[-1]))
            target_mode = getattr(model, "target_mode", "residual")
            if target_mode == "residual":
                pred = base + raw
            else:
                pred = raw

            pred_residual = (pred - base).squeeze().detach().cpu().numpy()
            target_residual = (y - base).squeeze().detach().cpu().numpy()
            pred_abs = float(np.mean(np.abs(pred_residual)))
            target_abs = float(np.mean(np.abs(target_residual)))
            rows.append(
                {
                    "sample_index": sample_idx,
                    "date": dataset.metadata(sample_idx).date.strftime("%Y-%m-%d"),
                    "predicted_residual_abs_mean": pred_abs,
                    "target_residual_abs_mean": target_abs,
                    "residual_magnitude_ratio": pred_abs / max(target_abs, 1e-8),
                    "sample_residual_map_pearson_r": pearson_r(pred_residual, target_residual),
                }
            )
            pred_residual_maps.append(pred_residual)
            target_residual_maps.append(target_residual)

    pred_stack = np.stack(pred_residual_maps, axis=0)
    target_stack = np.stack(target_residual_maps, axis=0)
    mean_pred = np.mean(pred_stack, axis=0)
    mean_target = np.mean(target_stack, axis=0)
    pred_mag = float(np.mean(np.abs(pred_stack)))
    target_mag = float(np.mean(np.abs(target_stack)))
    ratio = pred_mag / max(target_mag, 1e-8)
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_version": str(args.dataset_version),
        "input_set": str(args.input_set),
        "history_length": int(args.history_length),
        "num_samples": len(rows),
        "predicted_residual_magnitude_mean": pred_mag,
        "target_residual_magnitude_mean": target_mag,
        "residual_magnitude_ratio": float(ratio),
        "mean_map_pearson_r": pearson_r(mean_pred, mean_target),
        "collapse_confirmed": bool(ratio < 0.2),
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "residual_collapse_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "mean_predicted_residual.npy", mean_pred)
    np.save(output_dir / "mean_target_residual.npy", mean_target)

    image_path = repo_path(args.image_out)
    save_residual_maps(
        pred_mean=mean_pred,
        target_mean=mean_target,
        output_path=image_path,
        title="Residual collapse diagnosis",
    )
    write_findings_doc(
        output_path=repo_path(args.doc_out),
        summary=summary,
        image_path=image_path.relative_to(PROJECT_ROOT),
        checkpoint=checkpoint_path.relative_to(PROJECT_ROOT),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
