from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ORDER = ("persistence", "plain_encoder_decoder", "unet")


def _configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose boundary errors for controlled ERA5->PRISM spatial benchmark checkpoints."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--benchmark-root", type=str, default="results/spatial_benchmark_seed_stability")
    parser.add_argument("--output-dir", type=str, default="results/boundary_artifact_diagnosis")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--border-pixels", type=int, default=8)
    parser.add_argument("--center-crop-fraction", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mask_bundle(shape: Tuple[int, int], border_width: int, center_crop_fraction: float) -> Dict[str, np.ndarray]:
    h, w = shape
    if border_width < 1:
        raise ValueError("border-pixels must be >= 1")
    if border_width * 2 >= min(h, w):
        raise ValueError(f"border-pixels={border_width} is too large for target shape {(h, w)}")
    if not 0.0 < center_crop_fraction <= 1.0:
        raise ValueError("center-crop-fraction must be in (0, 1]")

    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    top[:border_width, :] = True
    bottom[-border_width:, :] = True
    left[:, :border_width] = True
    right[:, -border_width:] = True

    border = top | bottom | left | right
    center = ~border
    corners = (top | bottom) & (left | right)

    crop_h = max(1, int(round(h * center_crop_fraction)))
    crop_w = max(1, int(round(w * center_crop_fraction)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    center_crop = np.zeros((h, w), dtype=bool)
    center_crop[y0 : y0 + crop_h, x0 : x0 + crop_w] = True

    return {
        "border": border,
        "center": center,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "corner": corners,
        "center_crop": center_crop,
    }


def distance_from_edge(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    return np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.int32)


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def compute_region_metrics(errors: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "full_rmse": rmse(errors),
        "full_mae": mae(errors),
    }
    for name, mask2d in masks.items():
        mask = np.broadcast_to(mask2d, errors.shape)
        out[f"{name}_rmse"] = rmse(errors[mask])
        out[f"{name}_mae"] = mae(errors[mask])
    out["border_center_rmse_ratio"] = float(out["border_rmse"] / max(out["center_rmse"], 1e-8))
    return out


def distance_rows(
    *,
    seed: int,
    model_name: str,
    errors: np.ndarray,
    distance_map: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_distance = int(distance_map.max())
    for distance in range(max_distance + 1):
        mask2d = distance_map == distance
        mask = np.broadcast_to(mask2d, errors.shape)
        values = errors[mask]
        rows.append(
            {
                "seed": int(seed),
                "model": model_name,
                "distance_from_edge": int(distance),
                "rmse": rmse(values),
                "mae": mae(values),
                "n_values": int(values.size),
            }
        )
    return rows


def load_seed_models(
    *,
    seed_dir: Path,
    device: Any,
    history_length: int,
    input_set: str,
) -> Tuple[Dict[str, Tuple[Any, Optional[Dict[str, List[float]]], str]], List[int]]:
    from evaluation.evaluate_model import load_checkpoint_model

    loaded: Dict[str, Tuple[Any, Optional[Dict[str, List[float]]], str]] = {}
    split_signature: Optional[Tuple[int, ...]] = None

    for model_name in ("plain_encoder_decoder", "unet"):
        checkpoint = seed_dir / "checkpoints" / f"{model_name}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        model, ckpt_history, input_norm, checkpoint_input_set, _, val_indices = load_checkpoint_model(
            model_name, checkpoint, device
        )
        if int(ckpt_history) != int(history_length):
            raise RuntimeError(f"{model_name} checkpoint history {ckpt_history} != {history_length}")
        if checkpoint_input_set is not None and checkpoint_input_set != input_set:
            raise RuntimeError(f"{model_name} checkpoint input_set {checkpoint_input_set} != {input_set}")
        if val_indices is None:
            raise RuntimeError(f"{model_name} checkpoint does not include validation indices")
        signature = tuple(int(i) for i in val_indices)
        if split_signature is None:
            split_signature = signature
        elif signature != split_signature:
            raise RuntimeError(f"Checkpoints in {seed_dir} use different validation splits")
        loaded[model_name] = (model, input_norm, getattr(model, "target_mode", "direct"))

    assert split_signature is not None
    return loaded, list(split_signature)


def evaluate_seed(args: argparse.Namespace, seed: int, seed_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    import torch

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from evaluation.evaluate_model import normalize_input_batch, resolve_device, set_seed
    from models.baselines import upsample_latest_era5

    apply_dataset_version_to_args(args)
    set_seed(int(seed))
    device = resolve_device(args.device)

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        verbose=False,
    )
    loaded, eval_indices = load_seed_models(
        seed_dir=seed_dir,
        device=device,
        history_length=int(args.history_length),
        input_set=str(args.input_set),
    )

    pred_by_model: Dict[str, List[np.ndarray]] = {model: [] for model in MODEL_ORDER}
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for sample_idx in eval_indices:
            x, y = dataset[int(sample_idx)]
            xb = x.unsqueeze(0).to(device)
            yb = y.unsqueeze(0).to(device)
            target_size = (yb.shape[-2], yb.shape[-1])
            era5_up = upsample_latest_era5(xb, target_size=target_size)

            pred_by_model["persistence"].append(era5_up.squeeze().detach().cpu().numpy().astype(np.float64))
            for model_name in ("plain_encoder_decoder", "unet"):
                model, input_norm, target_mode = loaded[model_name]
                raw_pred = model(normalize_input_batch(xb, input_norm), target_size=target_size)
                if target_mode == "residual":
                    pred = era5_up + raw_pred
                else:
                    pred = raw_pred
                if pred.shape != yb.shape:
                    raise RuntimeError(
                        f"{model_name} prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(yb.shape)}"
                    )
                pred_by_model[model_name].append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
            targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))

    target_cube = np.stack(targets, axis=0)
    masks = mask_bundle(
        target_cube.shape[-2:],
        int(args.border_pixels),
        float(args.center_crop_fraction),
    )
    dist_map = distance_from_edge(target_cube.shape[-2:])

    metric_rows: List[Dict[str, Any]] = []
    distance_metric_rows: List[Dict[str, Any]] = []
    mean_abs_error_maps: Dict[str, np.ndarray] = {}

    for model_name in MODEL_ORDER:
        pred_cube = np.stack(pred_by_model[model_name], axis=0)
        errors = pred_cube - target_cube
        row: Dict[str, Any] = {
            "seed": int(seed),
            "model": model_name,
            "num_samples": int(len(eval_indices)),
            "border_pixels": int(args.border_pixels),
            "center_crop_fraction": float(args.center_crop_fraction),
        }
        row.update(compute_region_metrics(errors, masks))
        metric_rows.append(row)
        distance_metric_rows.extend(
            distance_rows(seed=int(seed), model_name=model_name, errors=errors, distance_map=dist_map)
        )
        mean_abs_error_maps[model_name] = np.mean(np.abs(errors), axis=0)

    return metric_rows, distance_metric_rows, mean_abs_error_maps


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def aggregate_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics = [
        "full_rmse",
        "full_mae",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "top_rmse",
        "bottom_rmse",
        "left_rmse",
        "right_rmse",
        "corner_rmse",
        "center_crop_rmse",
    ]
    payload: Dict[str, Dict[str, float]] = {}
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        summary: Dict[str, float] = {"n_seeds": float(len(model_rows))}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        payload[model] = summary
    return payload


def save_distance_plot(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model in MODEL_ORDER:
        distances = sorted({int(row["distance_from_edge"]) for row in rows if row["model"] == model})
        means = []
        for distance in distances:
            vals = [
                float(row["mae"])
                for row in rows
                if row["model"] == model and int(row["distance_from_edge"]) == distance
            ]
            means.append(float(np.mean(vals)))
        ax.plot(distances, means, marker="o", markersize=2.5, linewidth=1.6, label=model)
    ax.set_xlabel("Distance from nearest image edge (pixels)")
    ax.set_ylabel("Mean absolute error (deg C)")
    ax.set_title("Boundary error profile")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_border_mask(shape: Tuple[int, int], border_pixels: int, output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    masks = mask_bundle(shape, border_pixels, center_crop_fraction=0.5)
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    im = ax.imshow(masks["border"].astype(float), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Border mask ({border_pixels}px)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_mean_abs_error_maps(maps_by_model: Dict[str, List[np.ndarray]], output_path: Path) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    maps = {model: np.mean(np.stack(items, axis=0), axis=0) for model, items in maps_by_model.items()}
    vmax = float(max(np.max(arr) for arr in maps.values()))
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4 * len(MODEL_ORDER), 4), constrained_layout=True)
    im = None
    for ax, model in zip(axes, MODEL_ORDER):
        im = ax.imshow(maps[model], cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(model)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean |error| (deg C)")
    fig.suptitle("Mean absolute error map across seeds")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_root = repo_path(args.benchmark_root)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    distance_metric_rows: List[Dict[str, Any]] = []
    maps_by_model: Dict[str, List[np.ndarray]] = {model: [] for model in MODEL_ORDER}
    target_shape: Optional[Tuple[int, int]] = None

    for seed in args.seeds:
        seed_dir = benchmark_root / f"seed_{int(seed)}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed benchmark directory: {seed_dir}")
        rows, distance_rows_for_seed, mean_maps = evaluate_seed(args, int(seed), seed_dir)
        metric_rows.extend(rows)
        distance_metric_rows.extend(distance_rows_for_seed)
        for model, arr in mean_maps.items():
            maps_by_model[model].append(arr)
            if target_shape is None:
                target_shape = arr.shape

    boundary_fields = [
        "seed",
        "model",
        "num_samples",
        "full_rmse",
        "full_mae",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "top_rmse",
        "bottom_rmse",
        "left_rmse",
        "right_rmse",
        "corner_rmse",
        "center_crop_rmse",
        "border_pixels",
        "center_crop_fraction",
    ]
    write_csv(output_dir / "boundary_metrics.csv", metric_rows, boundary_fields)
    write_csv(
        output_dir / "error_by_distance.csv",
        distance_metric_rows,
        ["seed", "model", "distance_from_edge", "rmse", "mae", "n_values"],
    )

    aggregate = aggregate_metrics(metric_rows)
    summary = {
        "dataset_version": args.dataset_version,
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "seeds": [int(s) for s in args.seeds],
        "benchmark_root": str(benchmark_root.relative_to(PROJECT_ROOT) if benchmark_root.is_relative_to(PROJECT_ROOT) else benchmark_root),
        "border_pixels": int(args.border_pixels),
        "center_crop_fraction": float(args.center_crop_fraction),
        "aggregate_by_model": aggregate,
        "border_degradation_confirmed": all(float(row["border_center_rmse_ratio"]) > 1.0 for row in metric_rows),
        "learned_border_degradation_stronger_than_persistence": False,
    }
    persistence_ratios = {
        int(row["seed"]): float(row["border_center_rmse_ratio"])
        for row in metric_rows
        if row["model"] == "persistence"
    }
    learned_rows = [row for row in metric_rows if row["model"] in ("plain_encoder_decoder", "unet")]
    summary["learned_border_degradation_stronger_than_persistence"] = any(
        float(row["border_center_rmse_ratio"]) > persistence_ratios[int(row["seed"])]
        for row in learned_rows
    )
    (output_dir / "boundary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_distance_plot(distance_metric_rows, output_dir / "error_vs_boundary_distance.png")
    if target_shape is not None:
        save_border_mask(target_shape, int(args.border_pixels), output_dir / "border_mask.png")
    save_mean_abs_error_maps(maps_by_model, output_dir / "mean_abs_error_border_map.png")

    print("boundary artifact diagnosis")
    for model in MODEL_ORDER:
        item = aggregate[model]
        print(
            f"{model:>21} | full_rmse={item['full_rmse_mean']:.4f} "
            f"border={item['border_rmse_mean']:.4f} center={item['center_rmse_mean']:.4f} "
            f"ratio={item['border_center_rmse_ratio_mean']:.3f}"
        )
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
