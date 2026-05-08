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
        description="Diagnose spatial sharpness and fine-scale detail in ERA5->PRISM benchmark outputs."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--benchmark-root", type=str, default="results/spatial_benchmark_seed_stability")
    parser.add_argument("--output-dir", type=str, default="results/spatial_sharpness_diagnosis")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--high-pass-window", type=int, default=7)
    parser.add_argument("--local-window", type=int, default=7)
    parser.add_argument("--patch-size", type=int, default=32)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def gradient_magnitude(cube: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(cube, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx**2 + gy**2)


def box_mean(cube: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window size must be a positive odd integer")
    arr = np.asarray(cube, dtype=np.float64)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    pad = window // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float64)
    h, w = arr.shape[-2:]
    for dy in range(window):
        for dx in range(window):
            out += padded[:, dy : dy + h, dx : dx + w]
    out /= float(window * window)
    return out[0] if squeeze else out


def high_pass(cube: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(cube, dtype=np.float64) - box_mean(cube, window)


def local_contrast(cube: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    mean = box_mean(arr, window)
    mean_sq = box_mean(arr * arr, window)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def distance_from_edge(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    return np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.int32)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    if af.size < 2 or bf.size < 2:
        return 0.0
    value = float(np.corrcoef(af, bf)[0, 1])
    return value if np.isfinite(value) else 0.0


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


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


def collect_seed_predictions(
    args: argparse.Namespace,
    seed: int,
    seed_dir: Path,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
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
                pred = era5_up + raw_pred if target_mode == "residual" else raw_pred
                if pred.shape != yb.shape:
                    raise RuntimeError(
                        f"{model_name} prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(yb.shape)}"
                    )
                pred_by_model[model_name].append(pred.squeeze().detach().cpu().numpy().astype(np.float64))
            targets.append(yb.squeeze().detach().cpu().numpy().astype(np.float64))

    return {model: np.stack(items, axis=0) for model, items in pred_by_model.items()}, np.stack(targets, axis=0)


def compute_sharpness_metrics(
    *,
    seed: int,
    model_name: str,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
    high_pass_window: int,
    local_window: int,
) -> Dict[str, Any]:
    err = pred_cube - target_cube
    pred_grad = gradient_magnitude(pred_cube)
    target_grad = gradient_magnitude(target_cube)
    grad_err = pred_grad - target_grad

    pred_detail = high_pass(pred_cube, high_pass_window)
    target_detail = high_pass(target_cube, high_pass_window)
    detail_err = pred_detail - target_detail

    pred_contrast = local_contrast(pred_cube, local_window)
    target_contrast = local_contrast(target_cube, local_window)

    target_variance = float(np.var(target_cube))
    prediction_variance = float(np.var(pred_cube))
    target_detail_energy = float(np.mean(target_detail**2))
    prediction_detail_energy = float(np.mean(pred_detail**2))
    target_contrast_mean = float(np.mean(target_contrast))
    prediction_contrast_mean = float(np.mean(pred_contrast))
    target_grad_mean = float(np.mean(target_grad))
    prediction_grad_mean = float(np.mean(pred_grad))

    return {
        "seed": int(seed),
        "model": model_name,
        "num_samples": int(pred_cube.shape[0]),
        "rmse": rmse(err),
        "mae": mae(err),
        "bias": float(np.mean(err)),
        "correlation": corrcoef(pred_cube, target_cube),
        "target_variance": target_variance,
        "prediction_variance": prediction_variance,
        "variance_ratio": float(prediction_variance / max(target_variance, 1e-8)),
        "target_gradient_mean": target_grad_mean,
        "prediction_gradient_mean": prediction_grad_mean,
        "gradient_mean_ratio": float(prediction_grad_mean / max(target_grad_mean, 1e-8)),
        "gradient_magnitude_rmse": rmse(grad_err),
        "target_high_frequency_energy": target_detail_energy,
        "prediction_high_frequency_energy": prediction_detail_energy,
        "high_frequency_energy_ratio": float(prediction_detail_energy / max(target_detail_energy, 1e-8)),
        "high_frequency_residual_energy": float(np.mean(detail_err**2)),
        "target_local_contrast_mean": target_contrast_mean,
        "prediction_local_contrast_mean": prediction_contrast_mean,
        "local_contrast_ratio": float(prediction_contrast_mean / max(target_contrast_mean, 1e-8)),
        "local_contrast_difference": float(prediction_contrast_mean - target_contrast_mean),
    }


def compute_distance_rows(
    *,
    seed: int,
    model_name: str,
    pred_cube: np.ndarray,
    target_cube: np.ndarray,
) -> List[Dict[str, Any]]:
    err = pred_cube - target_cube
    dist = distance_from_edge(target_cube.shape[-2:])
    rows: List[Dict[str, Any]] = []
    for distance in range(int(dist.max()) + 1):
        mask2d = dist == distance
        mask = np.broadcast_to(mask2d, err.shape)
        values = err[mask]
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


def aggregate_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = [
        "rmse",
        "mae",
        "bias",
        "correlation",
        "target_variance",
        "prediction_variance",
        "variance_ratio",
        "target_gradient_mean",
        "prediction_gradient_mean",
        "gradient_mean_ratio",
        "gradient_magnitude_rmse",
        "target_high_frequency_energy",
        "prediction_high_frequency_energy",
        "high_frequency_energy_ratio",
        "high_frequency_residual_energy",
        "target_local_contrast_mean",
        "prediction_local_contrast_mean",
        "local_contrast_ratio",
        "local_contrast_difference",
    ]
    payload: Dict[str, Dict[str, float]] = {}
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        summary: Dict[str, float] = {"n_seeds": float(len(model_rows))}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        payload[model] = summary
    return payload


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate = aggregate_metrics(rows)
    out: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        row: Dict[str, Any] = {"model": model}
        row.update(aggregate[model])
        out.append(row)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def choose_patch(target_cube: np.ndarray, patch_size: int) -> Tuple[int, int, int]:
    h, w = target_cube.shape[-2:]
    if patch_size < 4 or patch_size > min(h, w):
        raise ValueError(f"patch-size must be between 4 and {min(h, w)}")
    pad = patch_size // 2
    grad = gradient_magnitude(target_cube)
    score = np.mean(grad, axis=0)
    score[:pad, :] = -np.inf
    score[-pad:, :] = -np.inf
    score[:, :pad] = -np.inf
    score[:, -pad:] = -np.inf
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    sample = int(np.argmax(grad[:, y, x]))
    return sample, int(y), int(x)


def patch_bounds(center_y: int, center_x: int, patch_size: int) -> Tuple[slice, slice]:
    half = patch_size // 2
    return slice(center_y - half, center_y - half + patch_size), slice(center_x - half, center_x - half + patch_size)


def save_gradient_plot(
    *,
    mean_targets: List[np.ndarray],
    mean_predictions: Dict[str, List[np.ndarray]],
    output_path: Path,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    target_mean = np.mean(np.stack(mean_targets, axis=0), axis=0)
    maps: Dict[str, np.ndarray] = {"target": gradient_magnitude(target_mean)}
    for model in MODEL_ORDER:
        pred_mean = np.mean(np.stack(mean_predictions[model], axis=0), axis=0)
        maps[model] = gradient_magnitude(pred_mean)

    vmax = float(max(np.percentile(arr, 99) for arr in maps.values()))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    im = None
    for ax, name in zip(axes, ("target", *MODEL_ORDER)):
        im = ax.imshow(maps[name], cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Gradient magnitude")
    fig.suptitle("Mean-field gradient comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_high_frequency_plot(
    *,
    mean_targets: List[np.ndarray],
    mean_predictions: Dict[str, List[np.ndarray]],
    high_pass_window: int,
    output_path: Path,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    target_mean = np.mean(np.stack(mean_targets, axis=0), axis=0)
    maps: Dict[str, np.ndarray] = {"target": np.abs(high_pass(target_mean, high_pass_window))}
    for model in MODEL_ORDER:
        pred_mean = np.mean(np.stack(mean_predictions[model], axis=0), axis=0)
        maps[model] = np.abs(high_pass(pred_mean, high_pass_window))

    vmax = float(max(np.percentile(arr, 99) for arr in maps.values()))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    im = None
    for ax, name in zip(axes, ("target", *MODEL_ORDER)):
        im = ax.imshow(maps[name], cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="|local detail|")
    fig.suptitle(f"High-frequency detail, {high_pass_window}x{high_pass_window} local mean removed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_zoomed_patch_plot(
    *,
    target_cube: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    patch_size: int,
    output_path: Path,
) -> None:
    _configure_plot_cache()
    import matplotlib.pyplot as plt

    sample, cy, cx = choose_patch(target_cube, patch_size)
    ys, xs = patch_bounds(cy, cx, patch_size)
    panels: Dict[str, np.ndarray] = {"target": target_cube[sample, ys, xs]}
    for model in MODEL_ORDER:
        panels[model] = pred_by_model[model][sample, ys, xs]

    vmin = float(min(np.min(arr) for arr in panels.values()))
    vmax = float(max(np.max(arr) for arr in panels.values()))
    err_vmax = float(max(np.max(np.abs(pred_by_model[model][sample, ys, xs] - panels["target"])) for model in MODEL_ORDER))

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0), constrained_layout=True)
    im = None
    for ax, name in zip(axes[0], ("target", *MODEL_ORDER)):
        im = ax.imshow(panels[name], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="Temperature")

    axes[1, 0].imshow(gradient_magnitude(panels["target"]), cmap="viridis")
    axes[1, 0].set_title("target gradient")
    axes[1, 0].axis("off")
    err_im = None
    for ax, model in zip(axes[1, 1:], MODEL_ORDER):
        err_im = ax.imshow(np.abs(panels[model] - panels["target"]), cmap="magma", vmin=0.0, vmax=err_vmax)
        ax.set_title(f"{model} |error|")
        ax.axis("off")
    fig.colorbar(err_im, ax=axes[1, 1:], shrink=0.85, label="Absolute error")
    fig.suptitle(f"Local patch around strong target gradient, sample {sample}, y={cy}, x={cx}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    ax.set_title("Error by boundary distance")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_root = repo_path(args.benchmark_root)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    mean_targets: List[np.ndarray] = []
    mean_predictions: Dict[str, List[np.ndarray]] = {model: [] for model in MODEL_ORDER}
    first_target_cube: Optional[np.ndarray] = None
    first_pred_by_model: Optional[Dict[str, np.ndarray]] = None

    for seed in args.seeds:
        seed_dir = benchmark_root / f"seed_{int(seed)}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed benchmark directory: {seed_dir}")

        pred_by_model, target_cube = collect_seed_predictions(args, int(seed), seed_dir)
        if first_target_cube is None:
            first_target_cube = target_cube
            first_pred_by_model = pred_by_model

        mean_targets.append(np.mean(target_cube, axis=0))
        for model in MODEL_ORDER:
            pred_cube = pred_by_model[model]
            metric_rows.append(
                compute_sharpness_metrics(
                    seed=int(seed),
                    model_name=model,
                    pred_cube=pred_cube,
                    target_cube=target_cube,
                    high_pass_window=int(args.high_pass_window),
                    local_window=int(args.local_window),
                )
            )
            distance_rows.extend(
                compute_distance_rows(
                    seed=int(seed),
                    model_name=model,
                    pred_cube=pred_cube,
                    target_cube=target_cube,
                )
            )
            mean_predictions[model].append(np.mean(pred_cube, axis=0))

    fields = [
        "seed",
        "model",
        "num_samples",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "target_variance",
        "prediction_variance",
        "variance_ratio",
        "target_gradient_mean",
        "prediction_gradient_mean",
        "gradient_mean_ratio",
        "gradient_magnitude_rmse",
        "target_high_frequency_energy",
        "prediction_high_frequency_energy",
        "high_frequency_energy_ratio",
        "high_frequency_residual_energy",
        "target_local_contrast_mean",
        "prediction_local_contrast_mean",
        "local_contrast_ratio",
        "local_contrast_difference",
    ]
    write_csv(output_dir / "summary.csv", metric_rows, fields)

    aggregate_fields = ["model", "n_seeds"]
    for field in fields[3:]:
        aggregate_fields.extend([f"{field}_mean", f"{field}_std"])
    aggregate_rows_payload = aggregate_rows(metric_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows_payload, aggregate_fields)

    write_csv(
        output_dir / "boundary_distance_error.csv",
        distance_rows,
        ["seed", "model", "distance_from_edge", "rmse", "mae", "n_values"],
    )

    aggregate = aggregate_metrics(metric_rows)
    summary = {
        "dataset_version": args.dataset_version,
        "input_set": args.input_set,
        "history_length": int(args.history_length),
        "seeds": [int(s) for s in args.seeds],
        "benchmark_root": relpath(benchmark_root),
        "high_pass_window": int(args.high_pass_window),
        "local_window": int(args.local_window),
        "aggregate_by_model": aggregate,
        "oversmoothing_indicators": {
            model: {
                "variance_ratio_below_one": aggregate[model]["variance_ratio_mean"] < 1.0,
                "gradient_ratio_below_one": aggregate[model]["gradient_mean_ratio_mean"] < 1.0,
                "high_frequency_ratio_below_one": aggregate[model]["high_frequency_energy_ratio_mean"] < 1.0,
                "local_contrast_ratio_below_one": aggregate[model]["local_contrast_ratio_mean"] < 1.0,
            }
            for model in MODEL_ORDER
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_gradient_plot(
        mean_targets=mean_targets,
        mean_predictions=mean_predictions,
        output_path=output_dir / "gradient_comparison.png",
    )
    save_high_frequency_plot(
        mean_targets=mean_targets,
        mean_predictions=mean_predictions,
        high_pass_window=int(args.high_pass_window),
        output_path=output_dir / "high_frequency_detail_maps.png",
    )
    if first_target_cube is not None and first_pred_by_model is not None:
        save_zoomed_patch_plot(
            target_cube=first_target_cube,
            pred_by_model=first_pred_by_model,
            patch_size=int(args.patch_size),
            output_path=output_dir / "zoomed_patch_comparison.png",
        )
    save_distance_plot(distance_rows, output_dir / "error_vs_boundary_distance.png")

    print("spatial sharpness diagnosis")
    for model in MODEL_ORDER:
        item = aggregate[model]
        print(
            f"{model:>21} | rmse={item['rmse_mean']:.4f} "
            f"var_ratio={item['variance_ratio_mean']:.3f} "
            f"grad_ratio={item['gradient_mean_ratio_mean']:.3f} "
            f"hf_ratio={item['high_frequency_energy_ratio_mean']:.3f} "
            f"contrast_ratio={item['local_contrast_ratio_mean']:.3f}"
        )
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
