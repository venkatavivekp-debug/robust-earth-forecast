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
import torch.nn.functional as F

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device


BANDS: Tuple[Tuple[str, float, float | None], ...] = (
    ("4-8km", 4.0, 8.0),
    ("8-16km", 8.0, 16.0),
    ("16-32km", 16.0, 32.0),
    ("32km+", 32.0, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure spectral content carried by U-Net skip feature maps.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--input-set", choices=["core4_topo"], default="core4_topo")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument(
        "--checkpoint",
        default="results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt",
    )
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--grid-spacing-km", type=float, default=4.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", default="results/skip_feature_quality")
    parser.add_argument("--image-out", default="docs/images/skip_feature_psd.png")
    parser.add_argument("--doc-out", default="docs/research/skip_feature_quality_findings.md")
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


def zscore(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    return (data - float(np.mean(data))) / max(float(np.std(data)), 1e-8)


def power_grid(field: np.ndarray, spacing_km: float) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    h, w = arr.shape[-2:]
    fy = np.fft.fftfreq(h, d=spacing_km)
    fx = np.fft.fftfreq(w, d=spacing_km)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    wavelength = np.divide(1.0, kr, out=np.full_like(kr, np.inf), where=kr > 0)
    power = np.abs(np.fft.fft2(arr)) ** 2
    power[kr == 0] = 0.0
    return wavelength, power


def band_power(field: np.ndarray, spacing_km: float, low_km: float, high_km: float | None) -> float:
    wavelength, power = power_grid(field, spacing_km)
    if high_km is None:
        mask = wavelength >= low_km
    else:
        mask = (wavelength >= low_km) & (wavelength < high_km)
    mask &= np.isfinite(wavelength)
    return float(np.mean(power[mask])) if np.any(mask) else float("nan")


def radial_psd(field: np.ndarray, spacing_km: float) -> Tuple[np.ndarray, np.ndarray]:
    wavelength, power = power_grid(field, spacing_km)
    finite = np.isfinite(wavelength)
    wavelengths = wavelength[finite].reshape(-1)
    powers = power[finite].reshape(-1)
    bins = np.geomspace(max(float(np.min(wavelengths)), spacing_km), float(np.max(wavelengths)), 36)
    centers = np.sqrt(bins[:-1] * bins[1:])
    which = np.digitize(wavelengths, bins) - 1
    values: List[float] = []
    for idx in range(len(centers)):
        mask = which == idx
        values.append(float(np.mean(powers[mask])) if np.any(mask) else np.nan)
    return centers, np.asarray(values, dtype=np.float64)


def upsample_feature_mean(tensor: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    mean_map = torch.mean(tensor, dim=1, keepdim=True)
    up = F.interpolate(mean_map, size=target_size, mode="bilinear", align_corners=False)
    return up.squeeze().detach().cpu().numpy()


def save_psd_plot(
    *,
    output_path: Path,
    curves: Dict[str, np.ndarray],
    spacing_km: float,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, arr in curves.items():
        wl, psd = radial_psd(zscore(arr), spacing_km)
        ax.plot(wl, psd, label=label, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("PSD of z-scored map")
    ax.set_title("Skip feature spectral content")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(
    *,
    output_path: Path,
    rows: List[Dict[str, Any]],
    image_path: Path,
    summary: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Skip Feature Quality Findings",
        "",
        "This diagnostic asks whether the U-Net skip tensors carry fine-scale content",
        "toward the decoder. The maps are z-scored before PSD comparison, so the",
        "numbers describe spectral distribution rather than physical temperature units.",
        "",
        f"- Encoder skip features classified as high-HF: `{'yes' if summary['skip_features_high_hf'] else 'no'}`",
        f"- Interpretation: {summary['interpretation']}",
        "",
        "| Map | spatial std | 4-8km retention | 8-16km retention | 16-32km retention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['map']} | {row['spatial_std']:.4f} | {row['retention_4-8km']:.4f} | "
            f"{row['retention_8-16km']:.4f} | {row['retention_16-32km']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"PSD figure: `{image_path}`.",
            "",
            "Because the model input is ERA5 resolution, the skip tensors are not",
            "PRISM-resolution skip connections. They preserve coarse spatial layout and",
            "terrain-conditioned features, but they cannot directly carry native 4 km",
            "PRISM detail into the final decoder.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    checkpoint_path = repo_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    static_path = repo_path(args.static_covariate_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        static_covariate_path=str(static_path),
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
    sample_index = int(args.sample_index) if int(args.sample_index) >= 0 else int(val_indices[0])

    activations: Dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            activations[name] = output.detach()

        return hook

    hooks = [
        model.enc1.register_forward_hook(capture("skip_enc1")),
        model.enc2.register_forward_hook(capture("skip_enc2")),
    ]
    try:
        x, y = dataset[sample_index]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        with torch.no_grad():
            model(normalize_input_batch(x, input_norm), target_size=(y.shape[-2], y.shape[-1]))
    finally:
        for hook in hooks:
            hook.remove()

    target_np = y.squeeze().detach().cpu().numpy()
    spacing = float(args.grid_spacing_km)
    curves = {"PRISM target": target_np}
    rows: List[Dict[str, Any]] = []
    target_band_powers = {
        band: band_power(zscore(target_np), spacing, low, high)
        for band, low, high in BANDS
    }
    for name in ("skip_enc1", "skip_enc2"):
        feature_map = upsample_feature_mean(activations[name], target_size=target_np.shape)
        curves[name] = feature_map
        row: Dict[str, Any] = {
            "map": name,
            "spatial_std": float(np.std(feature_map)),
        }
        for band, low, high in BANDS:
            feature_power = band_power(zscore(feature_map), spacing, low, high)
            target_power = target_band_powers[band]
            row[f"retention_{band}"] = float(feature_power / target_power) if target_power > 0 else float("nan")
        rows.append(row)

    high_hf = all(float(row["retention_8-16km"]) >= 0.5 and float(row["retention_16-32km"]) >= 0.5 for row in rows)
    interpretation = (
        "skip features retain substantial near-grid spectral content"
        if high_hf
        else "skip features are coarse relative to PRISM-scale detail; final decoder must synthesize the finest structure"
    )
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "sample_index": sample_index,
        "skip_features_high_hf": bool(high_hf),
        "interpretation": interpretation,
        "rows": rows,
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "skip_feature_band_power.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    image_path = repo_path(args.image_out)
    save_psd_plot(output_path=image_path, curves=curves, spacing_km=spacing)
    write_doc(
        output_path=repo_path(args.doc_out),
        rows=rows,
        image_path=image_path.relative_to(PROJECT_ROOT),
        summary=summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
