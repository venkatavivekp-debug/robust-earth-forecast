from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BANDS: Tuple[Tuple[str, float, Optional[float]], ...] = (
    (">64 km", 64.0, None),
    ("32-64 km", 32.0, 64.0),
    ("16-32 km", 16.0, 32.0),
    ("8-16 km", 8.0, 16.0),
    ("4-8 km", 4.0, 8.0),
)


def configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze wavelength-dependent recoverability for ERA5->PRISM.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument(
        "--bilinear-checkpoint",
        default="results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt",
    )
    parser.add_argument(
        "--pixelshuffle-checkpoint",
        default="results/pixelshuffle_full_training/seed_42/checkpoints/unet_core4_topo_h3_best.pt",
    )
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--input-set", default="core4_topo")
    parser.add_argument("--output-dir", default="results/recoverable_scale_analysis")
    parser.add_argument("--docs-image-dir", default="docs/images")
    parser.add_argument("--grid-km", type=float, default=4.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(af) & np.isfinite(bf)
    if int(ok.sum()) < 2:
        return float("nan")
    value = float(np.corrcoef(af[ok], bf[ok])[0, 1])
    return value if np.isfinite(value) else float("nan")


def gradient_magnitude(cube: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(cube, dtype=np.float64), axis=(-2, -1))
    return np.sqrt(gx * gx + gy * gy)


def upsample_latest_t2m(x: Any, target_size: Tuple[int, int]) -> Any:
    import torch.nn.functional as F

    latest_t2m = x[:, -1, 0:1, :, :]
    return F.interpolate(latest_t2m, size=target_size, mode="bilinear", align_corners=False)


def normalize_input_batch(x: Any, input_norm: Optional[Dict[str, List[float]]]) -> Any:
    if not input_norm:
        return x
    import torch

    mean_vals = input_norm.get("mean")
    std_vals = input_norm.get("std")
    if not mean_vals or not std_vals:
        return x
    mean = torch.tensor(mean_vals, device=x.device, dtype=x.dtype)
    std = torch.tensor(std_vals, device=x.device, dtype=x.dtype).clamp(min=1e-6)
    return (x - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)


def load_unet_checkpoint(checkpoint_path: Path, device: str) -> Tuple[Any, Dict[str, Any]]:
    import importlib.util
    import torch

    try:
        from models.unet_downscaler import UNetDownscaler
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location("unet_downscaler", PROJECT_ROOT / "models" / "unet_downscaler.py")
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        UNetDownscaler = module.UNetDownscaler

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    model = UNetDownscaler(
        in_channels=int(cfg.get("in_channels", 24)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 24)),
        padding_mode=str(cfg.get("padding_mode", "reflection")),
        upsample_mode=str(cfg.get("upsample_mode", "bilinear")),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def collect_cubes(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], List[str]]:
    import torch
    import torch.nn.functional as F
    import xarray as xr

    from datasets.dataset_paths import apply_dataset_version_to_args
    from datasets.prism_dataset import ERA5_PRISM_Dataset

    apply_dataset_version_to_args(args)
    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        static_covariate_path=str(repo_path(args.static_covariate_path)),
        verbose=False,
    )

    bilinear_model = None
    bilinear_ckpt: Optional[Dict[str, Any]] = None
    bilinear_path = repo_path(args.bilinear_checkpoint)
    if bilinear_path.exists():
        bilinear_model, bilinear_ckpt = load_unet_checkpoint(bilinear_path, args.device)
        val_indices = [int(i) for i in bilinear_ckpt.get("val_indices", [])]
    else:
        val_indices = list(range(len(dataset)))

    pixel_model = None
    pixel_ckpt: Optional[Dict[str, Any]] = None
    pixel_path = repo_path(args.pixelshuffle_checkpoint)
    if pixel_path.exists():
        pixel_model, pixel_ckpt = load_unet_checkpoint(pixel_path, args.device)

    if not val_indices:
        raise RuntimeError("No validation indices found for recoverability analysis")

    topo_ds = xr.open_dataset(repo_path(args.static_covariate_path))
    prism_elev = topo_ds["elevation"].values.astype(np.float32)

    cubes: Dict[str, List[np.ndarray]] = {
        "PRISM target": [],
        "ERA5 bilinear": [],
        "Lapse-rate ERA5": [],
    }
    if bilinear_model is not None:
        cubes["Residual U-Net bilinear"] = []
    if pixel_model is not None:
        cubes["Residual U-Net PixelShuffle"] = []

    dates: List[str] = []
    with torch.no_grad():
        for idx in val_indices:
            x, y = dataset[int(idx)]
            xb = x.unsqueeze(0).to(args.device)
            yb = y.unsqueeze(0).to(args.device)
            target_size = (int(yb.shape[-2]), int(yb.shape[-1]))
            era5_up = upsample_latest_t2m(xb, target_size)

            era5_elev = xb[:, -1, 4:5, :, :]
            era5_elev_up = F.interpolate(era5_elev, size=target_size, mode="bilinear", align_corners=False)
            lapse = era5_up - 6.5 * (
                torch.tensor(prism_elev, device=args.device, dtype=era5_up.dtype).view(1, 1, *target_size)
                - era5_elev_up
            ) / 1000.0

            cubes["PRISM target"].append(yb.squeeze().cpu().numpy().astype(np.float64))
            cubes["ERA5 bilinear"].append(era5_up.squeeze().cpu().numpy().astype(np.float64))
            cubes["Lapse-rate ERA5"].append(lapse.squeeze().cpu().numpy().astype(np.float64))

            if bilinear_model is not None and bilinear_ckpt is not None:
                raw = bilinear_model(
                    normalize_input_batch(xb, bilinear_ckpt.get("input_norm")),
                    target_size=target_size,
                )
                pred = era5_up + raw
                cubes["Residual U-Net bilinear"].append(pred.squeeze().cpu().numpy().astype(np.float64))

            if pixel_model is not None and pixel_ckpt is not None:
                raw = pixel_model(
                    normalize_input_batch(xb, pixel_ckpt.get("input_norm")),
                    target_size=target_size,
                )
                pred = era5_up + raw
                cubes["Residual U-Net PixelShuffle"].append(pred.squeeze().cpu().numpy().astype(np.float64))

            dates.append(dataset.metadata(int(idx)).date.strftime("%Y-%m-%d"))

    return {name: np.stack(values, axis=0) for name, values in cubes.items()}, dates


def fft_frequency_grid(shape: Tuple[int, int], grid_km: float) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0], d=grid_km)
    fx = np.fft.fftfreq(shape[1], d=grid_km)
    kx, ky = np.meshgrid(fx, fy)
    return np.sqrt(kx * kx + ky * ky)


def band_mask(freq: np.ndarray, lo_km: float, hi_km: Optional[float]) -> np.ndarray:
    if hi_km is None:
        return (freq > 0) & (freq < 1.0 / lo_km)
    if lo_km <= 4.0:
        return freq >= 1.0 / hi_km
    return (freq >= 1.0 / hi_km) & (freq < 1.0 / lo_km)


def bandpass(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    out = np.empty_like(arr)
    for i, sample in enumerate(arr):
        centered = sample - float(np.mean(sample))
        coeff = np.fft.fft2(centered)
        out[i] = np.fft.ifft2(coeff * mask).real
    return out


def compute_band_metrics(cubes: Mapping[str, np.ndarray], grid_km: float) -> List[Dict[str, Any]]:
    target = cubes["PRISM target"]
    freq = fft_frequency_grid(target.shape[-2:], grid_km)
    rows: List[Dict[str, Any]] = []

    for band_name, lo, hi in BANDS:
        mask = band_mask(freq, lo, hi)
        target_band = bandpass(target, mask)
        target_energy = float(np.mean(target_band**2))
        target_grad = gradient_magnitude(target_band)
        for name, cube in cubes.items():
            pred_band = bandpass(cube, mask)
            err = pred_band - target_band
            pred_energy = float(np.mean(pred_band**2))
            sse = float(np.sum(err**2))
            sst = float(np.sum(target_band**2))
            rows.append(
                {
                    "model": name,
                    "band": band_name,
                    "target_energy": target_energy,
                    "psd_energy": pred_energy,
                    "energy_retention": pred_energy / max(target_energy, 1e-12),
                    "band_correlation": corrcoef(pred_band, target_band),
                    "band_rmse": float(np.sqrt(np.mean(err**2))),
                    "gradient_correlation": corrcoef(gradient_magnitude(pred_band), target_grad),
                    "explained_variance": 1.0 - sse / max(sst, 1e-12),
                }
            )
    return rows


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_recoverability_curve(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    centers = {">64 km": 96.0, "32-64 km": 48.0, "16-32 km": 24.0, "8-16 km": 12.0, "4-8 km": 6.0}
    order = [band for band, _, _ in BANDS]
    models = [m for m in dict.fromkeys(row["model"] for row in rows) if m != "PRISM target"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model in models:
        subset = {row["band"]: row for row in rows if row["model"] == model}
        x = [centers[band] for band in order]
        corr = [subset[band]["band_correlation"] for band in order]
        retention = [subset[band]["energy_retention"] for band in order]
        axes[0].plot(x, corr, marker="o", label=model)
        axes[1].plot(x, retention, marker="o", label=model)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(128, 4)
        ax.set_xlabel("Wavelength (km)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Band-pass correlation with PRISM")
    axes[1].set_ylabel("Energy retention vs PRISM")
    axes[1].set_yscale("log")
    axes[0].set_title("Recoverability by wavelength")
    axes[1].set_title("Spatial energy retained")
    axes[1].legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_bandpass_panels(cubes: Mapping[str, np.ndarray], output_path: Path, grid_km: float) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    freq = fft_frequency_grid(cubes["PRISM target"].shape[-2:], grid_km)
    bands_to_show = [("16-32 km", 16.0, 32.0), ("8-16 km", 8.0, 16.0), ("4-8 km", 4.0, 8.0)]
    models = ["PRISM target", "ERA5 bilinear", "Residual U-Net bilinear", "Residual U-Net PixelShuffle"]
    models = [m for m in models if m in cubes]
    sample = 0

    fig, axes = plt.subplots(len(bands_to_show), len(models), figsize=(3.3 * len(models), 3.2 * len(bands_to_show)), constrained_layout=True)
    if len(bands_to_show) == 1:
        axes = axes[None, :]
    for r, (band, lo, hi) in enumerate(bands_to_show):
        mask = band_mask(freq, lo, hi)
        maps = [bandpass(cubes[model][sample : sample + 1], mask)[0] for model in models]
        vmax = float(max(np.percentile(np.abs(m), 99) for m in maps))
        vmax = max(vmax, 1e-6)
        for c, (model, arr) in enumerate(zip(models, maps)):
            im = axes[r, c].imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axes[r, c].set_title(f"{model}\n{band}")
            axes[r, c].axis("off")
        fig.colorbar(im, ax=axes[r, :], shrink=0.75, label="deg C band-pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_image_dir = repo_path(args.docs_image_dir)
    docs_image_dir.mkdir(parents=True, exist_ok=True)

    cubes, dates = collect_cubes(args)
    rows = compute_band_metrics(cubes, float(args.grid_km))
    save_csv(rows, output_dir / "recoverability_summary.csv")
    (output_dir / "recoverability_summary.json").write_text(
        json.dumps({"dates": dates, "rows": rows}, indent=2) + "\n"
    )
    save_csv(
        [
            {
                "model": row["model"],
                "band": row["band"],
                "band_rmse": row["band_rmse"],
                "band_correlation": row["band_correlation"],
                "energy_retention": row["energy_retention"],
            }
            for row in rows
        ],
        output_dir / "wavelength_error_table.csv",
    )

    curve_path = output_dir / "recoverability_curve.png"
    panel_path = output_dir / "bandpass_reconstruction_panels.png"
    save_recoverability_curve(rows, curve_path)
    save_bandpass_panels(cubes, panel_path, float(args.grid_km))

    # Commit selected figures for README/notebook use while leaving result tables ignored.
    import shutil

    shutil.copyfile(curve_path, docs_image_dir / "recoverability_curve.png")
    shutil.copyfile(panel_path, docs_image_dir / "bandpass_reconstruction_panels.png")

    print(f"wrote {output_dir / 'recoverability_summary.csv'}")
    for row in rows:
        if row["model"] in {"ERA5 bilinear", "Residual U-Net bilinear", "Residual U-Net PixelShuffle"}:
            print(
                f"{row['model']:>28} | {row['band']:>8} | "
                f"corr={row['band_correlation']:.3f} retention={row['energy_retention']:.3f} "
                f"rmse={row['band_rmse']:.3f}"
            )


if __name__ == "__main__":
    main()
