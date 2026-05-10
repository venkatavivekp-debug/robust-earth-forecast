from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.unet_downscaler import UNetDownscaler
from training.train_downscaler import (
    compute_input_stats,
    normalize_input_batch,
    residual_base,
    set_seed,
    split_dataset,
)


BANDS: Tuple[Tuple[str, float, float | None], ...] = (
    ("4-8km", 4.0, 8.0),
    ("8-16km", 8.0, 16.0),
    ("16-32km", 16.0, 32.0),
    ("32km+", 32.0, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze where the static-bias U-Net decoder loses spatial detail.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--input-set", choices=["core4_topo"], default="core4_topo")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-spacing-km", type=float, default=4.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", default="results/decoder_pathway_analysis")
    parser.add_argument("--image-out", default="docs/images/decoder_psd_by_stage.png")
    parser.add_argument("--doc-out", default="docs/research/decoder_pathway_findings.md")
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
    if wavelengths.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    bins = np.geomspace(max(float(np.min(wavelengths)), spacing_km), float(np.max(wavelengths)), 36)
    centers = np.sqrt(bins[:-1] * bins[1:])
    which = np.digitize(wavelengths, bins) - 1
    values: List[float] = []
    for idx in range(len(centers)):
        mask = which == idx
        values.append(float(np.mean(powers[mask])) if np.any(mask) else np.nan)
    return centers, np.asarray(values, dtype=np.float64)


def compute_static_bias(dataset: ERA5_PRISM_Dataset, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    residuals: List[torch.Tensor] = []
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[int(idx)]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            base = residual_base(x, target_size=(y.shape[-2], y.shape[-1]))
            residuals.append((y - base).detach().cpu())
    return torch.mean(torch.cat(residuals, dim=0), dim=0, keepdim=True).to(device)


def train_static_bias_model(
    *,
    args: argparse.Namespace,
    dataset: ERA5_PRISM_Dataset,
    train_indices: Sequence[int],
    device: torch.device,
) -> Tuple[UNetDownscaler, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(int(args.seed))
    x, y = dataset[int(args.sample_index)]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    static_bias = compute_static_bias(dataset, train_indices, device)
    input_mean_np, input_std_np = compute_input_stats(dataset, train_indices)
    input_mean = torch.tensor(input_mean_np, device=device)
    input_std = torch.tensor(input_std_np, device=device)
    model = UNetDownscaler(
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=1,
        base_channels=int(args.unet_base_channels),
        padding_mode="reflection",
        upsample_mode="bilinear",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    x_model = normalize_input_batch(x, input_mean, input_std)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        pred = model(x_model, target_size=(static_bias.shape[-2], static_bias.shape[-1]))
        loss = F.mse_loss(pred, static_bias)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            rmse = float(torch.sqrt(F.mse_loss(pred, static_bias)).item())
            print(f"epoch={epoch:04d} static_bias_rmse={rmse:.4f}")
    model.eval()
    return model, x, y, x_model, static_bias, residual_base(x, target_size=(y.shape[-2], y.shape[-1]))


def forward_with_stages(
    model: UNetDownscaler,
    x_model: torch.Tensor,
    target_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    x_flat = model._prepare_input(x_model)
    e1 = model.enc1(x_flat)
    e2 = model.enc2(F.avg_pool2d(e1, kernel_size=2, stride=2))
    b = model.bottleneck(F.avg_pool2d(e2, kernel_size=2, stride=2))
    d2 = model._upsample(b, size=e2.shape[-2:], layer=getattr(model, "up2", None))
    d2 = model.dec2(torch.cat([d2, e2], dim=1))
    d1 = model._upsample(d2, size=e1.shape[-2:], layer=getattr(model, "up1", None))
    d1 = model.dec1(torch.cat([d1, e1], dim=1))
    high = model._upsample(d1, size=target_size, layer=getattr(model, "up_high", None))
    high = model.high_res(high)
    out = model.out(high)
    return {
        "decoder_block_1": d2,
        "decoder_block_2": d1,
        "pre_output_high_res": high,
        "final_output": out,
    }


def stage_proxy(tensor: torch.Tensor) -> np.ndarray:
    return torch.mean(tensor, dim=1).squeeze(0).detach().cpu().numpy()


def resize_tensor(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def save_psd_plot(
    *,
    output_path: Path,
    residual_target: np.ndarray,
    residual_prediction: np.ndarray,
    stage_maps: Dict[str, np.ndarray],
    spacing_km: float,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for name, arr in stage_maps.items():
        wl, psd = radial_psd(zscore(arr), spacing_km)
        axes[0].plot(wl, psd, label=name, linewidth=1.5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Wavelength (km)")
    axes[0].set_ylabel("PSD of z-scored stage map")
    axes[0].set_title("Decoder stage spectral content")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for label, arr in [("static bias target", residual_target), ("final prediction", residual_prediction)]:
        wl, psd = radial_psd(arr, spacing_km)
        axes[1].plot(wl, psd, label=label, linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Wavelength (km)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Final residual PSD")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(
    *,
    output_path: Path,
    image_path: Path,
    band_rows: Sequence[Dict[str, Any]],
    stage_rows: Sequence[Dict[str, Any]],
    base_rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_below = summary.get("first_band_below_50pct") or "none"
    lines = [
        "# Decoder Pathway Findings",
        "",
        "This diagnostic trains the static-bias U-Net setup again and inspects the",
        "decoder pathway. Intermediate decoder tensors are activation maps, not",
        "calibrated predictions, so their spectra are compared after z-scoring.",
        "The final output is the actual static-bias residual prediction.",
        "",
        f"- Static-bias RMSE: `{summary['static_bias_rmse']:.4f}` deg C",
        f"- Static-bias banded 4-32 km retention: `{summary['static_bias_high_frequency_retention']:.4f}`",
        f"- First final-output band below 50% retention: `{first_below}`",
        f"- U-Net final-temperature PSD lower than bilinear ERA5 in any band: `{'yes' if summary['unet_lower_than_bilinear_any_band'] else 'no'}`",
        "",
        "Final residual PSD retention:",
        "",
        "| Band | Target power | Prediction power | Retention |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in band_rows:
        lines.append(
            f"| {row['band']} | {row['target_power']:.6g} | {row['prediction_power']:.6g} | {row['retention']:.4f} |"
        )
    lines.extend(["", "Decoder stage spectral fractions:", "", "| Stage | 4-8km | 8-16km | 16-32km | 32km+ |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['fraction_4-8km']:.4f} | {row['fraction_8-16km']:.4f} | "
            f"{row['fraction_16-32km']:.4f} | {row['fraction_32km+']:.4f} |"
        )
    lines.extend(["", "Final temperature PSD compared with bilinear ERA5:", "", "| Band | Bilinear ERA5 power | U-Net final power | U-Net lower? |", "| --- | ---: | ---: | --- |"])
    for row in base_rows:
        lines.append(
            f"| {row['band']} | {row['bilinear_power']:.6g} | {row['unet_final_power']:.6g} | "
            f"{'yes' if row['unet_lower_than_bilinear'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"PSD figure: `{image_path}`.",
            "",
            "The main read is whether the final residual prediction keeps power in the",
            "near-grid-scale bands. Low retention there is consistent with the decoder",
            "acting as a low-pass reconstruction path even when the static target is",
            "stable and learnable.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    device = torch.device(args.device)

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
    _train_set, _val_set, train_indices, _val_indices = split_dataset(dataset, float(args.val_fraction), int(args.split_seed))
    model, x, _y, x_model, static_bias, base = train_static_bias_model(
        args=args, dataset=dataset, train_indices=train_indices, device=device
    )
    with torch.no_grad():
        stages = forward_with_stages(model, x_model, target_size=(static_bias.shape[-2], static_bias.shape[-1]))

    target_np = static_bias.squeeze().detach().cpu().numpy()
    pred_np = stages["final_output"].squeeze().detach().cpu().numpy()
    err = pred_np - target_np
    spacing = float(args.grid_spacing_km)

    band_rows: List[Dict[str, Any]] = []
    for band, low, high in BANDS:
        target_power = band_power(target_np, spacing, low, high)
        pred_power = band_power(pred_np, spacing, low, high)
        band_rows.append(
            {
                "band": band,
                "target_power": target_power,
                "prediction_power": pred_power,
                "retention": float(pred_power / target_power) if np.isfinite(target_power) and target_power > 0 else float("nan"),
            }
        )

    stage_rows: List[Dict[str, Any]] = []
    stage_maps_for_plot: Dict[str, np.ndarray] = {}
    for stage_name in ("decoder_block_1", "decoder_block_2", "pre_output_high_res"):
        proxy = stage_proxy(stages[stage_name])
        stage_maps_for_plot[stage_name] = proxy
        total = 0.0
        powers: Dict[str, float] = {}
        for band, low, high in BANDS:
            power = band_power(zscore(proxy), spacing, low, high)
            powers[band] = power
            if np.isfinite(power):
                total += power
        row: Dict[str, Any] = {"stage": stage_name}
        for band, _low, _high in BANDS:
            row[f"fraction_{band}"] = float(powers[band] / total) if total > 0 and np.isfinite(powers[band]) else 0.0
        stage_rows.append(row)

    base_np = base.squeeze().detach().cpu().numpy()
    final_unet_np = base_np + pred_np
    base_rows: List[Dict[str, Any]] = []
    for band, low, high in BANDS:
        base_power = band_power(base_np, spacing, low, high)
        unet_power = band_power(final_unet_np, spacing, low, high)
        base_rows.append(
            {
                "band": band,
                "bilinear_power": base_power,
                "unet_final_power": unet_power,
                "unet_lower_than_bilinear": bool(np.isfinite(base_power) and np.isfinite(unet_power) and unet_power < base_power),
            }
        )

    valid_retention = [row for row in band_rows if np.isfinite(row["retention"])]
    below = next((row["band"] for row in valid_retention if float(row["retention"]) < 0.5), None)
    target_hf = sum(row["target_power"] for row in band_rows if row["band"] in {"4-8km", "8-16km", "16-32km"} and np.isfinite(row["target_power"]))
    pred_hf = sum(row["prediction_power"] for row in band_rows if row["band"] in {"4-8km", "8-16km", "16-32km"} and np.isfinite(row["prediction_power"]))
    summary = {
        "dataset_version": str(args.dataset_version),
        "sample_index": int(args.sample_index),
        "epochs": int(args.epochs),
        "static_bias_rmse": float(np.sqrt(np.mean(err**2))),
        "static_bias_mae": float(np.mean(np.abs(err))),
        "static_bias_high_frequency_retention": float(pred_hf / max(target_hf, 1e-8)),
        "first_band_below_50pct": below,
        "unet_lower_than_bilinear_any_band": any(row["unet_lower_than_bilinear"] for row in base_rows),
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frequency_band_retention.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)
    with (output_dir / "decoder_stage_spectral_fractions.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(stage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage_rows)
    with (output_dir / "bilinear_vs_unet_psd.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(base_rows[0].keys()))
        writer.writeheader()
        writer.writerows(base_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "static_bias_target.npy", target_np)
    np.save(output_dir / "static_bias_prediction.npy", pred_np)

    image_path = repo_path(args.image_out)
    save_psd_plot(
        output_path=image_path,
        residual_target=target_np,
        residual_prediction=pred_np,
        stage_maps=stage_maps_for_plot,
        spacing_km=spacing,
    )
    write_doc(
        output_path=repo_path(args.doc_out),
        image_path=image_path.relative_to(PROJECT_ROOT),
        band_rows=band_rows,
        stage_rows=stage_rows,
        base_rows=base_rows,
        summary=summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
