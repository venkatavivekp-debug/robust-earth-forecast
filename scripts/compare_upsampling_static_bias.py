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
import torch.nn as nn
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


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1)
        self.scale = int(scale)
        self.out_channels = int(channels)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        out = F.pixel_shuffle(self.proj(x), upscale_factor=self.scale)
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out


class PixelShuffleUNetDownscaler(UNetDownscaler):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int, padding_mode: str) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            padding_mode=padding_mode,
            upsample_mode="bilinear",
        )
        self.upsample_mode = "pixelshuffle"
        self.up2 = PixelShuffleUpsample(base_channels * 4, scale=2)
        self.up1 = PixelShuffleUpsample(base_channels * 2, scale=2)
        self.up_high = PixelShuffleUpsample(base_channels, scale=4)

    def _upsample(self, x: torch.Tensor, size: Tuple[int, int], layer: nn.Module | None = None) -> torch.Tensor:
        if isinstance(layer, PixelShuffleUpsample):
            return layer(x, size)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare U-Net upsampling methods on the static-bias residual target.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
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
    parser.add_argument("--output-dir", default="results/upsampling_static_bias_comparison")
    parser.add_argument("--psd-image-out", default="docs/images/upsampling_comparison_psd.png")
    parser.add_argument("--panel-image-out", default="docs/images/upsampling_comparison_panels.png")
    parser.add_argument("--doc-out", default="docs/research/upsampling_method_findings.md")
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
    psd: List[float] = []
    for idx in range(len(centers)):
        mask = which == idx
        psd.append(float(np.mean(powers[mask])) if np.any(mask) else np.nan)
    return centers, np.asarray(psd, dtype=np.float64)


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


def make_model(mode: str, in_channels: int, out_channels: int, base_channels: int) -> nn.Module:
    if mode == "pixelshuffle":
        return PixelShuffleUNetDownscaler(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            padding_mode="reflection",
        )
    return UNetDownscaler(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        padding_mode="reflection",
        upsample_mode="convtranspose" if mode == "convtranspose" else "bilinear",
    )


def train_one(
    *,
    mode: str,
    args: argparse.Namespace,
    x: torch.Tensor,
    x_model: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    set_seed(int(args.seed))
    model = make_model(
        mode,
        in_channels=int(x.shape[1] * x.shape[2]),
        out_channels=1,
        base_channels=int(args.unet_base_channels),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    pred = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        raw = model(x_model, target_size=(target.shape[-2], target.shape[-1]))
        loss = F.mse_loss(raw, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = raw.detach()
        if epoch == 1 or epoch % 100 == 0:
            rmse = float(torch.sqrt(F.mse_loss(raw, target)).item())
            print(f"{mode} epoch={epoch:04d} static_bias_rmse={rmse:.4f}")
    assert pred is not None
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    err = pred_np - target_np
    target_hf = sum(
        band_power(target_np, float(args.grid_spacing_km), low, high)
        for band, low, high in BANDS
        if band in {"4-8km", "8-16km", "16-32km"}
    )
    pred_hf = sum(
        band_power(pred_np, float(args.grid_spacing_km), low, high)
        for band, low, high in BANDS
        if band in {"4-8km", "8-16km", "16-32km"}
    )
    metrics = {
        "upsampling": mode,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "high_frequency_retention": float(pred_hf / max(target_hf, 1e-8)),
    }
    for band, low, high in BANDS:
        target_power = band_power(target_np, float(args.grid_spacing_km), low, high)
        pred_power = band_power(pred_np, float(args.grid_spacing_km), low, high)
        metrics[f"retention_{band}"] = float(pred_power / target_power) if target_power > 0 else float("nan")
    return pred_np, metrics


def save_psd_plot(
    *,
    output_path: Path,
    target: np.ndarray,
    predictions: Dict[str, np.ndarray],
    spacing_km: float,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, arr in [("target", target), *predictions.items()]:
        wl, psd = radial_psd(arr, spacing_km)
        ax.plot(wl, psd, label=label, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("PSD")
    ax.set_title("Static-bias residual PSD by upsampling method")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_panel(
    *,
    output_path: Path,
    target: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [target] + list(predictions.values())
    labels = ["target"] + list(predictions.keys())
    vmax = float(max(np.max(np.abs(arr)) for arr in arrays))
    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 4), constrained_layout=True)
    for ax, label, arr in zip(np.ravel(axes), labels, arrays):
        im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Static-bias predictions")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(
    *,
    output_path: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    psd_path: Path,
    panel_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_hf = max(metrics_rows, key=lambda row: float(row["high_frequency_retention"]))
    best_rmse = min(metrics_rows, key=lambda row: float(row["rmse"]))
    lines = [
        "# Upsampling Method Findings",
        "",
        "This diagnostic trains the same static-bias U-Net target with three decoder",
        "upsampling choices. It does not add these choices to the main pipeline.",
        "",
        f"- Best RMSE: `{best_rmse['upsampling']}` (`{best_rmse['rmse']:.4f}` deg C)",
        f"- Best HF retention: `{best_hf['upsampling']}` (`{best_hf['high_frequency_retention']:.4f}`)",
        "",
        "| Upsampling | RMSE | MAE | HF retention | 4-8km | 8-16km | 16-32km |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['upsampling']} | {row['rmse']:.4f} | {row['mae']:.4f} | "
            f"{row['high_frequency_retention']:.4f} | {row['retention_4-8km']:.4f} | "
            f"{row['retention_8-16km']:.4f} | {row['retention_16-32km']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"PSD plot: `{psd_path}`.",
            f"Prediction panels: `{panel_path}`.",
            "",
            "The useful question is not only which method has lower RMSE. A method is",
            "more interesting for this project if it retains more fine-scale power",
            "without badly damaging RMSE or creating obvious artifacts.",
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
        input_set="core4_topo",
        static_covariate_path=str(static_path),
        verbose=False,
    )
    _train_set, _val_set, train_indices, _val_indices = split_dataset(dataset, float(args.val_fraction), int(args.split_seed))
    x, _y = dataset[int(args.sample_index)]
    x = x.unsqueeze(0).to(device)
    target = compute_static_bias(dataset, train_indices, device)
    input_mean_np, input_std_np = compute_input_stats(dataset, train_indices)
    x_model = normalize_input_batch(
        x,
        torch.tensor(input_mean_np, device=device),
        torch.tensor(input_std_np, device=device),
    )
    predictions: Dict[str, np.ndarray] = {}
    metric_rows: List[Dict[str, Any]] = []
    for mode in ("bilinear", "convtranspose", "pixelshuffle"):
        pred_np, metrics = train_one(mode=mode, args=args, x=x, x_model=x_model, target=target, device=device)
        predictions[mode] = pred_np
        metric_rows.append(metrics)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_dir / "summary.json").write_text(json.dumps(metric_rows, indent=2) + "\n", encoding="utf-8")
    target_np = target.squeeze().detach().cpu().numpy()
    for mode, pred in predictions.items():
        np.save(output_dir / f"{mode}_prediction.npy", pred)
    np.save(output_dir / "static_bias_target.npy", target_np)
    psd_path = repo_path(args.psd_image_out)
    panel_path = repo_path(args.panel_image_out)
    save_psd_plot(output_path=psd_path, target=target_np, predictions=predictions, spacing_km=float(args.grid_spacing_km))
    save_panel(output_path=panel_path, target=target_np, predictions=predictions)
    write_doc(
        output_path=repo_path(args.doc_out),
        metrics_rows=metric_rows,
        psd_path=psd_path.relative_to(PROJECT_ROOT),
        panel_path=panel_path.relative_to(PROJECT_ROOT),
    )
    print(json.dumps(metric_rows, indent=2))


if __name__ == "__main__":
    main()
