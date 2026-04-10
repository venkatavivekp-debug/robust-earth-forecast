from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def upsample_latest_era5(x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Nearest temporal persistence input: latest ERA5 frame upsampled to target grid."""
    if x.dim() == 5:
        latest = x[:, -1, :, :, :]
    elif x.dim() == 4:
        latest = x[:, -1:, :, :]
    else:
        raise ValueError(f"Unsupported ERA5 tensor shape: {tuple(x.shape)}")

    return F.interpolate(latest, size=target_size, mode="bilinear", align_corners=False)


@dataclass
class GlobalLinearBaseline:
    slope: float
    intercept: float

    def predict(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        base = upsample_latest_era5(x, target_size)
        return self.slope * base + self.intercept


def fit_global_linear_baseline(
    dataset: Sequence,
    indices: Iterable[int],
    *,
    max_points: int = 400_000,
) -> GlobalLinearBaseline:
    """Fit y = a*x + b using aggregated pixels from training samples."""
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    count = 0

    for idx in indices:
        x, y = dataset[idx]
        x_b = x.unsqueeze(0)
        y_b = y.unsqueeze(0)

        x_up = upsample_latest_era5(x_b, target_size=(y.shape[-2], y.shape[-1]))
        x_flat = x_up.reshape(-1)
        y_flat = y_b.reshape(-1)

        if x_flat.numel() > max_points:
            step = max(1, x_flat.numel() // max_points)
            x_flat = x_flat[::step]
            y_flat = y_flat[::step]

        sum_x += float(x_flat.sum().item())
        sum_y += float(y_flat.sum().item())
        sum_xx += float((x_flat * x_flat).sum().item())
        sum_xy += float((x_flat * y_flat).sum().item())
        count += int(x_flat.numel())

    if count == 0:
        raise RuntimeError("No points available to fit linear baseline")

    denom = (count * sum_xx) - (sum_x * sum_x)
    if abs(denom) < 1e-8:
        slope = 1.0
        intercept = 0.0
    else:
        slope = ((count * sum_xy) - (sum_x * sum_y)) / denom
        intercept = (sum_y - slope * sum_x) / count

    return GlobalLinearBaseline(slope=float(slope), intercept=float(intercept))
