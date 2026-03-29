from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDownscaler(nn.Module):
    """Compact CNN baseline for ERA5 -> PRISM spatial downscaling."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        # Extract local spatial patterns on the coarse ERA5 grid.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Map interpolated features to the PRISM temperature target channel.
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        features = self.encoder(x)

        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)

        # Upsample feature maps to the desired high-resolution grid.
        upsampled = F.interpolate(
            features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(upsampled)
