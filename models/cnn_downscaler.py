from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDownscaler(nn.Module):
    """Plain encoder-decoder baseline for ERA5 -> PRISM downscaling.

    Accepts either:
    - [B, T, C, H, W] (temporal windows) or
    - [B, C, H, W] (already channel-stacked)

    Historical checkpoints and CLIs still use the name ``cnn``. This module
    does not implement U-Net skip connections.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _prepare_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b, t * c, h, w)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected input with 4 or 5 dims, got shape {tuple(x.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = self._prepare_input(x)
        features = self.encoder(x)

        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)

        upsampled = F.interpolate(
            features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(upsampled)


PlainEncoderDecoder = CNNDownscaler
EncoderDecoderBaseline = CNNDownscaler
