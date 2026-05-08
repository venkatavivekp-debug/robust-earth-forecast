from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDownscaler(nn.Module):
    """Small spatial U-Net baseline for ERA5 -> PRISM downscaling."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.high_res = ConvBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _prepare_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b, t * c, h, w)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected input with 4 or 5 dims, got shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self._prepare_input(x)
        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, stride=2))
        b = self.bottleneck(F.avg_pool2d(e2, kernel_size=2, stride=2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        high = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False)
        return self.out(self.high_res(high))
