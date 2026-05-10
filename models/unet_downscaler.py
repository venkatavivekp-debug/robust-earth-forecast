from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "reflection") -> None:
        super().__init__()
        if padding_mode not in {"reflection", "zero", "replicate"}:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        def padded_conv(in_ch: int, out_ch: int) -> list[nn.Module]:
            if padding_mode == "reflection":
                return [nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, kernel_size=3)]
            if padding_mode == "replicate":
                return [nn.ReplicationPad2d(1), nn.Conv2d(in_ch, out_ch, kernel_size=3)]
            return [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]

        self.net = nn.Sequential(
            *padded_conv(in_channels, out_channels),
            nn.ReLU(inplace=True),
            *padded_conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels: int, scale: int, padding_mode: str = "reflection") -> None:
        super().__init__()
        if padding_mode not in {"reflection", "zero", "replicate"}:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")
        if scale < 1:
            raise ValueError("scale must be >= 1")
        if padding_mode == "reflection":
            self.expand = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels * scale * scale, kernel_size=3),
            )
        elif padding_mode == "replicate":
            self.expand = nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(channels, channels * scale * scale, kernel_size=3),
            )
        else:
            self.expand = nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.out_channels = int(channels)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        out = self.shuffle(self.expand(x))
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out


class UNetDownscaler(nn.Module):
    """Small spatial U-Net baseline for ERA5 -> PRISM downscaling."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 24,
        padding_mode: str = "reflection",
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if upsample_mode not in {"bilinear", "convtranspose", "pixelshuffle"}:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")
        self.padding_mode = padding_mode
        self.upsample_mode = upsample_mode

        self.enc1 = ConvBlock(in_channels, base_channels, padding_mode=padding_mode)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, padding_mode=padding_mode)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, padding_mode=padding_mode)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, padding_mode=padding_mode)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, padding_mode=padding_mode)
        self.high_res = ConvBlock(base_channels, base_channels, padding_mode=padding_mode)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        if upsample_mode == "convtranspose":
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
            self.up_high = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=4)
        elif upsample_mode == "pixelshuffle":
            self.up2 = PixelShuffleUpsample(base_channels * 4, scale=2, padding_mode=padding_mode)
            self.up1 = PixelShuffleUpsample(base_channels * 2, scale=2, padding_mode=padding_mode)
            self.up_high = PixelShuffleUpsample(base_channels, scale=4, padding_mode=padding_mode)

    @staticmethod
    def _prepare_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b, t * c, h, w)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected input with 4 or 5 dims, got shape {tuple(x.shape)}")

    def _upsample(
        self,
        x: torch.Tensor,
        size: Tuple[int, int],
        layer: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if self.upsample_mode == "bilinear" or layer is None:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if isinstance(layer, PixelShuffleUpsample):
            return layer(x, size=size)
        target_shape = (x.shape[0], layer.out_channels, int(size[0]), int(size[1]))
        try:
            out = layer(x, output_size=target_shape)
        except ValueError:
            out = layer(x)
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self._prepare_input(x)
        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, stride=2))
        b = self.bottleneck(F.avg_pool2d(e2, kernel_size=2, stride=2))

        d2 = self._upsample(b, size=e2.shape[-2:], layer=getattr(self, "up2", None))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._upsample(d2, size=e1.shape[-2:], layer=getattr(self, "up1", None))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        high = self._upsample(d1, size=target_size, layer=getattr(self, "up_high", None))
        return self.out(self.high_res(high))
