from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMDownscaler(nn.Module):
    """Temporal ConvLSTM downscaler for ERA5 history -> PRISM target."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.convlstm = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)

        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"ConvLSTMDownscaler expects [B, T, C, H, W], got shape {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected input channels={self.input_channels}, got {c}")

        h_state = x.new_zeros((b, self.hidden_channels, h, w))
        c_state = x.new_zeros((b, self.hidden_channels, h, w))

        for step in range(t):
            h_state, c_state = self.convlstm(x[:, step], h_state, c_state)

        if target_size is None:
            target_size = (h * 4, w * 4)

        upsampled = F.interpolate(h_state, size=target_size, mode="bilinear", align_corners=False)
        return self.readout(upsampled)
