import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    """
    Input:  (B, T, H, W, C)
    Output: (B, H, W, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, T, H, W, C = x.shape

        # (B, T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3)

        h = torch.zeros((B, self.hidden_channels, H, W), device=x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        y = self.head(h)          # (B, out_channels, H, W)
        y = y.permute(0, 2, 3, 1) # (B, H, W, out_channels)

        return y
