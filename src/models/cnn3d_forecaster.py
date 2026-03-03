import torch
import torch.nn as nn

class CNN3DForecaster(nn.Module):
    """
    Input:  (B, T, H, W, C)
    Output: (B, H, W, 1)
    """
    def __init__(self, in_channels: int, t_in: int):
        super().__init__()
        self.t_in = t_in

        # We'll use Conv3d on (B, C, T, H, W)
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Collapse time dimension to 1 using kernel=(t_in,1,1)
        self.head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(t_in, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1),   # output channel = 1 (t2m)
        )

    def forward(self, x):
        # x: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        h = self.backbone(x)
        y = self.head(h)      # (B, 1, 1, H, W)
        y = y.squeeze(2)      # (B, 1, H, W)
        y = y.permute(0, 2, 3, 1)  # (B, H, W, 1)
        return y
