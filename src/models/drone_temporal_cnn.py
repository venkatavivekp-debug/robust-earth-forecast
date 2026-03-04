import torch
import torch.nn as nn


class DroneTemporalCNN(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        # spatial feature extractor
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # temporal modeling
        self.temporal = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        self.head = nn.Linear(128, 64)

    def forward(self, x):

        # x shape: (B, T, C, H, W)

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.spatial(x)

        x = x.view(B, T, 128)

        out, _ = self.temporal(x)

        return self.head(out[:, -1])
