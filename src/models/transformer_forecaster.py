import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, nhead=4, num_layers=2):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x):
        # x shape: (B, T, H, W, C)

        B, T, H, W, C = x.shape

        x = x.permute(0,1,4,2,3)   # B T C H W
        x = x.reshape(B*T, C, H, W)

        h = self.spatial_encoder(x)

        _, C2, H2, W2 = h.shape
        h = h.reshape(B, T, C2, H2*W2).mean(-1)

        h = self.transformer(h)

        h = h[:, -1]

        h = h.unsqueeze(-1).unsqueeze(-1).expand(-1, C2, H2, W2)

        out = self.head(h)

        return out.permute(0,2,3,1)
