import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=3):
        super().__init__()

        self.embedding = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(model_dim, input_dim)

    def forward(self, x):

        B, T, H, W, C = x.shape

        x = x.view(B, T, -1)

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:, -1]

        out = self.head(x)

        return out
