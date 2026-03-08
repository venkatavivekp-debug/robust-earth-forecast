import torch
import torch.nn as nn

class ClimateTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Conv2d(1, 64, kernel_size=4, stride=4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.decoder = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=4
        )

    def forward(self, x):

        x = self.embed(x)

        b, c, h, w = x.shape

        x = x.flatten(2).permute(2, 0, 1)

        x = self.transformer(x)

        x = x.permute(1, 2, 0).reshape(b, c, h, w)

        x = self.decoder(x)

        return x
