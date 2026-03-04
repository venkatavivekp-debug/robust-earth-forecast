import torch
import torch.nn as nn
from src.models.convlstm import ConvLSTMForecaster
from src.remote_sensing.cnn_landcover import LandCoverCNN


class MultimodalForecaster(nn.Module):

    def __init__(self, era5_channels, num_classes=10):
        super().__init__()

        # Atmospheric model
        self.atmos_model = ConvLSTMForecaster(era5_channels)

        # Satellite model
        self.land_model = LandCoverCNN()

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + era5_channels, 256),
            nn.ReLU(),
            nn.Linear(256, era5_channels)
        )

    def forward(self, era5, satellite):

        # ERA5 atmospheric prediction
        era5_feat = self.atmos_model(era5)

        # Satellite feature extraction
        sat_feat = self.land_model.features(satellite)
        sat_feat = sat_feat.view(sat_feat.size(0), -1)

        # Combine features
        combined = torch.cat([era5_feat, sat_feat], dim=1)

        return self.fusion(combined)
