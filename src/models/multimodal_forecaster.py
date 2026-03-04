import torch
import torch.nn as nn

from src.models.convlstm import ConvLSTMForecaster
from src.remote_sensing.cnn_landcover import LandCoverCNN


class MultimodalForecaster(nn.Module):

    def __init__(self, era5_channels=5):
        super().__init__()

        # Atmospheric model (ERA5 weather data)
        self.atmos_model = ConvLSTMForecaster(era5_channels)

        # Satellite / drone imagery model
        self.land_model = LandCoverCNN()

        # Fusion layer combining both signals
        self.fusion = nn.Sequential(
            nn.Linear(128 + era5_channels, 256),
            nn.ReLU(),
            nn.Linear(256, era5_channels)
        )

    def forward(self, era5, satellite):

        # ERA5 atmospheric features
        era5_features = self.atmos_model(era5)

        # Satellite image features
        sat_features = self.land_model.features(satellite)
        sat_features = sat_features.view(sat_features.size(0), -1)

        # Combine both modalities
        combined = torch.cat([era5_features, sat_features], dim=1)

        # Final prediction
        return self.fusion(combined)
