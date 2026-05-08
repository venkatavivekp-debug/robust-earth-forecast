from .baselines import GlobalLinearBaseline, fit_global_linear_baseline, upsample_latest_era5
from .cnn_downscaler import CNNDownscaler, EncoderDecoderBaseline, PlainEncoderDecoder
from .convlstm_downscaler import ConvLSTMDownscaler
from .unet_downscaler import UNetDownscaler

__all__ = [
    "CNNDownscaler",
    "ConvLSTMDownscaler",
    "EncoderDecoderBaseline",
    "GlobalLinearBaseline",
    "PlainEncoderDecoder",
    "UNetDownscaler",
    "fit_global_linear_baseline",
    "upsample_latest_era5",
]
