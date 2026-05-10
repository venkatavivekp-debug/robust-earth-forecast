# Upsampling Method Findings

This diagnostic trains the same static-bias U-Net target with three decoder
upsampling choices. It does not add these choices to the main pipeline.

- Best RMSE: `pixelshuffle` (`0.0781` deg C)
- Best HF retention: `pixelshuffle` (`0.9480`)

| Upsampling | RMSE | MAE | HF retention | 4-8km | 8-16km | 16-32km |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bilinear | 0.2343 | 0.1553 | 0.7515 | 0.0679 | 0.7365 | 0.7659 |
| convtranspose | 0.1441 | 0.0922 | 0.9148 | 0.3202 | 0.8889 | 0.9295 |
| pixelshuffle | 0.0781 | 0.0551 | 0.9480 | 0.5397 | 0.9288 | 0.9584 |

PSD plot: `docs/images/upsampling_comparison_psd.png`.
Prediction panels: `docs/images/upsampling_comparison_panels.png`.

The useful question is not only which method has lower RMSE. A method is
more interesting for this project if it retains more fine-scale power
without badly damaging RMSE or creating obvious artifacts.
