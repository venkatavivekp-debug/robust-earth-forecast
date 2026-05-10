# PixelShuffle Full-Training Check

This run tests whether the PixelShuffle decoder result from the static-bias and
single-sample diagnostics transfers to the normal medium-data training setting.
The setup is still narrow: U-Net, `core4_topo`, residual target, medium dataset,
80 epochs, learning rate `3e-4`, seeds `42`, `7`, and `123`.

The run used reflection padding, PixelShuffle upsampling, and the same
topography diagnostics used in the earlier residual-topography comparison.
The reference bilinear result is the existing residual-topography seed
stability run: `1.3858 +/- 0.0564` RMSE.

| Seed | RMSE | MAE | Border RMSE | Center RMSE | Grad Ratio | HF Ratio | Local Contrast |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.5112 | 1.2038 | 1.6672 | 1.4564 | 0.6104 | 0.0450 | 0.6897 |
| 7 | 1.3204 | 1.0222 | 1.5213 | 1.2476 | 0.5723 | 0.0563 | 0.6571 |
| 123 | 1.3995 | 1.0665 | 1.6073 | 1.3243 | 0.6109 | 0.0509 | 0.6968 |

| Metric | PixelShuffle mean +/- std | Bilinear residual-topo reference |
| --- | ---: | ---: |
| RMSE | 1.4104 +/- 0.0958 | 1.3858 +/- 0.0564 |
| MAE | 1.0975 +/- 0.0947 | 1.0697 +/- 0.0609 |
| Border RMSE | 1.5986 +/- 0.0733 | 1.5828 +/- 0.0452 |
| Center RMSE | 1.3427 +/- 0.1056 | 1.3146 +/- 0.0622 |
| Border/center ratio | 1.1926 +/- 0.0416 | 1.2048 +/- 0.0294 |
| Gradient ratio | 0.5979 +/- 0.0222 | 0.5665 +/- 0.0194 |
| High-frequency ratio | 0.0507 +/- 0.0057 | 0.0302 +/- 0.0023 |
| Local contrast ratio | 0.6812 +/- 0.0212 | 0.6572 +/- 0.0172 |

## Interpretation

PixelShuffle does not improve full-validation RMSE in this controlled medium
run. It improves detail-oriented diagnostics: gradient ratio, high-frequency
ratio, and local contrast ratio all increase relative to the bilinear reference.
Border/center ratio is slightly lower, but absolute border RMSE is slightly
higher.

So the diagnostic improvement transfers partially. PixelShuffle reconstructs
more local variation, but the daily residual target is still low-SNR and
split-sensitive. The static-bias result showed the decoder can recover
learnable detail; the full-training result shows that better upsampling alone
does not overcome the ERA5/PRISM daily residual information limit.

The next scientifically useful data step is still seasonal extension,
especially months where terrain-temperature structure should be stronger than
January. Another decoder change should not be treated as progress unless it
improves both detail metrics and validation error under the same split/seed
protocol.
