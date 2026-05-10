# Decoder Pathway Findings

This diagnostic trains the static-bias U-Net setup again and inspects the
decoder pathway. Intermediate decoder tensors are activation maps, not
calibrated predictions, so their spectra are compared after z-scoring.
The final output is the actual static-bias residual prediction.

- Static-bias RMSE: `0.2328` deg C
- Static-bias banded 4-32 km retention: `0.7639`
- First final-output band below 50% retention: `4-8km`
- U-Net final-temperature PSD lower than bilinear ERA5 in any band: `yes`

Final residual PSD retention:

| Band | Target power | Prediction power | Retention |
| --- | ---: | ---: | ---: |
| 4-8km | 104.851 | 7.26576 | 0.0693 |
| 8-16km | 1021.87 | 766.687 | 0.7503 |
| 16-32km | 6046.07 | 4705.53 | 0.7783 |
| 32km+ | 564295 | 564707 | 1.0007 |

Decoder stage spectral fractions:

| Stage | 4-8km | 8-16km | 16-32km | 32km+ |
| --- | ---: | ---: | ---: | ---: |
| decoder_block_1 | 0.0077 | 0.0336 | 0.1530 | 0.8058 |
| decoder_block_2 | 0.0003 | 0.0085 | 0.0591 | 0.9321 |
| pre_output_high_res | 0.0000 | 0.0013 | 0.0077 | 0.9910 |

Final temperature PSD compared with bilinear ERA5:

| Band | Bilinear ERA5 power | U-Net final power | U-Net lower? |
| --- | ---: | ---: | --- |
| 4-8km | 3.12318 | 1.26417 | yes |
| 8-16km | 1195.59 | 1046.93 | yes |
| 16-32km | 6745 | 6324.74 | yes |
| 32km+ | 1.42527e+06 | 1.2146e+06 | yes |

PSD figure: `docs/images/decoder_psd_by_stage.png`.

The main read is whether the final residual prediction keeps power in the
near-grid-scale bands. Low retention there is consistent with the decoder
acting as a low-pass reconstruction path even when the static target is
stable and learnable.
