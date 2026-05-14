# Recoverable Scale Analysis

This final diagnostic asks which spatial wavelengths are actually recoverable
from the current ERA5 + topography inputs and residual U-Net pipeline. It uses
the medium seed-42 validation split and saved checkpoints; no model was
retrained.

Compared fields:

- PRISM target
- ERA5 bilinear baseline
- lapse-rate corrected ERA5
- residual terrain U-Net with bilinear decoder
- residual terrain U-Net with PixelShuffle decoder

## Results

Band-pass correlation is the clearest readout here. Values near 1 mean the
spatial component matches PRISM well; values near 0 mean that scale is not being
reconstructed.

| Wavelength band | ERA5 bilinear corr | Lapse-rate corr | U-Net bilinear corr | U-Net PixelShuffle corr | PixelShuffle retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| >64 km | 0.741 | 0.742 | 0.904 | 0.904 | 1.026 |
| 32-64 km | 0.676 | 0.730 | 0.830 | 0.830 | 0.859 |
| 16-32 km | 0.652 | 0.703 | 0.815 | 0.815 | 0.843 |
| 8-16 km | 0.669 | 0.727 | 0.832 | 0.833 | 0.865 |
| 4-8 km | -0.056 | 0.167 | 0.109 | 0.109 | 0.019 |

Selected RMSE by band:

| Wavelength band | ERA5 bilinear RMSE | U-Net bilinear RMSE | U-Net PixelShuffle RMSE |
| --- | ---: | ---: | ---: |
| >64 km | 2.085 | 1.110 | 1.124 |
| 32-64 km | 0.463 | 0.305 | 0.307 |
| 16-32 km | 0.334 | 0.222 | 0.224 |
| 8-16 km | 0.274 | 0.177 | 0.178 |
| 4-8 km | 0.053 | 0.052 | 0.052 |

Figures:

- `docs/images/recoverability_curve.png`
- `docs/images/bandpass_reconstruction_panels.png`

## Interpretation

The current residual U-Net reconstructs broad and intermediate scales much
better than bilinear ERA5. From `>64 km` through `8-16 km`, band-pass
correlation is roughly `0.81-0.90` for the learned models.

The collapse happens at `4-8 km`, near PRISM native scale. Both U-Net variants
fall to about `0.109` correlation. PixelShuffle slightly raises 4-8 km energy
retention on full validation (`0.019` vs `0.015` for the bilinear decoder), but
this is still far below PRISM. The diagnostic gain seen on static-bias and
single-sample tests does not become reliable 4 km validation skill.

The lapse-rate baseline carries more 4-8 km energy than ERA5 bilinear, but its
correlation and explained variance are poor. That means adding terrain-shaped
detail alone is not enough; the detail has to match the day-specific PRISM
field.

Final bottleneck: both decoder behavior and information limits matter. The
decoder can suppress learnable detail, and PixelShuffle improves that in
controlled settings. But the full validation residual still lacks enough
recoverable 4-8 km signal from the current ERA5/topography inputs and January
sample. The next meaningful data step is seasonal extension or richer
atmospheric/observational context, not another local architecture tweak.
