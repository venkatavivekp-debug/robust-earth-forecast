# Problem structure notes

ERA5 -> PRISM downscaling is not just image sharpening. The input is a coarse reanalysis field; the target is a fine observation-informed gridded product. A useful model has to correct bias, recover local structure, and avoid inventing detail that is not supported by the predictors.

## Persistence is strong

Latest ERA5 `t2m` already carries the large-scale daily temperature state. After bilinear upsampling it often lands close enough to PRISM that a learned model has to improve a residual, not reconstruct the whole field from scratch. That is why persistence is the baseline to beat rather than a toy comparison.

## What PRISM adds

PRISM is built from station observations with terrain-aware interpolation. It contains local elevation, exposure, coastal/inland, and station-network information that is only indirectly present in ERA5. A small neural net fed only coarse atmospheric fields cannot fully recover those static controls.

## Interpolation is not enough

Bilinear ERA5 preserves the synoptic pattern but smooths local gradients and keeps ERA5 biases. A global linear correction can remove some mean bias, but it still cannot represent spatially varying residuals tied to terrain and observation density.

## Why the borders failed

The old CNN had a weak decoder: same-grid convolutions, smooth feature upsampling, then a shallow high-resolution readout. Near clipped regional edges, padded convolution sees artificial context. The diagnostic showed larger border error than center error, especially for the ConvLSTM residual readout.

## Skip connections and residuals

U-Net skip connections give the decoder direct access to higher-resolution feature structure before the final PRISM-size readout. Residual prediction helped because the model only learns the correction to upsampled ERA5 `t2m`; the physically plausible coarse field stays in the output by construction.

## Spatial before temporal

ConvLSTM is useful only if temporal context adds information beyond a stable spatial reconstruction baseline. Since residual U-Net beat ConvLSTM on the controlled medium spatial check, history-length claims should be retested against that baseline before adding more temporal machinery.

## Topography is probably missing signal

Temperature downscaling over Georgia should depend on elevation and terrain exposure. No DEM/static field is currently used, so topographic effects can only be approximated from coarse atmospheric variables and the limited PRISM samples. A real DEM channel is a cleaner next test than a larger temporal model.

## Current limits

- Short regional sample; split variance is still visible.
- Georgia-only domain; no transfer claim.
- ERA5 and PRISM differ in grid, physics, and observation influence.
- Border diagnostics are based on validation splits, not an independent external holdout.
- Residual U-Net is a better spatial baseline, but the border/center ratio is still above 1.
