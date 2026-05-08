# Boundary Ablation Protocol

This phase follows the boundary artifact audit. The earlier diagnostics showed that border RMSE stays higher than center RMSE for persistence, PlainEncoderDecoder, and U-Net. Professor Hu also pointed directly at padding, convolution, and deconvolution settings, so the next check should isolate boundary handling before adding topography or temporal modeling.

## Question

For the current ERA5 -> PRISM spatial reconstruction setup, are remaining border errors more consistent with padding behavior, decoder upsampling behavior, missing outside-domain context, or the coarse ERA5 input itself?

## Fixed setup

The ablation keeps the rest of the experiment fixed:

- dataset: medium ERA5/PRISM split
- input set: `core4`
- history: 3 days
- target mode: direct PRISM temperature
- model depth/channels: existing U-Net baseline
- optimizer, epochs, batch size, loss, scheduler, normalization: same training path
- validation split and seed: same as the controlled spatial benchmark
- metrics: RMSE, MAE, border RMSE, center RMSE, edge RMSE, corner RMSE, and error by distance from the nearest boundary

Only one boundary-handling choice is changed at a time.

## Ablations

| Variant | Padding | Decoder upsampling | Reason |
| --- | --- | --- | --- |
| `reflection_bilinear` | reflection | bilinear + conv | Current U-Net baseline |
| `zero_bilinear` | zero | bilinear + conv | Tests whether zero padding depresses edge predictions |
| `replicate_bilinear` | replicate | bilinear + conv | Tests whether copying edge values changes boundary behavior |
| `reflection_convtranspose` | reflection | ConvTranspose2d decoder | Tests whether learnable decoder upsampling changes artifacts |

The final PRISM target shape is not an integer multiple of every intermediate U-Net feature map. The ConvTranspose2d variant therefore uses transposed convolution in the decoder path and still enforces exact target-size alignment at the output. This is not a full architecture change; it is a controlled decoder-upsampling check.

## What outcomes would mean

Padding-related artifacts are supported if zero, reflection, and replicate padding produce different border/center ratios while full-image RMSE stays similar.

Interpolation-related artifacts are supported if ConvTranspose2d changes boundary profiles or edge-specific errors relative to the bilinear baseline.

Missing-context limitations are supported if all variants keep similar border degradation, especially when persistence already shows high border error. In that case the clipped domain and absent outside-region context are likely part of the problem.

No single ablation proves the cause. The goal is to decide whether a padding/upsampling ablation is worth carrying forward, or whether the next controlled step should move toward physical context such as topography after boundary behavior is understood.
