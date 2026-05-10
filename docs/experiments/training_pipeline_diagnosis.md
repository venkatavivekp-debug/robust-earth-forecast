# Training Pipeline Diagnosis

This pass checks whether the current terrain-aware residual U-Net is failing because of a basic training/pipeline issue or because the ERA5 -> PRISM target is weakly determined by the available inputs.

## Physics Baselines

The no-training baselines give the first constraint.

| Baseline | RMSE | MAE | Read |
| --- | ---: | ---: | --- |
| lapse-rate corrected ERA5 | 3.4525 | 2.4741 | best no-training baseline, but only barely |
| bilinear ERA5 t2m | 3.4580 | 2.4759 | almost identical to lapse-rate correction |
| previous-day PRISM | 3.8180 | 3.1278 | worse for this January subset |
| leave-one-out January PRISM mean | 4.2846 | 3.5636 | not competitive |

The lapse-rate result is the important one. A fixed `6.5 deg C/km` correction improves bilinear ERA5 by only `0.0055` RMSE, so the residual is not mainly a simple elevation correction.

Pixelwise ERA5 explained variance is mixed:

- mean R2: `0.2723`
- median R2: `0.5956`
- fraction positive R2: `0.8878`

ERA5 explains broad temporal structure in much of the domain, but the negative-R2 regions show that some PRISM variability is not well captured by simple bilinear ERA5.

## Residual Decomposition

Residual is `PRISM - bilinear ERA5 t2m`.

| Diagnostic | Value | Read |
| --- | ---: | --- |
| terrain-linear residual R2 | 0.0543 | elevation/slope/aspect explain only 5.4% of residual variance |
| unexplained by terrain-linear model | 94.6% | most residual is not simple terrain geometry |
| static bias map variance fraction | 0.2744 | repeated spatial bias is more informative than linear terrain alone |
| mean temporal residual std | 2.3281 deg C | residual still varies strongly day to day |

The model is not being asked to learn a clean DEM correction. It is being asked to learn a residual whose simple terrain-predictable component is small and whose day-to-day component remains large.

## Corrected One-Sample Overfit

Setup:

- U-Net with skip connections
- `core4_topo`
- residual target mode
- reflection padding
- base width 24
- fixed LR `1e-3`
- no scheduler
- seed 42
- sample index 0
- normalization fit on the full training split (`70` indices)

Residual target context for the fixed sample:

| Mean | Std | Min | Max |
| ---: | ---: | ---: | ---: |
| -1.4401 deg C | 1.6933 deg C | -7.3277 deg C | 3.5697 deg C |

Training curve checkpoints:

| Epoch | Loss | RMSE |
| ---: | ---: | ---: |
| 1 | 5.0485 | 2.2110 |
| 100 | 0.4227 | 0.6143 |
| 200 | 0.1764 | 0.3847 |
| 500 | 0.0807 | 0.2508 |
| 900 | 0.0507 | 0.1939 |
| 1000 | 0.0478 | 0.1875 |

The curve does not show a sharp early plateau. It decays smoothly through 1000 epochs, with only `0.0064` RMSE improvement in the last 100 epochs. That looks like slow convergence toward a nonzero floor, not a scheduler failure or dead-gradient bug.

Gradients were nonzero in encoder, bottleneck, and decoder blocks throughout the run. The model did not memorize the sample perfectly, but the training path is active.

## Skip-Connection Ablation

Same sample, seed, normalization, LR, epochs, residual target, and reflection padding. The only change is whether decoder skip tensors are used or zeroed.

| Model | RMSE | MAE | Border RMSE | Interior RMSE | Border/Interior | HF retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| skip U-Net | 0.1875 | 0.1260 | 0.2090 | 0.1745 | 1.1982 | 0.7834 |
| no-skip U-Net | 0.2429 | 0.1752 | 0.2827 | 0.2177 | 1.2988 | 0.7491 |

Skip connections help on the fixed sample. They lower RMSE, border RMSE, and border/interior ratio, and they retain more high-frequency power. This supports Professor Hu's point that the skip path matters for spatial reconstruction.

The effect is real but bounded. Skips improve the learnable reconstruction pathway; they do not change the fact that the broader residual has weak terrain-linear predictability.

## Residual Collapse Diagnostic

The visual benchmark suggested that the U-Net output was close to bilinear ERA5, so we tested whether the residual branch was simply predicting near-zero corrections.

| Diagnostic | Value |
| --- | ---: |
| mean \|predicted residual\| | 1.7364 deg C |
| mean \|target residual\| | 1.9428 deg C |
| predicted/target residual magnitude ratio | 0.8938 |
| Pearson r between mean residual maps | 0.9632 |

By the strict rule `ratio < 0.2`, residual collapse is **not** confirmed. The model is not returning a zero residual field. It learns a spatially coherent mean residual pattern that is close to the target mean residual map.

This changes the interpretation. The full temperature prediction can still look close to bilinear ERA5 because the learned residual is a smoothed stable component plotted on top of a broad temperature field. The remaining issue is not "the residual head is dead"; it is that daily residual variation is weakly predictable from the current inputs.

## Static Bias Learning

The static-bias test removes day-to-day residual variability. The target is the training-split temporal mean of `PRISM - ERA5_bilinear`, repeated for one input sample.

| Diagnostic | Value |
| --- | ---: |
| final RMSE | 0.2343 deg C |
| final MAE | 0.1553 deg C |
| pixelwise correlation | 0.9861 |
| target std | 1.4101 deg C |
| prediction std | 1.3906 deg C |
| high-frequency retention | 0.2050 |

The model can learn the stable residual map. That argues against a broken decoder or unusable skip pathway. It also explains why daily residual training can look smooth: the stable spatial part is learnable, but the daily anomaly around that stable map remains much harder.

The low high-frequency retention is still important. Even when the mean residual map is learned with high correlation, the finest detail is muted. The model is learning broad/static spatial correction better than sharp local PRISM detail.

## Feature Map Structure

Feature-map visualization was run on the trained residual U-Net checkpoint.

| Layer | Mean-map spatial std | Mean-map range |
| --- | ---: | ---: |
| enc1 | 0.1088 | 0.5565 |
| enc2 | 0.1246 | 0.6288 |
| bottleneck | 0.0295 | 0.1045 |
| decoder first upsample | 0.3456 | 1.8696 |
| decoder second upsample | 0.8361 | 4.4838 |

The encoder maps are spatially structured, and the decoder maps become more spatially variable after skip fusion. This rules out the simple failure mode where the encoder is ignoring the input/topography and producing uniform activations.

## Data Scale Implication

Using residual std `1.6933 deg C` and terrain-linear R2 `0.0543`:

- terrain-predictable residual std is about `0.39 deg C`;
- unexplained residual std is about `1.65 deg C`;
- signal-to-noise ratio is about `0.24`.

A crude two-sigma detection calculation gives roughly `70` independent samples just to detect that small terrain-predictable component. A spatial U-Net needs more than that because the target is a map, the pixels are correlated, and January weather regimes are not independent. The current dataset is therefore too small to support reliable daily fine-scale residual reconstruction.

## Conclusion

The dominant bottleneck is **daily residual information and data scale**, with architecture as a secondary factor.

Evidence:

- Terrain-linear residual R2 is only `0.0543`, so simple geometry explains little of the ERA5 -> PRISM residual.
- Lapse-rate correction barely improves bilinear ERA5 (`3.4525` vs `3.4580` RMSE), so the missing signal is not a simple elevation correction.
- The corrected one-sample overfit improves smoothly to `0.1875` RMSE but does not collapse to zero, suggesting no obvious scheduler/padding/gradient failure.
- Skip connections improve RMSE and high-frequency retention on the fixed sample, so architecture affects the learnable part.
- Residual collapse is not confirmed (`0.8938` predicted/target magnitude ratio), so the residual head is not dead.
- Static bias learning succeeds (`0.2343` RMSE, `0.9861` correlation), so the model can learn stable spatial residual structure.
- Feature maps are spatially structured, so the encoder is not ignoring terrain/input information.
- The small sample count is still too low to learn stable day-to-day PRISM residual behavior.

Current falsifiable read:

> The pipeline is not obviously broken by normalization, scheduler, padding, dead skip gradients, or a dead residual head. The model can learn the stable mean residual map, and skip connections help on the learnable component. The primary limitation is that daily `PRISM - ERA5_bilinear` anomalies are weakly constrained by the current ERA5/topography inputs and the sample count is too small to learn that low-SNR component reliably.

Next work should extend data coverage into a terrain-informative season and rerun the physics/residual diagnostics before adding another architecture.
