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

## Conclusion

The dominant bottleneck is a combination of **information content** and **data scale**, with architecture as a secondary factor.

Evidence:

- Terrain-linear residual R2 is only `0.0543`, so simple geometry explains little of the ERA5 -> PRISM residual.
- Lapse-rate correction barely improves bilinear ERA5 (`3.4525` vs `3.4580` RMSE), so the missing signal is not a simple elevation correction.
- The corrected one-sample overfit improves smoothly to `0.1875` RMSE but does not collapse to zero, suggesting no obvious scheduler/padding/gradient failure.
- Skip connections improve RMSE and high-frequency retention on the fixed sample, so architecture affects the learnable part.
- The small sample count is still too low to learn stable day-to-day PRISM residual behavior.

Current falsifiable read:

> The pipeline is not obviously broken by normalization, scheduler, or dead skip gradients. The residual target contains a small learnable spatial component that skip connections help recover, but most residual variance is not explained by simple terrain geometry. The main limitation is the information gap between ERA5 and PRISM, amplified by limited data scale.

Next work should test data coverage and product-level residual structure before adding another architecture.
