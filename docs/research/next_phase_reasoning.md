# Next Phase Reasoning: Terrain-Conditioned Residuals

This note fixes the research question before the next run. The project should not move to temporal models or new architecture until the spatial reconstruction question is cleaner.

## What ERA5 -> PRISM needs beyond interpolation

ERA5 gives the broad atmospheric state: daily near-surface temperature, low-level winds, and surface pressure over a coarse grid. Upsampling ERA5 is a useful baseline because most day-to-day temperature variability is already there.

PRISM adds something different. It blends observations and terrain-aware climatological structure, so the target contains local gradients tied to elevation, ridges, valleys, slope exposure, and station-informed spatial corrections. Those features are not explicit in the current ERA5 channels. Interpolation can resize the ERA5 field, but it cannot create missing terrain-conditioned structure.

The actual task is therefore not "make a bigger image." It is coarse atmospheric field plus static spatial context -> PRISM-scale local temperature field.

## Why U-Net helped

The no-skip encoder-decoder baseline compressed spatial information through a bottleneck and then decoded a smooth field. A skip-connected U-Net improved the controlled spatial benchmark because it keeps multi-scale spatial features available during reconstruction. That matches the project diagnosis and the broader weather-postprocessing direction in Hu et al. (2023), where a U-Net-style model is used for structured meteorological postprocessing and is evaluated beyond one scalar score.

U-Net helped, but it did not solve the problem. The sharpness diagnosis still shows weak gradient, local contrast, and high-frequency detail.

## Why topography helped

Topography supplies static information that ERA5 does not resolve at PRISM scale. Elevation, slope, aspect, and terrain-gradient magnitude are physically connected to local temperature structure. In the seed-stability check, both terrain variants beat the no-topography U-Net on RMSE across seeds 42, 7, and 123.

This supports Professor Hu's suggestion that terrain should be tested before adding temporal complexity. It also fits the GAIM direction: geospatial AI should use Earth-data context rather than treating the task as image translation alone.

## Why blur remains

The direct U-Net still predicts the full PRISM field. With MSE/L1-style objectives and limited data, it can reduce average error by learning a smooth reconstruction. The high-frequency ratio remains near zero for direct U-Net topography runs, even when RMSE improves. That means topography helps the large and medium-scale structure, but the model still does not recover fine PRISM detail well.

Longer training gave only mild gains, so undertraining is not a sufficient explanation. The remaining issue is more likely a combination of missing spatial context, direct-field learning, loss/verification mismatch, and limited regional samples.

## Why residual prediction is physically meaningful

Residual prediction changes the target from:

```text
PRISM temperature
```

to:

```text
PRISM temperature - upsampled ERA5 temperature
```

That residual is closer to the physical correction term: terrain-conditioned local adjustment beyond the coarse ERA5 field. It asks the model to learn where PRISM differs from the broad atmospheric background instead of relearning the whole temperature image.

The first seed-42 check supports this idea: residual topography lowered RMSE and recovered more gradient/detail signal than direct topography. It did not remove border degradation, so it needs the same multi-seed stability check before becoming the next main result.

## Why ConvLSTM stays postponed

ConvLSTM answers a different question: whether temporal context improves prediction. The current failure mode is still spatial: blur, weak high-frequency detail, and border degradation. Adding recurrence now could hide the spatial problem behind extra temporal capacity. It would also mix two variables at once: target formulation and temporal architecture.

ConvLSTM should remain archived until the spatial residual/topography baseline is stable and diagnosed.

## Hypothesis

Terrain-conditioned residual reconstruction should improve PRISM-scale spatial reconstruction relative to direct terrain-conditioned U-Net because the model learns the local correction from ERA5 to PRISM rather than the full field.

## Controlled test

Compare only:

1. U-Net `core4_topo_h3`, direct target mode;
2. U-Net `core4_topo_h3`, residual target mode.

Hold fixed:

- medium dataset;
- seeds 42, 7, 123;
- same split protocol;
- same U-Net architecture;
- same optimizer, learning rate, training budget, and diagnostics;
- same DEM-derived static features.

## Support criteria

The hypothesis is supported if residual topography gives:

- lower mean RMSE and MAE across seeds;
- comparable or lower bias;
- comparable or higher correlation;
- better gradient, high-frequency, or local-contrast ratios;
- no major degradation in border RMSE;
- visually less smooth prediction panels.

## Rejection criteria

The hypothesis is rejected or left inconclusive if residual topography:

- improves only one seed and not the mean;
- lowers training loss but not validation metrics;
- worsens border RMSE or border/center ratio enough to offset full-image gains;
- creates artifacts in prediction/error maps;
- does not improve gradient, high-frequency, or local-contrast diagnostics.

Either result is useful. If residual helps across seeds, the next phase can diagnose residual terrain errors by elevation/slope bins. If it does not, the project should return to data scale, loss design, or static-feature alignment rather than adding temporal models.
