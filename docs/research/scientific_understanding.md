# Scientific Understanding: ERA5 -> PRISM Reconstruction

The active research question is:

> What limits fine-scale spatial reconstruction in terrain-aware ERA5 -> PRISM downscaling?

The current evidence says this is not mainly a search for a larger architecture. It is a conditional reconstruction problem with a severe information gap between ERA5 and PRISM.

## Information Deficit

ERA5 is roughly 31 km resolution. PRISM is roughly 4 km. ERA5 carries the broad atmospheric state: near-surface temperature, wind, pressure, moisture, and synoptic-scale structure. PRISM contains station-informed, terrain-aware spatial detail that is not explicitly present in the ERA5 grid.

Fine-scale temperature variation can come from:

- **Elevation lapse rate.** Higher terrain is often cooler. A DEM can provide elevation, but the actual lapse rate changes with air mass, stability, humidity, and time of day.
- **Cold-air pooling.** Valleys can be colder than surrounding slopes under calm stable conditions. Terrain identifies where pooling is possible; atmospheric state determines whether it happens.
- **Slope/aspect radiation effects.** Slope and aspect influence solar exposure. A DEM gives the geometry, but clouds, season, snow, and wind mixing control the realized temperature effect.
- **Land-cover transitions.** Forest, urban, cropland, and water-adjacent areas can produce local thermal differences. These are mostly absent from the current inputs.
- **Proximity to water.** Lakes, rivers, wetlands, and coastal influence can damp or shift local temperatures. This is not recoverable from ERA5 temperature alone.

The model is therefore not just sharpening an image. It is trying to infer a fine PRISM field from coarse atmospheric predictors plus limited static terrain information.

## Why MSE Produces Smooth Fields Here

MSE predicts a conditional mean. In this problem that has a physical meaning.

A single coarse ERA5 `t2m` value can be consistent with several PRISM-scale realizations inside the same grid cell: a valley cold pool may or may not occur; ridge/valley contrast can vary; local radiation effects can be stronger or weaker. With limited data, the model cannot know which fine-scale realization occurred from ERA5/topography alone. Under MSE, the statistically safest answer is the mean of plausible realizations.

That mean is spatially smooth. A smooth field can lower RMSE while still missing terrain boundaries, local gradients, and high-frequency PRISM detail. This is not automatically a training bug. It is the posterior-mean behavior expected when the inputs do not determine the target.

Sharper reconstruction requires at least one of:

- enough data to separate similar ERA5 states with different local outcomes;
- additional physically relevant inputs;
- a target/loss/evaluation setup that does not reward averaging away local spread.

## Revised Understanding After Physics Baseline Analysis

The physics baselines changed the interpretation.

| Diagnostic | Result | Meaning |
| --- | ---: | --- |
| Bilinear ERA5 RMSE | 3.4580 | Broad ERA5 temperature alone is a weak but meaningful baseline for January samples. |
| Lapse-rate ERA5 RMSE | 3.4525 | A fixed 6.5 deg C/km elevation correction barely improves ERA5. |
| Terrain-linear residual R2 | 0.0543 | Elevation, slope, and aspect explain only 5.4% of `PRISM - ERA5_bilinear` residual variance. |
| Static residual bias fraction | 0.2744 | A repeated spatial bias map explains more than linear terrain geometry, but most residual variance remains time-varying. |

The lapse-rate result is important: if the missing PRISM structure were mostly simple elevation correction, lapse-rate ERA5 should beat bilinear ERA5 clearly. It does not. The improvement is only `0.0055` RMSE.

The terrain-linear R2 is more direct. A linear terrain model explains `0.0543` of residual variance, leaving about `94.6%` unexplained by simple elevation/slope/aspect geometry. This does not mean terrain is irrelevant; nonlinear terrain effects and interactions with atmospheric state may still matter. But it does mean the current January residual is not mainly a simple DEM-correctable signal.

The revised bottleneck is the information gap between ERA5 and PRISM as data products. PRISM contains observation-informed local structure and product-methodology behavior that ERA5 plus simple terrain geometry cannot determine.

## Learning Ceiling With 18 Samples

The small January setup has about 18 usable aligned daily samples. At that scale, the model can at best learn:

1. the large-scale ERA5 relationship already present in bilinear `t2m`;
2. a small terrain-correlated residual component;
3. perhaps a repeated static bias pattern if it is stable across the sample.

It cannot reliably learn the day-to-day conditions under which valley pooling, radiation effects, or local PRISM corrections appear. The data are too sparse to distinguish many atmospheric regimes, and the inputs do not fully encode the local processes.

So the ceiling is low: architecture can help recover the small structured component, but it cannot invent the 94.6% residual variance that is not explained by the simple terrain baseline.

## What Architecture Can and Cannot Fix

Skip connections still matter. A no-skip encoder-decoder can erase spatial layout in the bottleneck, while a proper U-Net passes high-resolution feature maps into the decoder. That should help preserve the terrain-correlated component that is present.

But skip connections cannot change the information content of the inputs. If ERA5/topography do not determine most of the PRISM residual, a better decoder can only improve the learnable fraction. It cannot make the target fully predictable.

This is the current interpretation:

- U-Net and residual learning are useful controls.
- Topography is physically justified, but the simple terrain signal is small.
- Lapse-rate correction is not enough.
- More architecture does not solve missing information.
- The immediate task is to verify training behavior and isolate how much of the small learnable residual component the model actually captures.

## What Success Means Now

At this data scale, success is not state-of-the-art downscaling. A credible result should show:

1. the pipeline can overfit one fixed PRISM target or approach a clear nonzero information/optimization floor;
2. residual learning beats direct full-field reconstruction on the same sample;
3. skip connections improve high-frequency/border behavior relative to bypassed skips;
4. physics baselines define what ERA5 and simple terrain already explain;
5. the conclusion separates pipeline bugs, architecture limits, data scale, and input-information limits.

If the model cannot learn the small terrain/static component, debug training and target formulation. If it learns that component but remains smooth, the dominant limitation is information and data scale.
