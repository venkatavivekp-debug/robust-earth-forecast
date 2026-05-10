# Data Scale Analysis

The residual diagnostics put a hard scale on the problem. For the January Georgia setup:

- residual std: `1.6933 deg C`
- terrain-linear R2: `0.0543`
- terrain-predictable residual std: `sqrt(0.0543) * 1.6933 ~= 0.39 deg C`
- unexplained residual std: `sqrt(1 - 0.0543) * 1.6933 ~= 1.65 deg C`
- signal-to-noise ratio: `0.39 / 1.65 ~= 0.24`

That is a weak signal. A rough detection calculation gives `(2 / 0.24)^2 ~= 70` independent samples just to detect the terrain-predictable component at a basic two-sigma level. A U-Net is not estimating one scalar. It is fitting a spatial map with correlated pixels, weather-regime dependence, and product-level PRISM behavior. The practical sample need is therefore higher than 70 independent days, likely several hundred effective samples before the daily residual target becomes stable enough for reliable fine-scale learning.

## Seasonality

January is not the best season for terrain signal in this small dataset. Synoptic frontal passages can move temperature over the domain as a broad field, which raises the ERA5-scale signal and weakens the relative contribution of local terrain geometry.

Terrain-linked temperature structure should be easier to detect when local radiation and boundary-layer processes matter more:

- warm-season daytime heating can strengthen slope/aspect and land-surface contrasts;
- shoulder seasons may expose elevation gradients without only frontal control;
- stable clear nights can strengthen cold-air drainage and valley pooling, but only when the atmospheric state supports it.

The next data extension should therefore target a terrain-informative season, not just more January days. A practical first extension is to add a summer or shoulder-season block from 2023 and recompute the terrain R2, residual collapse ratio, and static-bias learnability. If that raises the terrain-predictable fraction, then a multi-year version of the same season is justified.

## Recommendation

The most justified next data step is:

1. add a non-January season with stronger terrain-temperature structure;
2. rerun the physics baselines and residual decomposition before retraining;
3. only scale training if the terrain-predictable residual fraction increases.

More January samples may improve statistical stability, but the current R2 suggests that January residuals are not the cleanest target for learning terrain-aware fine detail.
