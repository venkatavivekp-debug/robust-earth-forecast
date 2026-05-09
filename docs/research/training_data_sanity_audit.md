# Training/Data Sanity Audit

This audit was added after the U-Net/topography/residual experiments still looked too smooth. The goal is to check whether the current pipeline can learn sharp PRISM structure at all before adding another model component.

## Pipeline Checks

| Check | Finding |
| --- | --- |
| Date matching | ERA5 daily fields and PRISM rasters are matched by exact normalized dates. History windows require consecutive matched dates. |
| Grid alignment | PRISM rasters are clipped/reprojected to a consistent template. ERA5 predictors stay on the coarse grid; model outputs are resized to the PRISM target size during decoding/evaluation. |
| Physical units | ERA5 `t2m` is converted from K to C when needed. PRISM is auto-scaled if stored as centi-degrees. Targets are trained/evaluated in Celsius, not normalized target space. |
| Input normalization | Predictor statistics are computed from training indices only. Validation inputs use the training split mean/std. |
| Static topo scaling | DEM-derived features are loaded as optional input channels and normalized with the rest of the inputs from the training split. No fake terrain channels are used. |
| Residual target | Training target is `PRISM - upsampled ERA5 t2m`; evaluation reconstructs final temperature as `upsampled ERA5 t2m + predicted residual`. |
| Output range | U-Net has no output activation or clipping. Predictions are not hard-limited before metrics. |
| Visualization scale | Prediction and target panels use shared temperature limits inside each figure. Cross-experiment visual comparisons are not forced to a single global color scale. |

No obvious date/unit/inverse-scaling bug was found in the inspected code. The weaker signal is in memorization behavior.

## Overfit Sanity Results

All checks used the medium dataset, U-Net, `core4_topo`, replicate padding, bilinear upsampling, and the same preprocessing path as the main terrain experiments.

### One Sample

| Target mode | Epochs | Train RMSE | MAE | Gradient ratio | High-frequency ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 600 | 0.4563 | 0.3214 | 0.4689 | 0.0098 |
| residual | 600 | 0.2249 | 0.1546 | 0.6650 | 0.2325 |

Residual training is clearly easier than direct prediction, but neither mode reaches near-zero error on one fixed sample.

### Small Subsets

| Train samples | Epochs | Train RMSE | Validation RMSE | Train gradient ratio | Train high-frequency ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 500 | 0.2958 | 2.4328 | 0.5850 | 0.0936 |
| 8 | 500 | 0.4906 | 2.1597 | 0.5550 | 0.0674 |

The model fits the tiny subsets better than the full validation distribution, but it still does not memorize them strongly. This is not the expected behavior for a healthy overfit test.

## Learning Rate / Width Check

The 8-sample residual overfit case was repeated with three learning rates and the current/2x U-Net width.

| Base channels | LR | Epochs | Train RMSE | Validation RMSE | Train HF ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 1e-3 | 300 | 0.5289 | 2.0561 | 0.0527 |
| 24 | 3e-4 | 300 | 0.5102 | 2.2982 | 0.0363 |
| 24 | 1e-4 | 300 | 0.6659 | 2.3938 | 0.0324 |
| 48 | 1e-3 | 300 | 0.3660 | 2.0311 | 0.0778 |
| 48 | 3e-4 | 300 | 0.4444 | 2.2916 | 0.0411 |
| 48 | 1e-4 | 300 | 0.5312 | 2.3315 | 0.0339 |

Doubling width helps training fit, especially at `1e-3`, but the tiny subset still does not collapse to low error. LR/capacity matter, but they do not fully explain the smooth predictions.

## Current Interpretation

- The pipeline does not show a simple unit-conversion, residual-reconstruction, or validation-normalization leak.
- The one-sample and small-subset checks fail the strict memorization expectation.
- Residual mode is easier to optimize than direct prediction.
- More width helps, but not enough to call this only a capacity issue.
- The next debug step should inspect target formulation, decoder/output resolution behavior, and whether the model/loss is suppressing local residual amplitudes before running larger experiments.
