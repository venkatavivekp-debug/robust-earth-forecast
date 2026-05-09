# Detail-Preserving Loss Results

Hypothesis: adding a simple spatial-gradient consistency term to the terrain-conditioned residual U-Net should improve local structure without badly hurting RMSE or border behavior.

This was tested now because the earlier diagnostics converged on the same failure mode: U-Net, topography, and residual prediction help, but prediction maps still lose PRISM-scale detail. Professor Hu's comments pointed to blur, boundary behavior, and multi-scale reconstruction before temporal complexity. The Hu et al. MWR U-Net/weather paper is relevant in that same limited sense: evaluate structured weather fields with field-aware diagnostics, not only a scalar score.

Outputs are local and ignored by git:

```text
results/detail_preserving_loss/
```

## Setup

- dataset: medium
- model: U-Net
- input: `core4_topo`
- target mode: residual
- seeds: 42, 7, 123
- padding: replicate
- upsampling: bilinear
- epochs: 80
- static features: elevation, slope, aspect, terrain-gradient magnitude

Loss modes:

- `mse`: MSE on the residual target
- `mse_l1`: MSE + L1 on the residual target, matching the previous default training behavior
- `mse_grad`: MSE plus spatial-gradient consistency on the reconstructed PRISM prediction
- `mse_l1_grad`: MSE + L1 plus spatial-gradient consistency

## Aggregate results

Values are mean +/- sample standard deviation across seeds.

| Loss | RMSE | MAE | Bias | Corr. | Border RMSE | Center RMSE | Border/Center | Grad. ratio | HF ratio | Contrast ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mse` | 1.4047 +/- 0.0723 | 1.0899 +/- 0.0807 | -0.0919 +/- 0.1049 | 0.9656 +/- 0.0065 | 1.6051 +/- 0.0644 | 1.3323 +/- 0.0775 | 1.2058 +/- 0.0327 | 0.5656 +/- 0.0235 | 0.0303 +/- 0.0023 | 0.6563 +/- 0.0217 |
| `mse_l1` | 1.3858 +/- 0.0564 | 1.0697 +/- 0.0609 | -0.0933 +/- 0.0481 | 0.9661 +/- 0.0063 | 1.5828 +/- 0.0452 | 1.3146 +/- 0.0622 | 1.2048 +/- 0.0294 | 0.5665 +/- 0.0194 | 0.0302 +/- 0.0023 | 0.6572 +/- 0.0172 |
| `mse_grad` | 1.3975 +/- 0.0540 | 1.0824 +/- 0.0593 | -0.0365 +/- 0.0845 | 0.9651 +/- 0.0060 | 1.5884 +/- 0.0513 | 1.3289 +/- 0.0566 | 1.1957 +/- 0.0223 | 0.5639 +/- 0.0235 | 0.0309 +/- 0.0020 | 0.6541 +/- 0.0227 |
| `mse_l1_grad` | 1.4027 +/- 0.0638 | 1.0817 +/- 0.0723 | -0.1253 +/- 0.0705 | 0.9652 +/- 0.0057 | 1.5966 +/- 0.0419 | 1.3329 +/- 0.0734 | 1.1992 +/- 0.0378 | 0.5687 +/- 0.0249 | 0.0309 +/- 0.0017 | 0.6600 +/- 0.0254 |

## Read

The result does not justify changing the default loss. The existing `mse_l1` objective still has the best mean RMSE, MAE, correlation, border RMSE, and center RMSE.

The gradient terms do what they were supposed to do, but weakly. `mse_l1_grad` gives the highest mean gradient ratio, high-frequency ratio, and local contrast ratio. The gain is small, and it comes with worse RMSE/MAE than `mse_l1`.

Border behavior remains unresolved. `mse_grad` gives the lowest mean border/center ratio, but absolute border RMSE is still higher than center RMSE for every loss mode. None of these objectives removes boundary degradation.

The visual/detail maps should be read cautiously. The gradient-aware losses can recover slightly more detail signal, but the high-frequency ratio remains around `0.03`, so outputs are still much smoother than PRISM.

## Conclusion

Gradient-aware loss is a useful diagnostic, not a clear next default. It gives a small sharpness signal but does not deliver a strong combined win across RMSE, border behavior, and detail.

The next grounded step should be terrain-bin error analysis and data-scale/coverage checks for the residual-topography setup. Temporal modeling remains postponed.
