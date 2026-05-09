# Detail-Preserving Loss Reasoning

The next question is not which architecture to add. The current U-Net with topography already improves RMSE, and residual target mode improves gradient/detail diagnostics. The remaining issue is that predictions still look too smooth.

## Why RMSE can favor smooth predictions

RMSE penalizes large pixel-wise errors, so an uncertain model can reduce loss by predicting a spatial average. That is useful for broad temperature fields but weak for PRISM-scale structure. Local ridges, valleys, and sharp gradients occupy fewer pixels than the broad synoptic pattern, so RMSE can improve while detail disappears.

This matches the diagnostics: direct topography improves RMSE, while high-frequency ratio stays near zero. Residual topography improves detail ratios, but still leaves most fine-scale PRISM energy missing.

## Why U-Net plus topography still loses detail

Skip connections preserve multi-scale features, and terrain channels add physically meaningful static context. But the training objective still mostly rewards point-wise temperature accuracy. With limited samples, the model can use terrain to place a better smooth correction without learning sharper local gradients.

That does not mean U-Net or topography failed. It means the objective may not match the failure mode we now care about.

## Why gradient-aware loss is a grounded next step

A simple gradient consistency term directly asks the model to match spatial changes in the target, not just absolute values. For ERA5 -> PRISM, that is physically relevant because the missing signal is often local structure: slope-related transitions, terrain boundaries, and neighborhood-scale contrast.

This follows Professor Hu's feedback about blurred outputs, boundary artifacts, and thinking through downscaling structure before adding temporal complexity. It also fits the Hu et al. MWR weather-postprocessing framing: U-Net-style models should be evaluated and tuned with the structure of the meteorological field in mind, not just a single scalar score.

## Why temporal modeling is still postponed

ConvLSTM would test whether time history helps. The current weakness is spatial detail in a terrain-conditioned reconstruction. Adding temporal recurrence now would mix two questions:

1. does residual/topographic spatial reconstruction work?
2. does temporal context add useful information?

Those should stay separate.

## Controlled hypothesis

For terrain-conditioned residual U-Net, adding a simple gradient loss should improve gradient, high-frequency, or local-contrast diagnostics without badly degrading RMSE, MAE, bias, border RMSE, or visual realism.

## Evidence that supports the phase

Supportive evidence would be:

- comparable or lower RMSE/MAE than the residual MSE baseline;
- higher gradient ratio and local contrast ratio;
- higher high-frequency ratio without noisy artifacts;
- border RMSE that does not worsen substantially;
- prediction/error/detail panels that look less smooth.

## Evidence that rejects it

The idea should be rejected or treated as inconclusive if:

- sharpness ratios improve only by adding noisy artifacts;
- RMSE or MAE worsens enough to offset spatial-detail gains;
- border degradation worsens;
- improvements appear on one seed only;
- visual panels remain smooth despite added gradient loss.

If this fails, the next step should be terrain-bin diagnostics, more data coverage, or target/feature alignment checks, not temporal modeling.
