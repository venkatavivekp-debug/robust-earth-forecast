# Topography Residual Stability

Hypothesis: terrain-conditioned residual reconstruction should be easier than direct PRISM reconstruction. The model sees ERA5 `core4` plus DEM-derived terrain features, then predicts the local correction from upsampled ERA5 temperature to PRISM.

This follows from Professor Hu's comments in two ways. First, the remaining failure is spatial: blurred fields, weak high-frequency detail, and boundary degradation. Second, topography is a physically meaningful missing input for PRISM-scale temperature. Hu et al. (2023) is not this task, but it supports the same discipline: use U-Net-style weather postprocessing under controlled verification rather than adding architecture complexity without diagnosis.

Outputs are local and ignored by git:

```text
results/topography_residual_stability/
```

## Setup

- dataset: medium
- model: U-Net
- input: `core4_topo`
- history: 3
- target modes: direct vs residual
- seeds: 42, 7, 123
- padding: replicate
- upsampling: bilinear
- epochs: 80
- static features: elevation, slope, aspect, terrain-gradient magnitude

## Across seeds

Values are mean +/- sample standard deviation across seeds.

| Target mode | RMSE | MAE | Bias | Corr. | Border RMSE | Center RMSE | Border/Center | Var. ratio | Grad. ratio | HF ratio | Contrast ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 1.4481 +/- 0.1428 | 1.1190 +/- 0.1250 | 0.0612 +/- 0.1389 | 0.9634 +/- 0.0062 | 1.6630 +/- 0.1168 | 1.3694 +/- 0.1620 | 1.2203 +/- 0.0906 | 0.9991 +/- 0.0629 | 0.3509 +/- 0.0247 | 0.0005 +/- 0.0000 | 0.4097 +/- 0.0238 |
| residual | 1.3858 +/- 0.0564 | 1.0697 +/- 0.0609 | -0.0933 +/- 0.0481 | 0.9661 +/- 0.0063 | 1.5828 +/- 0.0452 | 1.3146 +/- 0.0622 | 1.2048 +/- 0.0294 | 0.9838 +/- 0.0670 | 0.5665 +/- 0.0194 | 0.0302 +/- 0.0023 | 0.6572 +/- 0.0172 |

## Per-seed RMSE

| Seed | Direct RMSE | Residual RMSE | Residual - direct |
| ---: | ---: | ---: | ---: |
| 42 | 1.5886 | 1.3791 | -0.2094 |
| 7 | 1.3031 | 1.3330 | 0.0299 |
| 123 | 1.4525 | 1.4451 | -0.0074 |

Residual is better on mean RMSE and on two of three seeds, but seed 7 is a small regression. The evidence supports residual learning as useful, not uniformly dominant.

## Spatial diagnostics

Residual target mode improves the spatial ratios more clearly than it improves RMSE:

- gradient ratio: `0.3509 -> 0.5665`
- high-frequency ratio: `0.0005 -> 0.0302`
- local contrast ratio: `0.4097 -> 0.6572`

That is consistent with residual prediction recovering more local structure. It still does not recover full PRISM detail. A high-frequency ratio around `0.03` means most fine-scale target energy is still missing.

Border degradation remains. Mean border RMSE drops from `1.6630` to `1.5828`, and mean border/center ratio drops slightly from `1.2203` to `1.2048`, but every run remains above 1.0. Residual learning helps absolute error more than it fixes boundary behavior.

## Read

Terrain-conditioned residual reconstruction is the best current direction inside this project. It is better aligned with the physical problem than asking the U-Net to regenerate the full PRISM field from scratch.

The next limitation is not ConvLSTM. The residual model still needs terrain-bin diagnostics, more data coverage, and possibly loss/verification changes aimed at gradients or extremes. Temporal modeling should stay postponed until the residual spatial baseline is understood.
