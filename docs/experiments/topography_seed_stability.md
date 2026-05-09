# Topography Seed Stability

This repeats the controlled topography-context comparison across the same seed set used in the spatial benchmark: **42**, **7**, and **123**.

Setup:

- dataset: medium
- model: U-Net
- history: 3
- target mode: direct
- padding: replicate
- upsampling: bilinear
- epochs: 80
- static data: USGS 3DEP DEM-derived elevation, slope, aspect, terrain-gradient magnitude

Outputs are local and ignored by git:

```text
results/topography_seed_stability/
```

## Direct target mode

Values are mean +/- sample standard deviation across the three seeds.

| Model | Inputs | RMSE | MAE | Border RMSE | Center RMSE | Border/Center | Gradient ratio | HF ratio | Contrast ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | core4 | 2.7171 +/- 0.2484 | 1.9295 +/- 0.0740 | 3.3803 +/- 0.3489 | 2.4608 +/- 0.2075 | 1.3722 +/- 0.0302 | 0.6154 +/- 0.0291 | 0.0307 +/- 0.0029 | 0.7143 +/- 0.0279 |
| U-Net | core4 | 1.6629 +/- 0.2085 | 1.2906 +/- 0.1803 | 1.8592 +/- 0.2161 | 1.5932 +/- 0.2074 | 1.1688 +/- 0.0325 | 0.2857 +/- 0.0113 | 0.0002 +/- 0.0000 | 0.3338 +/- 0.0088 |
| U-Net | core4 + elevation | 1.4649 +/- 0.0847 | 1.1156 +/- 0.0740 | 1.6809 +/- 0.0855 | 1.3867 +/- 0.0865 | 1.2130 +/- 0.0283 | 0.3300 +/- 0.0177 | 0.0004 +/- 0.0001 | 0.3855 +/- 0.0160 |
| U-Net | core4 + topo | 1.4481 +/- 0.1428 | 1.1190 +/- 0.1250 | 1.6630 +/- 0.1168 | 1.3694 +/- 0.1620 | 1.2203 +/- 0.0906 | 0.3509 +/- 0.0247 | 0.0005 +/- 0.0000 | 0.4097 +/- 0.0238 |

## Per-seed read

| Seed | Best RMSE among U-Net variants | RMSE |
| ---: | --- | ---: |
| 42 | core4 + elevation | 1.5146 |
| 7 | core4 + topo | 1.3031 |
| 123 | core4 + topo | 1.4525 |

Both static-context variants beat the no-topography U-Net on all three seeds. Elevation-only is best for seed 42; the full topography set is best for seeds 7 and 123 and has the lowest mean RMSE.

The border result is more mixed. Topography lowers absolute border RMSE relative to the no-topography U-Net, but the border/center ratio remains above 1.0 for every model and seed. Static context helps the error level; it does not remove boundary degradation.

The sharpness metrics remain the limiting point. Gradient and local contrast ratios improve with terrain channels, especially the full topo set, but high-frequency ratio stays near zero for direct U-Net predictions. This is still a smooth reconstruction model.

## Residual check

The existing U-Net code already supports residual target mode, so one seed-42 topo run was checked without changing architecture.

| Mode | Inputs | RMSE | MAE | Border RMSE | Center RMSE | Border/Center | Gradient ratio | HF ratio | Contrast ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | core4 + topo | 1.5886 | 1.2438 | 1.7235 | 1.5417 | 1.1179 | 0.3792 | 0.0005 | 0.4372 |
| residual | core4 + topo | 1.3791 | 1.0754 | 1.5617 | 1.3138 | 1.1887 | 0.5765 | 0.0289 | 0.6600 |

Residual topography improves seed-42 RMSE, MAE, correlation, gradient ratio, high-frequency ratio, and local contrast ratio. The border/center ratio increases, so it should not be described as solving the boundary issue. This is a useful next controlled experiment, not a settled result.

## Interpretation

The terrain result is stable enough to keep static spatial context as the next active direction. The remaining blur is not well explained by undertraining alone: longer training gave only mild improvement, while terrain channels and residual formulation change the spatial diagnostics more directly.

The next clean step is a multi-seed residual-topography comparison with the same diagnostics, followed by terrain-bin error analysis. Temporal modeling should still wait.
