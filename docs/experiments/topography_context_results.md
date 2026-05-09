# Topography Context Results

This is the first controlled static-context run after the U-Net, boundary, undertraining, and sharpness diagnostics.

The DEM source is a real [USGS 3DEP / The National Map](https://www.usgs.gov/tools/download-data-maps-national-map) Elevation ImageServer export over the ERA5/PRISM Georgia bbox (`-85,30,-80,35`) at `1200 x 1200` pixels. The source raster was aligned to the PRISM grid clipped to the ERA5 domain and saved locally as:

```text
data_processed/static/georgia_prism_topography.nc
```

The processed static channels are:

- elevation;
- slope;
- aspect;
- terrain gradient magnitude.

For this first run, these PRISM-aligned covariates are resampled onto the ERA5 input grid inside the dataset loader. That keeps the U-Net architecture and input convention unchanged. It also means this is not yet a full-resolution terrain-conditioning design.

The data files are intentionally ignored by git.

## Setup

- dataset: medium
- model: U-Net only
- history: 3
- target mode: direct
- padding: replicate
- upsampling: bilinear
- seed/split: 42
- training budget: 80 epochs with the same optimizer/loss settings as the controlled spatial runs

Command:

```bash
.venv/bin/python scripts/run_topography_experiment.py \
  --dataset-version medium \
  --static-covariate-path data_processed/static/georgia_prism_topography.nc \
  --output-dir results/topography_context \
  --epochs 80 \
  --seed 42 \
  --split-seed 42 \
  --device cpu \
  --overwrite
```

## Metrics

| Model | Inputs | RMSE | MAE | Bias | Corr. | Border RMSE | Center RMSE | Border/Center | Var. Ratio | Grad. Ratio | HF Ratio | Contrast Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | core4 | 2.8467 | 1.9428 | 0.8326 | 0.8788 | 3.5826 | 2.5596 | 1.3996 | 0.9369 | 0.6475 | 0.0303 | 0.7423 |
| U-Net | core4 | 1.7995 | 1.4031 | -0.1161 | 0.9475 | 2.0285 | 1.7178 | 1.1809 | 0.9336 | 0.2975 | 0.0002 | 0.3430 |
| U-Net | core4 + elevation | 1.5146 | 1.1824 | -0.2530 | 0.9639 | 1.7517 | 1.4283 | 1.2264 | 0.9179 | 0.3504 | 0.0005 | 0.4040 |
| U-Net | core4 + topo | 1.5886 | 1.2438 | 0.1557 | 0.9622 | 1.7235 | 1.5417 | 1.1179 | 1.0697 | 0.3792 | 0.0005 | 0.4372 |

## Read

Elevation helped on this split. `core4 + elevation` gives the lowest RMSE and MAE, with a large drop relative to the same U-Net without static context.

The full topo feature set does not beat elevation-only on RMSE, but it gives the lowest border RMSE and lowest border/center ratio among the learned variants. It also has the highest gradient ratio and local contrast ratio. That is consistent with terrain context helping spatial structure, but it is still one seed.

Oversmoothing is reduced but not solved. Gradient and local-contrast ratios improve with static context, yet high-frequency detail remains near zero in this diagnostic. The model still behaves like a smooth reconstruction model, just with better physically anchored spatial information.

Terrain-sensitive regions look more plausible in the saved panels, but this has not yet been quantified by terrain class or elevation band. That should be a follow-up diagnostic before making a stronger terrain-specific claim.

## Conclusion

This run supports Professor Hu's suggestion that topography is a useful next input. It improves RMSE and several spatial diagnostics under a controlled setup, without adding temporal complexity or changing the U-Net architecture.

It is not enough to claim the issue is solved. The seed-stability check has now been run in [`topography_seed_stability.md`](topography_seed_stability.md); the next diagnostic should inspect error by elevation, slope, and terrain-gradient bins.
