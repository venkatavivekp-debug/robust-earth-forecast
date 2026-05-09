# Topography and static context plan

Professor Hu suggested adding topography after the boundary and blur diagnostics. That is the next physically grounded direction: PRISM temperature fields include terrain-aware local structure, while the current ERA5 predictors are coarse atmospheric fields over a clipped Georgia domain.

## Why terrain matters

Temperature varies with elevation, slope exposure, cold-air drainage, ridges, valleys, and land-surface gradients. Upsampled ERA5 can carry the broad synoptic pattern, but it cannot directly identify a mountain valley or a local ridge at PRISM scale. A U-Net can learn some spatial correction from examples, but without static terrain context it has to infer persistent fine-scale structure indirectly from a small sample.

This matches the current evidence:

- U-Net improves over the no-skip `PlainEncoderDecoder`, so multi-scale spatial reconstruction matters.
- Border RMSE remains above center RMSE, including for persistence, so clipped-domain context is part of the issue.
- Padding and upsampling choices change the error profile but do not remove blur.
- Longer training gives only mild improvement, so undertraining alone is not the dominant explanation.
- Spatial sharpness diagnostics show that U-Net keeps broad variance close to PRISM but loses much of the target gradient, local contrast, and high-frequency detail.

## Current data state

No real DEM/topography file is currently present in the repository or local ignored data folders. Do not create synthetic terrain or placeholder channels. The next step is to place a real DEM raster locally, document the source, and run the preparation script before any dataset or model change.

Expected local source path:

```text
data_raw/static/source_dem/<source_dem_file>.tif
```

Expected processed output path:

```text
data_raw/static/topography/georgia_prism_topography.nc
data_raw/static/topography/georgia_prism_topography.metadata.json
```

These paths are under `data_raw/` and should remain ignored by git.

## Why before temporal modeling

ConvLSTM or longer temporal context can help only after the spatial reconstruction baseline is interpretable. If the model lacks elevation or terrain-gradient information, temporal recurrence may hide the missing spatial variable rather than explain it. The next comparison should therefore add static context to the same U-Net setup before revisiting temporal models.

This is also closer to the GAIM direction: geospatial AI with Earth data, remote sensing/GIS context, and uncertainty-aware environmental modeling. The Hu et al. Monthly Weather Review paper is not a temperature-downscaling paper, but it is relevant because it uses a U-Net-style architecture for weather postprocessing and evaluates more than one scalar score.

## Candidate data

Do not add topography until a real source is selected and documented. Reasonable candidates:

- USGS 3DEP / National Elevation Dataset DEM;
- other public DEM products with reproducible download and citation;
- derived static fields such as elevation, slope, aspect, terrain gradient, and possibly distance-to-coast or land mask if justified.

All static fields must be reprojected/aligned to the PRISM target grid and normalized using training-split statistics only.

## Preparation path

The preparation script is intentionally source-file driven:

```bash
python3 scripts/prepare_topography_context.py \
  --dem-path data_raw/static/source_dem/<source_dem_file>.tif \
  --dataset-version medium \
  --source-name "USGS 3DEP/NED or selected DEM source"
```

It should:

- load a real DEM raster with CRS metadata;
- use an existing PRISM raster as the target grid;
- reproject/resample the DEM to the PRISM grid;
- write elevation, slope, aspect, and terrain-gradient channels;
- write metadata with source name, grid shape, CRS, resolution, and feature statistics.

The processed channels are raw static fields. Model training should still learn normalization from the training split only.

## Fair comparison

The first topography experiment should change only the input channels:

- same dataset version;
- same train/validation split;
- same seed;
- same U-Net architecture;
- same direct target mode;
- same normalization protocol;
- same metrics and diagnostics;
- same border/center and error-map checks;
- same training budget, preferably the 300-epoch undertraining reference unless runtime becomes limiting.

Compare:

1. persistence / upsampled ERA5;
2. U-Net `core4_h3` without static covariates;
3. U-Net `core4_h3 + topography` with the same training settings.

## Success criteria

Success is not just a lower scalar RMSE. Evidence should include:

- lower RMSE and MAE on validation;
- reduced bias;
- better prediction variance relative to PRISM;
- sharper local structure in prediction panels;
- lower border RMSE or a lower border/center ratio;
- better terrain-gradient preservation;
- stable behavior across seeds.

## Failure criteria

Topography should be considered unhelpful or inconclusive if:

- training loss improves but validation metrics do not;
- outputs gain artifacts or unrealistic local texture;
- border RMSE worsens;
- seed behavior becomes less stable;
- improvement appears only on one split and not in maps.

## Next implementation step

Add a small, reproducible static-covariate data path for DEM-derived channels, then run one controlled U-Net comparison against the current no-topography reference. Do not add residual mode, ConvLSTM, attention, or new loss functions in the same experiment.
