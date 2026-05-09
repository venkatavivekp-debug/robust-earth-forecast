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

The first reproducible DEM workflow uses the USGS 3DEP Elevation ImageServer. The local source raster is downloaded over the ERA5/PRISM Georgia bbox (`-85,30,-80,35`) as a real DEM export, then aligned to the PRISM grid clipped to ERA5 bounds. The raw and processed files remain ignored by git.

Local source path:

```text
data_raw/static/source_dem/usgs_3dep_georgia_*.tif
```

Processed output path:

```text
data_processed/static/georgia_prism_topography.nc
data_processed/static/georgia_prism_topography.metadata.json
```

The prepared grid is `121 x 121`, matching the current clipped PRISM target grid. The source export is `1200 x 1200` over the Georgia bbox and is resampled to PRISM support during preparation.

For model input, the dataset loader currently resamples these static fields onto the ERA5 grid and appends them as extra channels. This keeps the architecture unchanged for the first controlled test. A later full-resolution terrain-conditioning design would be a separate architecture/input-convention change.

## Why before temporal modeling

ConvLSTM or longer temporal context can help only after the spatial reconstruction baseline is interpretable. If the model lacks elevation or terrain-gradient information, temporal recurrence may hide the missing spatial variable rather than explain it. The next comparison should therefore add static context to the same U-Net setup before revisiting temporal models.

This is also closer to the GAIM direction: geospatial AI with Earth data, remote sensing/GIS context, and uncertainty-aware environmental modeling. The Hu et al. Monthly Weather Review paper is not a temperature-downscaling paper, but it is relevant because it uses a U-Net-style architecture for weather postprocessing and evaluates more than one scalar score.

## Candidate data

The current source is:

- [USGS 3DEP / The National Map](https://www.usgs.gov/tools/download-data-maps-national-map) Elevation ImageServer export;
- EPSG:4326 bbox request over the ERA5/PRISM Georgia domain;
- GeoTIFF output, then PRISM-grid NetCDF static covariates.

Other reasonable candidates for later comparison:

- USGS 3DEP / National Elevation Dataset DEM;
- other public DEM products with reproducible download and citation;
- derived static fields such as elevation, slope, aspect, terrain gradient, and possibly distance-to-coast or land mask if justified.

All static fields must be reprojected/aligned to the PRISM target grid and normalized using training-split statistics only.

## Preparation path

The preparation script is intentionally source-file driven:

```bash
python3 scripts/download_dem_data.py \
  --output-dir data_raw/static/source_dem \
  --bbox=-85,30,-80,35 \
  --source usgs_3dep_image_service \
  --size 1200,1200

python3 scripts/prepare_topography_context.py \
  --dem-path data_raw/static/source_dem/<downloaded_dem>.tif \
  --dataset-version medium \
  --output data_processed/static/georgia_prism_topography.nc \
  --source-name "USGS 3DEP Elevation ImageServer"
```

It should:

- load a real DEM raster with CRS metadata;
- use an existing PRISM raster as the target grid;
- clip the PRISM reference grid to ERA5 bounds;
- reproject/resample the DEM to that clipped PRISM grid;
- write elevation, slope, aspect, and terrain-gradient-magnitude channels;
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

## Current next step

The direct topography comparison has now been repeated across seeds 42, 7, and 123. Terrain context improves RMSE and spatial ratios on average, but high-frequency detail remains weak and border degradation remains.

Residual topography has also been checked across the same seeds. It improves mean RMSE and sharpness diagnostics, but not enough to resolve fine-scale detail loss. The next step is error-by-elevation/slope diagnostics. Do not add ConvLSTM, attention, or new loss functions in the same experiment.
