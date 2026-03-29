# Robust Earth Forecast

A compact, professor-facing mini research prototype for **Georgia surface temperature downscaling**:

- **Input**: ERA5 coarse-resolution temperature (NetCDF)
- **Target**: PRISM higher-resolution temperature rasters
- **Model**: CNN baseline for spatial downscaling
- **Showcase**: notebook-first workflow with reproducible training/evaluation scripts

## Research Motivation

Regional climate analysis often needs finer spatial detail than global reanalysis products provide. This project studies a practical baseline mapping:

**ERA5 (coarse) -> CNN downscaler -> PRISM-like high-resolution temperature**

The objective is to establish a clean first research pipeline before moving to more advanced spatiotemporal/transformer approaches.

## Current Scope

This repository currently focuses on:

- Georgia-only prototype
- Daily temperature downscaling baseline
- ERA5-to-PRISM data alignment
- Compact CNN training/evaluation
- Notebook-based result presentation

This is intentionally a baseline, not a production climate system.

## Repository Structure

```text
robust-earth-forecast/
├── data_pipeline/
│   ├── download_era5_georgia.py
│   ├── download_prism.py
│   └── validate_prism.py
├── datasets/
│   └── prism_dataset.py
├── models/
│   └── cnn_downscaler.py
├── training/
│   └── train_downscaler.py
├── evaluation/
│   └── evaluate_model.py
├── notebooks/
│   └── climate_forecasting_demo.ipynb
├── data_raw/                # local data only (not committed)
├── checkpoints/             # local model checkpoints (ignored)
└── results/                 # evaluation outputs (ignored)
```

## Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Automatic PRISM + ERA5 Setup

You can bootstrap both datasets from scripts:

```bash
python3 data_pipeline/download_era5_georgia.py --year 2023 --month 1
python3 data_pipeline/download_prism.py
```

Default PRISM behavior:

- downloads 3 daily `tmean` files starting at `20230101`
- validates each download (size + real ZIP)
- auto-extracts archives to `data_raw/prism/`
- removes ZIP files after extraction

Expected local layout:

```text
data_raw/
├── era5_georgia_temp.nc
└── prism/
    ├── PRISM_tmean_stable_4kmD1_20230101_bil.bil
    ├── PRISM_tmean_stable_4kmD1_20230101_bil.hdr
    ├── PRISM_tmean_stable_4kmD1_20230102_bil.bil
    └── ...
```

Expected outputs after training/evaluation:

- `results/evaluation/metrics.json` (RMSE, MAE)
- `results/evaluation/comparison_*.png` (ERA5 input vs prediction vs PRISM target)

## Data Placement and Validation

Do not commit large raw datasets. Keep local files under `data_raw/`.

Expected baseline layout:

```text
data_raw/
├── era5_georgia_temp.nc
└── prism/
    ├── PRISM_tmean_stable_4kmD1_20230101_bil.bil
    ├── PRISM_tmean_stable_4kmD1_20230102_bil.bil
    └── ...
```

Supported PRISM raster types for this baseline: `.bil`, `.tif`, `.tiff`, `.asc`.
PRISM filenames must include `YYYYMMDD` (for ERA5/PRISM date pairing).

Validate PRISM files:

```bash
python3 data_pipeline/validate_prism.py --path data_raw/prism
```

If you have zipped PRISM downloads:

```bash
python3 data_pipeline/validate_prism.py \
  --path data_raw/prism/PRISM_tmean_stable_4kmD1_20230101_bil.zip \
  --extract-dir data_raw/prism
```

Optional conversion from BIL to GeoTIFF:

```bash
python3 data_pipeline/validate_prism.py \
  --path data_raw/prism \
  --convert-bil-to-geotiff \
  --geotiff-dir data_raw/prism/geotiff
```

## Quick Demo (Without Full Data)

If ERA5 and PRISM files are not available yet, the training/evaluation scripts still run and fail with clear, actionable error messages.

To run the full pipeline:

1. Download ERA5 data and place it under `data_raw/` (for example `data_raw/era5_georgia_temp.nc`).
2. Download/extract PRISM rasters and place them under `data_raw/prism/`.
3. Run training and evaluation commands from this README.

Expected outputs after a full run:

- RMSE and MAE metrics (saved in `results/evaluation/metrics.json`)
- Prediction vs ground-truth comparison plots (saved in `results/evaluation/`)

## Training (from Project Root)

```bash
python3 training/train_downscaler.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --device auto \
  --checkpoint-out checkpoints/cnn_downscaler_best.pt
```

## Evaluation (from Project Root)

```bash
python3 evaluation/evaluate_model.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --checkpoint-path checkpoints/cnn_downscaler_best.pt \
  --num-samples 8 \
  --num-plots 1 \
  --results-dir results/evaluation
```

Outputs:

- `results/evaluation/metrics.json` (RMSE, MAE)
- `results/evaluation/comparison_*.png` (ERA5 input, prediction, PRISM target)

## Notebook Showcase

Open the main presentation notebook:

```bash
jupyter notebook notebooks/climate_forecasting_demo.ipynb
```

The notebook demonstrates:

- motivation and problem framing
- ERA5/PRISM loading and shape checks
- trained model loading
- prediction and visual comparison
- brief conclusion and future research direction

## Current Results Summary

The baseline provides:

- reproducible ERA5-PRISM alignment
- a working CNN downscaler training loop
- standardized evaluation with RMSE/MAE and comparison plots

Exact metric values depend on local PRISM coverage, date range, and training settings.

## Next Research Step (Prithvi WxC Direction)

This baseline is the foundation for moving toward Prithvi WxC-style research:

- longer temporal context and spatiotemporal modeling
- multi-variable atmospheric conditioning
- transformer-based geospatial token modeling
- Lightning-based experiment management for larger runs

## Author

Venkata Vivek Panguluri
