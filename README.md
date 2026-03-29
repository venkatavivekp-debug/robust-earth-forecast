# Robust Earth Forecast

Master's-level research baseline for geospatial climate downscaling over Georgia.

## Project Overview

This project demonstrates a practical downscaling task:
**map coarse ERA5 near-surface temperature to higher-resolution PRISM temperature**.

Why this matters:
- climate analyses often need finer spatial detail than global reanalysis grids
- downscaling is a core step for regional climate risk and environmental modeling

What is implemented now:
- robust ERA5 + PRISM data ingestion
- date-aligned ERA5->PRISM supervised dataset
- compact CNN downscaler baseline
- reproducible train/evaluate pipeline with metrics + visualization

## Pipeline Summary

```text
ERA5 (coarse daily temperature, NetCDF)
        ->
PRISM (higher-resolution daily temperature rasters)
        ->
ERA5/PRISM date + spatial alignment
        ->
CNN Downscaler
        ->
Predicted high-resolution temperature
        ->
Evaluation: RMSE, MAE, comparison plot
```

## Run Order

Run from project root in this exact order:

1. Create virtual environment
2. Install requirements
3. Download ERA5
4. Download PRISM
5. Train model
6. Evaluate model
7. Open notebook

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_pipeline/download_era5_georgia.py
python data_pipeline/download_prism.py
python training/train_downscaler.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --device auto \
  --checkpoint-out checkpoints/cnn_downscaler_best.pt
python evaluation/evaluate_model.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --checkpoint-path checkpoints/cnn_downscaler_best.pt \
  --num-samples 8 \
  --num-plots 1 \
  --results-dir results/evaluation
jupyter notebook notebooks/climate_forecasting_demo.ipynb
```

## Example Output

Example evaluation figure:

![ERA5 PRISM comparison](results/evaluation/comparison_1_20230101.png)

Interpretation:
- left: ERA5 input (upsampled from coarse grid)
- middle: CNN prediction at PRISM-like resolution
- right: PRISM ground-truth target

Example metrics (from `results/evaluation/metrics.json` in this repo):
- RMSE: `14.546989758809408`
- MAE: `14.394099553426107`
- Evaluated samples: `3`

## Notebook Demo

Open:

```bash
jupyter notebook notebooks/climate_forecasting_demo.ipynb
```

The notebook is the walkthrough of:
- ERA5 vs PRISM data framing
- baseline CNN model setup
- inference and visualization
- saved evaluation artifact display

## Data Notes

Expected local inputs:
- ERA5 NetCDF: `data_raw/era5_georgia_temp.nc`
- PRISM rasters directory: `data_raw/prism/`

PRISM downloader defaults:
- variable: `tmean`
- dates: 3 days starting `20230101`
- destination: `data_raw/prism/`

## Research Direction

This repository is intentionally a baseline foundation.
Next step is to move from static spatial CNN mapping to richer **spatiotemporal models** (for example ConvLSTM and Transformer approaches), with future work aligned to **Prithvi WxC-style climate modeling directions**.

## Common Errors

- Missing ERA5 file
  - Error: `ERA5 file not found: data_raw/era5_georgia_temp.nc`
  - Fix: `python data_pipeline/download_era5_georgia.py`

- Missing PRISM rasters
  - Error: `No PRISM raster files found in data_raw/prism`
  - Fix: `python data_pipeline/download_prism.py`

- Missing checkpoint
  - Error: `Checkpoint not found: checkpoints/cnn_downscaler_best.pt`
  - Fix: run training, then rerun evaluation

## Repository Structure

```text
robust-earth-forecast/
├── data_pipeline/
├── datasets/
├── models/
├── training/
├── evaluation/
├── notebooks/
├── results/
│   └── evaluation/
│       ├── comparison_1_20230101.png
│       └── metrics.json
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

Venkata Vivek Panguluri
