# Robust Earth Forecast

Baseline research prototype for Georgia temperature downscaling:

- Input: ERA5 coarse 2m temperature (`.nc`)
- Target: PRISM higher-resolution daily rasters (`.bil/.tif/.tiff/.asc`)
- Model: compact CNN downscaler

## Overview

This project builds a clean first baseline for regional climate downscaling:
ERA5 coarse fields are mapped to PRISM higher-resolution temperature over Georgia.
The goal is a reproducible research workflow that is easy to present and extend.

## Research Motivation

Global reanalysis products are useful but often too coarse for local analysis.
Downscaling ERA5 to PRISM resolution provides a practical starting point for
state-level climate modeling and model benchmarking.

## Run Order

1. Create venv
2. Install requirements
3. Download ERA5
4. Download PRISM
5. Run training
6. Run evaluation
7. Open notebook

Exact command sequence (from project root):

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

## Fresh Clone Quickstart

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the Run Order commands above.

## Automatic PRISM + ERA5 Setup

### ERA5 downloader

```bash
python data_pipeline/download_era5_georgia.py
```

Behavior:

- defaults to Georgia bounds, year `2023`, month `01`
- output path: `data_raw/era5_georgia_temp.nc`
- if file already exists, it skips download unless `--overwrite` is passed
- requires CDS API credentials in `~/.cdsapirc`

### PRISM downloader

```bash
python data_pipeline/download_prism.py
```

Behavior:

- defaults: `tmean`, 3 days starting `20230101`
- source endpoint: NACSE PRISM public service (`/prism/data/get/us/4km/...`)
- saves to: `data_raw/prism/`
- validates each archive:
  - file size > 1MB
  - valid ZIP signature (rejects HTML/error responses)
- auto-extracts and removes ZIPs
- validates extracted rasters and date pattern (`YYYYMMDD`)

Expected local data layout:

```text
data_raw/
├── era5_georgia_temp.nc
└── prism/
    ├── PRISM_tmean_stable_4kmD1_20230101_bil.bil
    ├── PRISM_tmean_stable_4kmD1_20230101_bil.hdr
    ├── PRISM_tmean_stable_4kmD1_20230102_bil.bil
    └── ...
```

## Training

```bash
python training/train_downscaler.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --device auto \
  --checkpoint-out checkpoints/cnn_downscaler_best.pt
```

Output checkpoint:

- `checkpoints/cnn_downscaler_best.pt`

## Evaluation

```bash
python evaluation/evaluate_model.py \
  --era5-path data_raw/era5_georgia_temp.nc \
  --prism-path data_raw/prism \
  --checkpoint-path checkpoints/cnn_downscaler_best.pt \
  --num-samples 8 \
  --num-plots 1 \
  --results-dir results/evaluation
```

Expected outputs:

- `results/evaluation/metrics.json` (RMSE, MAE)
- `results/evaluation/comparison_*.png` (ERA5 input vs prediction vs PRISM target)

## Notebook Demo

```bash
jupyter notebook notebooks/climate_forecasting_demo.ipynb
```

Notebook includes:

- ERA5/PRISM loading
- model inference and visualization
- metric summary
- future direction note toward Prithvi WxC-style spatiotemporal/transformer modeling

## Future Work (Prithvi WxC Direction)

This CNN baseline is intentionally simple. The next research upgrade is to move
from static spatial mapping toward stronger spatiotemporal architectures,
including transformer-based approaches inspired by Prithvi WxC.

## Common Errors

- Missing ERA5 file:
  - `ERA5 file not found: data_raw/era5_georgia_temp.nc`
  - Fix: run `python data_pipeline/download_era5_georgia.py`

- Missing PRISM rasters:
  - `No PRISM raster files found in data_raw/prism`
  - Fix: run `python data_pipeline/download_prism.py`

- Missing checkpoint:
  - `Checkpoint not found: checkpoints/cnn_downscaler_best.pt`
  - Fix: run training first, then evaluation

## Repository Structure

```text
robust-earth-forecast/
├── data_pipeline/
├── datasets/
├── models/
├── training/
├── evaluation/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

Venkata Vivek Panguluri
