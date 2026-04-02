# Robust Earth Forecast

A baseline deep learning pipeline for ERA5 to PRISM climate downscaling over Georgia.

## Overview

This repository implements a temperature downscaling framework that maps coarse ERA5 fields to higher-resolution PRISM targets over Georgia. The objective is to establish a reproducible baseline model and evaluation pipeline before extending to richer spatiotemporal architectures.

## ERA5 vs PRISM

- ERA5: coarse-resolution global reanalysis with strong temporal coverage.
- PRISM: higher-resolution gridded climate product over regional domains.

The downscaling task learns the mapping from ERA5 structure to PRISM local detail.

## What Is Implemented

- ERA5 ingestion from NetCDF using `xarray`
- PRISM ingestion from raster products using `rioxarray`
- Temporal input windows: `ERA5(t-k+1 ... t) -> PRISM(t)`
- Baseline CNN downscaling model
- Evaluation pipeline with RMSE, MAE, and spatial comparison plots

## Pipeline Summary

```text
ERA5 -> spatial/temporal alignment -> dataset -> baseline model -> prediction -> evaluation pipeline
```

## Run Instructions

Execute from project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_pipeline/download_era5_georgia.py
python data_pipeline/download_prism.py
python training/train_downscaler.py \
  --history-length 3 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --checkpoint-out checkpoints/cnn_downscaler_best.pt
python evaluation/evaluate_model.py \
  --history-length 3 \
  --checkpoint-path checkpoints/cnn_downscaler_best.pt \
  --num-samples 8 \
  --num-plots 1 \
  --results-dir results/evaluation
jupyter notebook notebooks/
```

Default paths used by training/evaluation scripts:

- ERA5: `data_raw/era5_georgia_temp.nc`
- PRISM: `data_raw/prism`
- Checkpoint: `checkpoints/cnn_downscaler_best.pt`

## Results

Evaluation outputs are written to:

- `results/evaluation/comparison_*.png`
- `results/evaluation/metrics.json`

Current tracked output:

![Downscaling Output](results/evaluation/comparison_1_20230101.png)

Interpretation:

- The baseline model captures large-scale thermal structure.
- Fine spatial details are partially smoothed, consistent with a compact CNN baseline.

## Limitations

- Temporal context is limited to a short fixed history window.
- No multi-source predictors are included yet.
- CNN outputs are spatially smooth relative to PRISM fine structures.
- Current experiments use a limited local dataset size.

## Future Work

- Temporal sequence encoders (ConvLSTM / transformer-based models)
- Multi-source inputs (remote sensing, terrain, drone or airborne products)
- Uncertainty-aware prediction (multi-output heads or probabilistic objectives)

## Common Errors

- Missing ERA5 file:
  - `ERA5 file not found: data_raw/era5_georgia_temp.nc`
  - Run: `python data_pipeline/download_era5_georgia.py`

- Missing PRISM rasters:
  - `No PRISM raster files found in data_raw/prism`
  - Run: `python data_pipeline/download_prism.py`

- Missing checkpoint:
  - `Checkpoint not found: checkpoints/cnn_downscaler_best.pt`
  - Run training before evaluation.

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
