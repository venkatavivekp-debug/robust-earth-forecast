# Robust Earth Forecast

A baseline deep learning pipeline for ERA5 to PRISM climate downscaling over Georgia.

## Overview

This repository implements a baseline deep learning pipeline for climate downscaling, mapping ERA5 reanalysis temperature to high-resolution PRISM observations over Georgia. The pipeline includes data ingestion (NetCDF and raster formats), spatial alignment, supervised dataset construction, and CNN-based modeling with evaluation through RMSE, MAE, and visual comparisons.

The current model captures large-scale temperature patterns but smooths finer spatial variability, reflecting the inherent gap between coarse reanalysis data and high-resolution observations.

The implementation is intentionally structured as a strong baseline for further extensions. Future work includes incorporating temporal context (multi-step ERA5 inputs), integrating additional data sources such as remote sensing or drone imagery, and moving toward uncertainty-aware forecasting approaches.

## Data

- ERA5: coarse-resolution reanalysis temperature fields loaded from NetCDF with `xarray`.
- PRISM: higher-resolution daily temperature rasters loaded with `rioxarray`.
- The pipeline performs spatial alignment to a shared geospatial frame and temporal matching by date before supervised training.

## What Is Implemented

- ERA5 ingestion and validation from NetCDF
- PRISM raster ingestion and validation (`.bil`, `.tif`, `.tiff`, `.asc`)
- Temporal input windows: `ERA5(t-k+1 ... t) -> PRISM(t)`
- Baseline CNN downscaling model
- Training pipeline with checkpointing
- Evaluation with RMSE, MAE, and comparison plots

## Pipeline Summary

```text
ERA5 -> spatial/temporal alignment -> dataset -> baseline model -> prediction -> evaluation
```

## Run Instructions

Run from project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_pipeline/download_era5_georgia.py
python data_pipeline/download_prism.py
python training/train_downscaler.py --history-length 3 --epochs 20 --batch-size 4 --learning-rate 1e-3
python evaluation/evaluate_model.py --history-length 3 --checkpoint-path checkpoints/cnn_downscaler_best.pt --num-samples 8 --num-plots 1 --results-dir results/evaluation
jupyter notebook notebooks/
```

Default paths used by training/evaluation:

- ERA5: `data_raw/era5_georgia_temp.nc`
- PRISM: `data_raw/prism`
- Checkpoint: `checkpoints/cnn_downscaler_best.pt`

## Results

![ERA5 to PRISM Comparison](results/evaluation/comparison_1_20230101.png)

- Left: ERA5 input (coarse resolution, upsampled for visual comparison)
- Middle: model prediction
- Right: PRISM ground truth

Metrics are reported in `results/evaluation/metrics.json`.

- RMSE: 14.5470
- MAE: 14.3941

## Limitations

- Temporal context is limited to a short fixed history window.
- The CNN baseline introduces spatial smoothing relative to PRISM fine-scale variability.
- Multi-source inputs are not yet integrated.
- Current experiments use a limited local date range.

## Future Work

- Temporal modeling with ConvLSTM or other sequence-based architectures
- Multi-source data integration (remote sensing, drone imagery, terrain features)
- Higher-dimensional spatiotemporal multi-channel inputs
- Uncertainty-aware prediction with probabilistic objectives

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
