# Robust Earth Forecast

A literature-aligned regional climate downscaling pipeline for ERA5 to PRISM temperature prediction over Georgia.

## Problem Statement

Regional climate analysis often requires finer spatial detail than global reanalysis products provide. This project formulates downscaling as a supervised learning problem:

`ERA5(t-k+1 ... t) -> PRISM(t)`

where ERA5 provides coarse-resolution temperature context and PRISM provides higher-resolution daily targets.

## Motivation From Literature

- ConvLSTM (Shi et al.) provides a practical spatiotemporal modeling approach for sequence-to-field prediction.
- FourCastNet and GraphCast show the impact of learned spatiotemporal dynamics for weather forecasting at scale.
- Prithvi WxC highlights the trend toward large weather-climate foundation models.

This repository is not a reproduction of those large systems. It is a compact regional pipeline that is technically aligned with those directions while remaining tractable for iterative research.

## What Is Implemented

- ERA5 ingestion from NetCDF using `xarray`
- PRISM ingestion from rasters (`.bil`, `.tif`, `.tiff`, `.asc`) using `rioxarray`
- Temporal dataset construction with date matching and history windows
- Baselines:
  - persistence baseline (upsampled latest ERA5 frame)
  - global linear baseline (`y = a*x + b` fitted on training split)
  - spatial CNN baseline
- Main temporal model:
  - ConvLSTM downscaler
- Training pipeline with model selection (`cnn` or `convlstm`) and checkpointing
- Evaluation pipeline with RMSE, MAE, bias, per-model plots, and model-comparison outputs

## Repository Structure

```text
robust-earth-forecast/
├── data_pipeline/
├── datasets/
├── models/
├── training/
├── evaluation/
├── notebooks/
│   └── era5_prism_downscaling.ipynb
├── results/
│   └── evaluation/
├── README.md
├── requirements.txt
└── .gitignore
```

## Run Commands

Run from project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1
python data_pipeline/download_prism.py --start-date 20230101 --days 3 --variable tmean

python training/train_downscaler.py --model cnn --history-length 5 --epochs 20 --batch-size 4 --learning-rate 1e-3
python training/train_downscaler.py --model convlstm --history-length 5 --epochs 20 --batch-size 4 --learning-rate 1e-3

python evaluation/evaluate_model.py \
  --models persistence linear cnn convlstm \
  --history-length 5 \
  --cnn-checkpoint checkpoints/cnn_best.pt \
  --convlstm-checkpoint checkpoints/convlstm_best.pt \
  --num-samples 8

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```

Default data/checkpoint locations:

- ERA5: `data_raw/era5_georgia_temp.nc`
- PRISM: `data_raw/prism`
- CNN checkpoint: `checkpoints/cnn_best.pt`
- ConvLSTM checkpoint: `checkpoints/convlstm_best.pt`

## Results

Sample CNN output:

![CNN Downscaling Output](results/evaluation/cnn/comparison_1_20230101.png)

Interpretation:

- left/top panel: ERA5 input upsampled to PRISM grid
- prediction panel: model output
- target panel: PRISM reference
- error panel: absolute residual map

Metric outputs are written to:

- `results/evaluation/<model>/metrics.json`
- `results/evaluation/metrics_summary.csv`
- `results/evaluation/model_comparison.png`

Tracked sample metric file:

- `results/evaluation/cnn/metrics.json` (RMSE/MAE from a prior run)

## Limitations

- Single region (Georgia) and a limited date range
- Single target variable (temperature)
- Temporal context is short relative to seasonal and synoptic variability
- CNN/ConvLSTM outputs can smooth fine-scale terrain-driven structure

## Future Work

- Multi-variable ERA5 predictors (e.g., humidity, wind, pressure)
- Multimodal fusion with remote sensing, terrain, and airborne/drone data
- Uncertainty-aware objectives and calibrated probabilistic outputs
- Adaptation and fine-tuning strategies inspired by large weather-climate models

## Common Failure Modes

- `ERA5 file not found`: run `python data_pipeline/download_era5_georgia.py`
- `PRISM path not found` or no rasters: run `python data_pipeline/download_prism.py`
- missing checkpoints during evaluation: train `cnn` and/or `convlstm` first
