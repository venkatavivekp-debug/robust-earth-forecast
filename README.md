# Robust Earth Forecast

A spatiotemporal deep learning pipeline for regional climate downscaling from ERA5 to PRISM over Georgia.

## Problem Statement

This project learns downscaling from coarse ERA5 reanalysis temperature fields to higher-resolution PRISM temperature targets.

`ERA5(t-k+1 ... t) -> PRISM(t)`

- ERA5 provides the coarse spatiotemporal input context.
- PRISM provides the finer-resolution target raster for supervised learning.

## Why This Matters

Regional climate analysis depends on fine spatial detail, but many global products are coarse. A robust regional downscaling pipeline makes it possible to evaluate how temporal context and model choice affect local prediction quality.

## Implemented Pipeline

- Temporal ERA5/PRISM dataset construction with date alignment and history windows
- Persistence baseline
- Global linear baseline
- Spatial CNN baseline
- ConvLSTM temporal model (main model)
- Unified evaluation pipeline with RMSE, MAE, bias, per-model plots, and model-comparison outputs

## Data Requirements

Minimum files:

- `data_raw/era5_georgia_temp.nc`
- `data_raw/prism/` containing dated rasters (`.bil`, `.tif`, `.tiff`, `.asc`)

For temporal runs, usable samples depend on both date coverage and `--history-length`.

- Approximate usable samples: `num_aligned_dates - history_length + 1`
- At least 2 usable samples are required for train/validation split
- For `--history-length 5`, start with at least 10 PRISM days

If insufficient data is available, training/evaluation now fails with a clear message that includes the usable count, minimum required count, and exact download commands.

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
│       ├── baselines_summary.csv
│       ├── model_comparison.png
│       ├── cnn/
│       │   ├── comparison_*.png
│       │   └── metrics.json
│       └── convlstm/
│           ├── comparison_*.png
│           └── metrics.json
├── README.md
├── requirements.txt
└── .gitignore
```

## Run Order (Copy-Paste Ready)

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1
python data_pipeline/download_prism.py --start-date 20230101 --days 20 --variable tmean

python training/train_downscaler.py --model cnn --history-length 5 --epochs 20 --batch-size 4 --learning-rate 1e-3
python training/train_downscaler.py --model convlstm --history-length 5 --epochs 20 --batch-size 4 --learning-rate 1e-3

python evaluation/evaluate_model.py --models persistence linear cnn convlstm --history-length 5 --num-samples 8

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```

## Results

Sample CNN output:

![CNN Downscaling Output](results/evaluation/cnn/comparison_1_20230101.png)

Evaluation outputs:

- `results/evaluation/<model>/metrics.json`
- `results/evaluation/baselines_summary.csv`
- `results/evaluation/model_comparison.png`

Interpretation:

- The temporal model is expected to improve over the spatial CNN by using multi-step ERA5 context.
- Performance remains constrained by data volume, coarse-to-fine resolution gap, and limited predictor variables.

## Limitations

- Single region (Georgia)
- Single variable target (temperature)
- Short temporal context relative to seasonal and synoptic dynamics
- ConvLSTM implementation is intentionally compact and far smaller than foundation-scale weather models
- Fine-scale details can remain smoothed

## Relation to Published Work

This project is inspired by spatiotemporal weather and climate modeling literature, including ConvLSTM-style sequence modeling and large modern systems such as FourCastNet, GraphCast, and Prithvi WxC. It is a lightweight regional implementation and does not reproduce those large-scale foundation models. The focus here is a robust regional pipeline with interpretable model progression:

`persistence -> linear -> CNN -> ConvLSTM`

## Future Work

- Longer temporal windows and seasonal-context modeling
- Additional ERA5 predictor variables
- Multi-source fusion (remote sensing, terrain, drone imagery)
- Uncertainty-aware forecasting objectives
- Stronger model families after stabilizing the temporal baseline
