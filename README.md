# Robust Earth Forecast

ERA5 to PRISM regional downscaling over Georgia with multi-variable inputs and temporal modeling.

## Problem

ERA5 is coarse in space. PRISM is finer in space. The task is to learn:

`ERA5(t-k+1 ... t) -> PRISM(t)`

## Approach

Models used in this pipeline:

- persistence baseline
- global linear baseline
- CNN baseline
- ConvLSTM temporal model

ERA5 input channels:

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `surface_pressure`

Using only temperature was limiting. Adding wind and pressure improved the model behavior.

## Data Requirements

- `data_raw/era5_georgia_temp.nc`
- `data_raw/prism/` with dated rasters (`.bil`, `.tif`, `.tiff`, `.asc`)

For temporal runs, use at least 20-30 consecutive PRISM dates.

## Temporal Extension

The primary temporal experiment now uses `history-length=6` (instead of 3) to give ConvLSTM longer context.

ConvLSTM becomes more competitive as temporal context increases, but still depends heavily on data size.

## Run

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1
python data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean

python training/train_downscaler.py --model cnn --history-length 6 --epochs 30 --learning-rate 1e-3 --split-seed 42 --grad-clip 1.0
python training/train_downscaler.py --model convlstm --history-length 6 --epochs 40 --learning-rate 3e-4 --split-seed 42 --grad-clip 1.0

python evaluation/evaluate_model.py --models persistence linear cnn convlstm --history-length 6 --num-samples 8 --split-seed 42

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```

Optional secondary check:

```bash
python training/train_downscaler.py --model convlstm --history-length 8 --epochs 30 --learning-rate 3e-4 --split-seed 42 --grad-clip 1.0 --checkpoint-out checkpoints/convlstm_h8_best.pt
```

## Results

![Model Comparison](results/evaluation/model_comparison.png)

Latest history-6 metrics (`results/evaluation/baselines_summary.csv`):

- persistence: RMSE 3.993, MAE 3.316
- linear: RMSE 3.285, MAE 2.835
- cnn: RMSE 4.017, MAE 3.432
- convlstm: RMSE 3.011, MAE 2.660

ConvLSTM is now the strongest learned model on this split and slightly better than linear in RMSE/MAE.

## Current Observation

Simple baselines stay strong on small regional datasets.
Temporal context helps ConvLSTM when training is stable and the window is long enough.
The multi-variable setup is a stronger foundation than the earlier temperature-only version.

## Limitations

- one region (Georgia)
- short overall time range
- one target variable (temperature)
- deep models are still sensitive to sample count

## Relation to Existing Work

ConvLSTM is a standard temporal model for sequence-conditioned spatial prediction. This repository stays focused on a small regional downscaling setup rather than large global weather systems.

## Direction

- longer temporal windows with larger date ranges
- more ERA5 variables
- multi-source geospatial inputs
- uncertainty-aware forecasting
