# Robust Earth Forecast

A spatiotemporal deep learning pipeline for regional climate downscaling from ERA5 to PRISM over Georgia.

## Problem

ERA5 provides coarse reanalysis fields. PRISM provides higher-resolution observed temperature grids. The target is a date-aligned mapping from ERA5 history windows to PRISM daily rasters:

`ERA5(t-k+1 ... t) -> PRISM(t)`

## Approach

Model ladder in this repository:

- persistence baseline
- global linear baseline
- CNN spatial baseline
- ConvLSTM temporal model

ERA5 inputs use four variables: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, and `surface_pressure`.

Using only temperature was limiting. Adding wind and pressure improved the model behavior.

## Data

- ERA5: NetCDF from CDS (`data_raw/era5_georgia_temp.nc`)
- PRISM: daily rasters (`.bil`, `.tif`, `.tiff`, `.asc`) under `data_raw/prism/`

The dataset loader handles date parsing, CRS checks, reprojection to EPSG:4326, clipping to ERA5 extent, and temporal window construction.

## Run

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1
python data_pipeline/download_prism.py --start-date 20230101 --days 10 --variable tmean

python training/train_downscaler.py --model cnn --history-length 1 --epochs 5
python training/train_downscaler.py --model convlstm --history-length 3 --epochs 5
python training/train_downscaler.py --model cnn --history-length 3 --epochs 5

python evaluation/evaluate_model.py --models persistence linear cnn convlstm --history-length 3 --num-samples 8

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```

## Results

![Model Comparison](results/evaluation/model_comparison.png)

Figure layout: left is upsampled ERA5 input, middle panels are model predictions, right is PRISM target.

Recent run (`results/evaluation/baselines_summary.csv`):

- linear: RMSE 2.195, MAE 1.470
- cnn: RMSE 2.561, MAE 2.180
- convlstm: RMSE 9.278, MAE 9.062

With this small sample, linear remains strongest. Deep models need more temporal coverage and training volume.

## Limitations

- short temporal span in current run
- one region (Georgia)
- only four ERA5 predictors
- ConvLSTM still data-limited in this setup

## Relation to Existing Work

ConvLSTM is a standard spatiotemporal model for sequence-conditioned spatial prediction. This codebase follows that direction in a small regional downscaling setting and does not attempt to reproduce large global systems.

## Direction

- longer temporal windows
- more ERA5 variables
- multi-source geospatial inputs
- uncertainty-aware prediction
