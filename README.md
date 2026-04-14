# Robust Earth Forecast

ERA5 to PRISM downscaling over Georgia with multi-variable inputs and temporal modeling.

## Pipeline

- ERA5 input variables: `t2m`, `u10`, `v10`, `sp`
- Target: PRISM daily temperature rasters
- Models: persistence, linear, CNN, ConvLSTM

## Why early training was weak

The weak runs were mostly undertrained and sensitive to optimizer settings. ConvLSTM improved after using a longer window, stable learning-rate scheduling, and consistent train-split normalization.

## Training and tuning

Training now includes:

- train-split input normalization
- gradient clipping
- weight decay
- LR scheduler
- per-epoch train/val/lr logs
- best-checkpoint saving
- loss-curve plots and JSON summaries in `results/training_logs/`

Sweep script (`results/tuning/`):

- learning rate: `1e-3`, `5e-4`, `1e-4`
- history length: `3`, `6`
- weight decay: `0`, `1e-5`

## Run

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1
python data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean

python training/tune_downscaler.py --models cnn convlstm --history-lengths 3 6 --learning-rates 1e-3 5e-4 1e-4 --weight-decays 0 1e-5 --epochs 20 --split-seed 42

python training/train_downscaler.py --model cnn --history-length 6 --epochs 30 --learning-rate 5e-4 --weight-decay 0 --split-seed 42 --grad-clip 1.0 --run-name cnn
python training/train_downscaler.py --model convlstm --history-length 6 --epochs 40 --learning-rate 5e-4 --weight-decay 1e-5 --split-seed 42 --grad-clip 1.0 --run-name convlstm

python evaluation/evaluate_model.py --models persistence linear cnn convlstm --history-length 6 --num-samples 8 --split-seed 42

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```

## Results

![Model Comparison](results/evaluation/model_comparison.png)

Latest history-6 metrics (`results/evaluation/baselines_summary.csv`):

- persistence: RMSE 3.993, MAE 3.316, CORR 0.650
- linear: RMSE 3.285, MAE 2.835, CORR 0.650
- cnn: RMSE 3.757, MAE 2.993, CORR 0.242
- convlstm: RMSE 2.425, MAE 2.045, CORR 0.787

ConvLSTM is now clearly stronger than CNN and linear on this split after tuning.

## Temporal extension

History length was extended from 3 to 6 for the main temporal run, with an optional history-8 check.

ConvLSTM becomes more competitive as temporal context increases, but still depends heavily on data size.

## Limitations

- one region (Georgia)
- limited date range
- single target variable
- performance still depends on sample count and seasonal coverage
