# Robust Earth Forecast

ERA5 to PRISM spatiotemporal downscaling over Georgia with multi-variable inputs and tuned CNN/ConvLSTM baselines.

## Pipeline

- ERA5 ingestion (single-level + pressure-level) and PRISM raster ingestion
- Spatial clipping, temporal alignment, supervised sample building
- Models: persistence, linear, CNN, ConvLSTM
- Metrics: RMSE, MAE, bias, correlation

## Input variables

Current `extended` input set includes:

- `t2m`, `u10`, `v10`, `sp`
- `tp`, `rh2m`
- `rh_850`, `rh_500`
- `t_850`, `t_500`
- `gh_850`, `gh_500`

## What was causing weak training

Main issues were short training runs and optimizer sensitivity. The updates below improved stability:

- train-split per-channel normalization with checkpoint reuse in evaluation
- LR scheduler + gradient clipping + weight decay
- longer training runs with saved best checkpoints
- consistent split seed across training and evaluation

## Training and tuning artifacts

Saved under:

- `results/training_logs/` (loss curves + per-epoch logs)
- `results/tuning/` (sweep summary + best config)

Tuning sweep covers:

- learning rate: `1e-3`, `5e-4`, `1e-4`
- history length: `3`, `6`
- weight decay: `0`, `1e-5`

## Temporal depth analysis

`results/temporal_analysis/temporal_summary.csv` compares CNN and ConvLSTM for history lengths `1`, `3`, `6`.

Observation: ConvLSTM benefits from temporal context (`history=3/6`) while CNN gains are smaller.

## Input ablation

`results/ablation/ablation_summary.csv` compares ConvLSTM with:

- `t2m` only
- `core4`
- `extended`

Observation: the extended variable set gives the best RMSE/MAE/correlation.

## Latest tuned evaluation

![Model Comparison](results/evaluation/model_comparison.png)

From `results/evaluation/baselines_summary.csv` (history=3, extended input set):

- persistence: RMSE 3.251, MAE 2.611, CORR 0.640
- linear: RMSE 2.965, MAE 2.605, CORR 0.640
- cnn: RMSE 3.121, MAE 2.566, CORR 0.626
- convlstm: RMSE 1.749, MAE 1.377, CORR 0.778

ConvLSTM is strongest in the tuned configuration.

## Visual diagnostics

- `results/visualizations/sample_prediction.png`
- `results/visualizations/error_map.png`

## Run

```bash
git clone https://github.com/venkatavivekp-debug/robust-earth-forecast.git
cd robust-earth-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite
python data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean

python training/tune_downscaler.py --models cnn convlstm --input-set extended --history-lengths 3 6 --learning-rates 1e-3 5e-4 1e-4 --weight-decays 0 1e-5 --epochs 20 --split-seed 42

python training/train_downscaler.py --model cnn --input-set extended --history-length 3 --epochs 25 --learning-rate 1e-3 --weight-decay 0 --split-seed 42 --grad-clip 1.0 --run-name cnn
python training/train_downscaler.py --model convlstm --input-set extended --history-length 3 --epochs 30 --learning-rate 1e-3 --weight-decay 1e-5 --split-seed 42 --grad-clip 1.0 --run-name convlstm

python training/run_temporal_analysis.py --input-set extended --histories 1 3 6 --cnn-lr 1e-3 --convlstm-lr 1e-3 --cnn-epochs 20 --convlstm-epochs 25 --split-seed 42
python training/run_ablation.py --model convlstm --input-sets t2m core4 extended --history-length 6 --epochs 25 --learning-rate 1e-3 --weight-decay 1e-5 --split-seed 42

python evaluation/evaluate_model.py --input-set extended --models persistence linear cnn convlstm --history-length 3 --num-samples 8 --split-seed 42

jupyter notebook notebooks/era5_prism_downscaling.ipynb
```
