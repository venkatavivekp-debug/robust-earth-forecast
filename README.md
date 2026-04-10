# Robust Earth Forecast

Spatiotemporal regional downscaling from ERA5 to PRISM temperature over Georgia.

## Problem

ERA5 gives broad atmospheric context but at coarse resolution. PRISM provides finer regional temperature grids. The task here is to learn a mapping from ERA5 history windows to a PRISM target map:

`ERA5(t-k+1 ... t) -> PRISM(t)`

## Approach

Model progression in this repository:

- persistence baseline (upsampled latest ERA5 frame)
- global linear baseline
- spatial CNN baseline
- ConvLSTM spatiotemporal model

The comparison is used to check whether temporal context improves downscaling quality over spatial-only baselines.

## Data Requirements

Expected inputs:

- `data_raw/era5_georgia_temp.nc`
- `data_raw/prism/` with dated rasters (`.bil`, `.tif`, `.tiff`, `.asc`)

Temporal sample count depends on both available dates and `--history-length`.

Approximate usable samples:

`usable_samples ~= aligned_dates - history_length + 1`

At least 2 usable samples are required for train/validation split. The training and evaluation scripts now fail with a clear message when this requirement is not met, including suggested download commands.

## How To Run

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

Tracked sample output:

![CNN Downscaling Output](results/evaluation/cnn/comparison_1_20230101.png)

Saved outputs:

- `results/evaluation/cnn/metrics.json`
- `results/evaluation/convlstm/metrics.json` (when ConvLSTM is evaluated)
- `results/evaluation/baselines_summary.csv`
- `results/evaluation/model_comparison.png` (when multiple model outputs are available)

Current interpretation:

- ERA5 structure is captured at large scales.
- Fine spatial structure remains harder to recover.
- ConvLSTM is expected to help when enough aligned temporal samples are available.

## Limitations

- single region (Georgia)
- limited predictor set
- short temporal windows in default runs
- data volume can be small for stable temporal training
- small model scale relative to modern large weather systems

## Relation to Existing Work

ConvLSTM is a standard spatiotemporal sequence model and is used here as the main temporal baseline. Modern weather ML systems (such as FourCastNet, GraphCast, and Prithvi WxC) motivate the emphasis on temporal dynamics. This repository focuses on a smaller regional downscaling setup rather than reproducing large global models.

## Future Direction

- longer temporal windows
- more ERA5 variables
- multi-source inputs (remote sensing, drone imagery, terrain)
- uncertainty-aware forecasting objectives
- stronger temporal architectures after expanding stable data coverage
