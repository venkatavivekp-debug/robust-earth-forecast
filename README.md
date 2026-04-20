# Robust Earth Forecast

Regional ERA5 → PRISM temperature downscaling over Georgia using simple baselines and two temporal/spatial neural models (CNN, ConvLSTM).

## Problem

- **Input**: ERA5 daily fields on a coarse grid.
- **Target**: PRISM daily temperature rasters on a finer grid.
- **Goal**: learn a supervised mapping from coarse atmospheric context to a higher-resolution regional field.

## Models

- **persistence**: upsample the most recent ERA5 temperature frame
- **linear**: global linear mapping on the upsampled persistence baseline
- **cnn**: spatial mapping from stacked ERA5 history → PRISM field
- **convlstm**: temporal sequence model over ERA5 history → PRISM field

This repository intentionally stays within these model families (no transformers, no large architectures).

## Repository layout

- `data_pipeline/`: download + validate ERA5/PRISM data
- `datasets/`: dataset construction and alignment checks
- `models/`: baselines and neural models
- `training/`: training + experiment runners
- `evaluation/`: evaluation and plotting

Generated artifacts are written under `results/` and `checkpoints/` but are **ignored by git**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data (example)

```bash
python3 data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite
python3 data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean
```

## Train a model

```bash
python3 training/train_downscaler.py \
  --model convlstm \
  --input-set t2m \
  --history-length 3 \
  --epochs 20 \
  --learning-rate 8e-4 \
  --weight-decay 1e-6 \
  --l1-weight 0.1 \
  --grad-clip 1.0 \
  --split-seed 42 \
  --seed 42
```

Training writes:
- a checkpoint to `checkpoints/`
- a training log (`.csv` + `.json`) to `results/`

## Evaluate

```bash
python3 evaluation/evaluate_model.py \
  --models persistence linear cnn convlstm \
  --input-set t2m \
  --history-length 3 \
  --num-samples 8 \
  --split-seed 42
```

Evaluation writes per-model `metrics.json` and a `baselines_summary.csv`. It validates that the CSV and JSON are consistent; mismatches raise `ValueError("Metrics mismatch between JSON and CSV")`.

## Experiments

### Temporal history-length sweep

Runs short train+eval loops for a list of history lengths and writes `results/temporal_analysis/temporal_summary.csv`.

```bash
python3 training/run_temporal_analysis.py --histories 1 3 6 --models cnn convlstm --epochs 5
```

### ERA5-variable ablation (optional)

This only applies if your ERA5 NetCDF contains multiple variables. The default example dataset in this repo ships with `t2m` only.

```bash
python3 training/run_ablation.py --model convlstm --era5-variables t2m --epochs 5
```

## Limitations

- Regional scope (Georgia) and limited time coverage by default
- Models are trained from scratch and are meant as baselines/controlled studies
- Results depend on the available ERA5 variables and the PRISM date coverage you download

## Future work

- Extend temporal coverage (more months/years)
- Expand ERA5 variable sets where available
- Improve calibration/uncertainty reporting after baseline performance is stable
