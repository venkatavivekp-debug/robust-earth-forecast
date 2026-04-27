# Robust Earth Forecast

## Overview

Daily **ERA5** fields over **Georgia** (default bounding box in `data_pipeline/download_era5_georgia.py`: roughly 30–35°N, 80–85°W) are aligned to **PRISM** daily mean temperature (`tmean`) on a finer grid. The code learns a **supervised map** from a short **history** of coarse inputs to the target day’s PRISM field—**spatiotemporal** in the sense that ConvLSTM (and the CNN, via stacked history channels) consume multiple consecutive days, not a single snapshot.

## Data and Setup

**ERA5 (predictors)**  
The downloader builds `data_raw/era5_georgia_multi.nc` for a chosen calendar month. Surface variables include **2 m temperature (`t2m`)**, **10 m u/v wind (`u10`, `v10`)**, **surface pressure (`sp`)**, **total precipitation (`tp`)**, and **2 m relative humidity (`rh2m`)** derived from dewpoint; pressure-level **temperature, geopotential height, and relative humidity** at **850 hPa and 500 hPa** are merged for the extended channel list in `datasets/prism_dataset.py`.

**PRISM (target)**  
**Daily mean temperature** rasters (`tmean`), same dates as ERA5 after alignment. Files live under `data_raw/prism/` (see `data_pipeline/download_prism.py`).

**Temporal window**  
Examples use **January 2023** (`--year 2023 --month 1` for ERA5; `--start-date 20230101 --days 30` for PRISM). The committed summaries used **18** aligned daily pairs; with `history_length=3`, each sample uses **three consecutive ERA5 days** ending on the PRISM date.

**Input format**  
Tensor `[T, C, H, W]` with `T = history_length` (e.g. 1, 3, 6). **`t2m`**: one channel. **`core4`**: `t2m`, `u10`, `v10`, `sp`. **`extended`**: more channels if present in the NetCDF.

**Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Download**

```bash
python3 data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite
python3 data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean
```

**Train / evaluate / core grid**

```bash
python3 training/train_downscaler.py \
  --model convlstm --input-set core4 --history-length 3 \
  --epochs 80 --learning-rate 3e-4 --weight-decay 1e-6 \
  --l1-weight 0.1 --grad-clip 1.0 --split-seed 42 --seed 42

python3 evaluation/evaluate_model.py \
  --models persistence era5_upsampled linear cnn convlstm \
  --input-set core4 --history-length 3 --num-samples 8 --split-seed 42

python3 scripts/run_core_experiments.py \
  --input-sets t2m core4 --histories 1 3 6 --split-seed 42 --overwrite
```

Artifacts go to `results/` and `checkpoints/` (gitignored). Training writes `.csv` / `.json` logs; evaluation writes `baselines_summary.csv` and per-model `metrics.json`.

## Models

- **CNN (`CNNDownscaler`)**: stacks history in the channel dimension and applies a spatial CNN to the PRISM grid size.  
- **ConvLSTM (`ConvLSTMDownscaler`)**: ConvLSTM over time, then decodes to PRISM resolution.

**Baselines** in evaluation: **persistence**, **era5_upsampled**, **linear** (`evaluation/evaluate_model.py`).

## Experiments

Primary grid: **`scripts/run_core_experiments.py`** with **`t2m` and `core4`**, histories **`1, 3, 6`**, fixed **`--split-seed 42`** / **`--seed 42`**, script defaults for epochs, LR, early stopping, and batch size. Each cell trains CNN + ConvLSTM, then evaluates against persistence and other baselines; checkpoints and CSVs land under `results/experiments/<input>_h<h>/`.

**Training** (`training/train_downscaler.py`): **80/20** train/val split on indices, **normalization** from train indices only, checkpoint stores **`train_indices` / `val_indices`** for matched eval.

`training/run_temporal_analysis.py` and `training/run_ablation.py` are **out of sync** with the current `train_downscaler.py` CLI; use the commands above for reproducible runs.

## Results

From **`docs/experiments/final_comparison.json`** (same regime as the table). Persistence RMSE at history 3: **2.356**. **CNN** remained **worse than persistence** for all histories in that grid. **ConvLSTM at history 1** was **worse than persistence** for both input sets.

| setup | ConvLSTM RMSE | vs persistence (2.356) |
| --- | ---: | --- |
| `t2m`, history 3 | 2.004 | lower RMSE |
| `core4`, history 3 | 1.570 | lower RMSE |
| `core4`, history 6 | 2.304 | higher RMSE |
| `t2m`, history 6 | 2.992 | higher RMSE |

**`core4` beat `t2m` for ConvLSTM at history 3**; **history 6 did not improve on history 3** here.

Figures: `docs/images/model_comparison.png`, `sample_prediction.png`, `error_map.png`.

## Observations

- **History**: Moving from **1 → 3** days helped ConvLSTM a lot; **6** was **worse than 3** on this split—worth treating as a **local** optimum, not a guarantee.  
- **Training setup**: Same code yields **below-persistence** runs for weak configs; **hyperparameters and input set** matter.  
- **Sample size**: **18** pairs and **four** validation samples make RMSE a **rough** number; another month or another seed can move ordering.  
- **Spatial structure**: Gradient–error correlation from `docs/experiments/error_analysis.json` is **small** (**r ≈ 0.08** on mean maps, **≈ 0.04** pooled); fine-scale error is real but **not fully explained** by “steep terrain only.”

## Limitations

- **Few aligned days** and **short calendar span** in the default example.  
- **ERA5 vs PRISM** product mismatch (grid, physics, observations).  
- **High metric variance** on small validation sets.  
- **Georgia-only** bbox; no transfer claims.  
- Next step for the codebase itself: extend downloads, or repair the optional temporal/ablation scripts to call the current trainer flags.

## Research Context

**GraphCast**-class models and **Prithvi WxC**-style pretraining use **global domains and long multivariate archives** with large compute. This repository is a **small controlled** ERA5→PRISM study to test pipelines and baselines—not that scale.

## References

- [Literature notes](docs/research/literature_notes.md)  
- [Research gap](docs/research/research_gap.md)  
- [Next experiment plan](docs/research/next_experiment_plan.md)  

Companion: **`notebooks/analysis.ipynb`** (plots, checkpoint reload from `results/experiments/core4_h3/` when you have run the grid).
