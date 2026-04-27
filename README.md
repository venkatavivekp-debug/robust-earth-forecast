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

Full grid (Persistence + ConvLSTM RMSE, beat flags): **[`docs/experiments/results_summary.md`](docs/experiments/results_summary.md)** — values are copied from **`docs/experiments/final_comparison.json`**.

Key numbers from that JSON: persistence RMSE **2.355815142393112**; best ConvLSTM cell **`core4`, history 3**, RMSE **1.5704300999641418**; worst ConvLSTM rows **history 1** (e.g. **4.246121346950531** for `core4`, **4.956881523132324** for `t2m`). **CNN** is not stored in that JSON; the same sweep reports CNN **below persistence** for every cell (see per-run `results/experiments/*/evaluation/` when regenerated).

Figures: `docs/images/model_comparison.png`, `sample_prediction.png`, `error_map.png`.

## Observations

- **Failure cases**: ConvLSTM at **history 1** is far **above** persistence for both `t2m` and `core4`; **`t2m` + history 6** is **above** persistence (**2.992** vs **2.356**). **CNN** did not beat persistence for any history in that experiment. These are as important as the best cell.  
- **Why persistence is hard to beat**: the baseline upsamples the **latest** coarse 2 m temperature; for daily fields it already tracks large-scale warmth anomalies, so the model must learn **residual** fine-grid and product differences with **very few** target days.  
- **History 6 vs 3**: RMSE **worsens** from history 3 → 6 for ConvLSTM on **both** input sets, even though **`core4`+6** still **beats** persistence marginally—**longer context is not reliably better** here.  
- **Noise**: **18** aligned samples / **four** validation points → treat ordering between close RMSEs cautiously.  
- **Space**: gradient–error **r ≈ 0.08** (mean maps; **≈ 0.04** pooled) in `docs/experiments/error_analysis.json` — weak link, so **sub-grid detail** is not “solved” by this baseline.

## Reproducibility

Run `python3 scripts/run_core_experiments.py --input-sets t2m core4 --histories 1 3 6 --split-seed 42 --overwrite`; artifacts live under `results/experiments/<input>_h<h>/` (gitignored). Check committed JSON: `python3 scripts/validate_results.py`. Print the same RMSE grid: `python3 scripts/summarize_results.py`. After a new sweep, export an updated `final_comparison.json` and refresh `docs/experiments/results_summary.md` if you want the doc table to match.

## Data scaling experiment

More calendar days use the **same models and configs** via `--dataset-version medium` and paths under `data_raw/medium/` (`datasets/medium/paths.json`). **Small** stays the default demo (~**18** aligned samples). **Medium** metrics are not committed until you download data and run `python3 scripts/run_core_experiments.py --dataset-version medium --overwrite`; see **`docs/experiments/data_scaling.md`**.

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
