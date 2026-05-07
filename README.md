# Robust Earth Forecast

ERA5 -> PRISM daily temperature downscaling over Georgia. The repo trains small CNN and ConvLSTM baselines that map a short ERA5 history to the target-day PRISM `tmean` field, then checks them against persistence and simple interpolation baselines.

This is a small regional experiment, not a global weather model. The useful part is the controlled ERA5/PRISM pipeline, the baseline comparison, and the result stability check.

## Result Snapshot

| Dataset | Best config | Best RMSE | Persistence RMSE | Delta vs persistence |
| --- | --- | ---: | ---: | ---: |
| small | ConvLSTM `core4_h3` | 1.5704300999641418 | 2.355815142393112 | -0.7853850424289703 |
| medium | ConvLSTM `core4_h3` | 1.581842489540577 | 2.966560184955597 | -1.38471769541502 |

The medium run uses more days, but the best single-split RMSE is basically flat. Its main benefit is stability: weak configurations fail less often, and multi-seed results make the split sensitivity visible.

![Model comparison across baselines and learned models](docs/images/model_comparison.png)

![Sample ConvLSTM prediction compared with PRISM target](docs/images/sample_prediction.png)

![Spatial error map for the best small ConvLSTM run](docs/images/error_map.png)

Figures above are committed outputs from the `core4_h3` evaluation run. Full tables are in [`docs/experiments/results_summary.md`](docs/experiments/results_summary.md), [`docs/experiments/data_scaling.md`](docs/experiments/data_scaling.md), and the JSON summaries under `docs/experiments/`.

## Short Readout

- Persistence is a hard baseline: latest ERA5 `t2m`, upsampled, already carries much of the daily temperature field.
- ConvLSTM is the strongest model family in the archived runs, but the exact winner still depends on split, input set, and history length.
- Moderate temporal context works better than history 1; longer history is not automatically better.
- More data improved stability more than peak RMSE.
- Spatial error has structure, but PRISM gradient alone explains little of it (`r ~= 0.08` on mean maps, `~0.04` pooled).

All results are reported with variability across splits rather than as a single clean run.

## Data and Models

- **Predictors:** ERA5 over Georgia, mainly `t2m` and `core4` (`t2m`, `u10`, `v10`, `sp`).
- **Target:** PRISM daily mean temperature (`tmean`).
- **Histories:** 1, 3, and 6 days.
- **Models:** CNN stacks history as channels; ConvLSTM keeps the time axis explicit.
- **Baselines:** persistence, upsampled ERA5, and a linear baseline.

Default small data is January 2023. Medium data is 2023-01-01 through 2023-03-31.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite
python3 data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean

python3 scripts/run_core_experiments.py \
  --input-sets t2m core4 \
  --histories 1 3 6 \
  --split-seed 42 \
  --seed 42 \
  --overwrite

python3 scripts/validate_results.py --dataset-version small
python3 scripts/summarize_results.py --dataset-version small
```

Medium validation:

```bash
python3 scripts/validate_results.py --dataset-version medium
python3 scripts/summarize_results.py --dataset-version medium
```

Use `scripts/run_core_experiments.py` for current sweeps. `training/run_temporal_analysis.py` and `training/run_ablation.py` are older runners and do not match the current trainer CLI.

## Limitations

- Short regional sample, especially in the default small run.
- Georgia-only bbox; no transfer claim.
- ERA5 and PRISM differ in grid, physics, and observation influence.
- RMSE rankings are split-sensitive, so the stability tables matter.
- Next useful step: add calendar coverage before changing model class.

## Notes

- Notebook companion: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)
- Literature notes: [`docs/research/literature_notes.md`](docs/research/literature_notes.md)
- Research gap: [`docs/research/research_gap.md`](docs/research/research_gap.md)
- Next experiment plan: [`docs/research/next_experiment_plan.md`](docs/research/next_experiment_plan.md)
