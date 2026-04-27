# Data scaling experiment (small vs medium)

**What stayed the same:** Georgia bbox, ERA5 variable set (`core4` channels), CNN/ConvLSTM code, train/eval metrics, `--split-seed 42`, `--seed 42`, and `scripts/run_core_experiments.py` hyperparameters.

**What changed:** only the **calendar span** and on-disk paths (`datasets/small/` vs `datasets/medium/` presets → `data_raw/...` vs `data_raw/medium/...`).

## Sample counts

| Version | Typical aligned samples (this repo) | Notes |
| --- | ---: | --- |
| **small** | **18** | January 2023 demo overlap (`docs/experiments/final_comparison.json` regime). |
| **medium** | *Not run in CI* | Build with extended `--start-date` / `--end-date` downloads into `data_raw/medium/`, then count with `len(ERA5_PRISM_Dataset(...))` after PRISM exists for the same span. |

## RMSE table — small (archived JSON only)

Values copied from `docs/experiments/final_comparison.json`. **CNN** is not stored there; the controlled sweep still finds CNN **≥ persistence RMSE** for every `(input, history)` cell—see `results/experiments/*/evaluation/baselines_summary.csv` after a local run.

| Dataset | Model | Variables | History | RMSE | Beats persistence |
| --- | --- | --- | ---: | ---: | --- |
| small | Persistence | — | — | 2.355815142393112 | baseline |
| small | ConvLSTM | core4 | 1 | 4.246121346950531 | No |
| small | ConvLSTM | core4 | 3 | 1.5704300999641418 | Yes |
| small | ConvLSTM | core4 | 6 | 2.3035560051600137 | Yes |
| small | ConvLSTM | t2m | 1 | 4.956881523132324 | No |
| small | ConvLSTM | t2m | 3 | 2.0036779940128326 | Yes |
| small | ConvLSTM | t2m | 6 | 2.991683403650919 | No |

## RMSE table — medium

**Not populated yet.** After you run:

`python3 scripts/run_core_experiments.py --dataset-version medium --overwrite`

copy the key RMSE values into `docs/experiments/final_comparison_medium.json` (same schema as `final_comparison.json`) and extend this table. Use `python3 scripts/summarize_results.py --dataset-version medium` to print the grid from that JSON.

## Short analysis (small-only facts)

- **RMSE vs data:** we do **not** yet have a committed medium run, so we **cannot** claim RMSE drops with more days.  
- **Stability:** on **small *N***, ConvLSTM **beats and loses** to persistence depending on history/input—expect variance until medium results exist.  
- **ConvLSTM vs CNN:** CNN never beats persistence in the archived small sweep; ConvLSTM sometimes does—**no medium evidence** yet for whether that gap widens.

**Honest conclusion until medium is logged:** extending the calendar is the right experiment, but **you must re-run the identical script** on downloaded medium data and archive metrics before claiming any scaling benefit.
