# Data scaling experiment (small vs medium)

**What stayed the same:** Georgia bbox, ERA5/PRISM pipeline, CNN/ConvLSTM code, train/eval metrics, `--split-seed 42`, `--seed 42`, and `scripts/run_core_experiments.py` hyperparameters.

**What changed:** only the calendar span and dataset paths. **Small** uses the January 2023 demo overlap. **Medium** uses January 1 through March 31, 2023 under `data_raw/medium/`.

## Sample counts

| Dataset | History 1 | History 3 | History 6 |
| --- | ---: | ---: | ---: |
| small | 20 | 18 | 15 |
| medium | 90 | 88 | 85 |

## RMSE comparison

Values come from `results/experiments/summary.csv`, `results/experiments_medium/summary.csv`, `docs/experiments/final_comparison.json`, and `docs/experiments/final_comparison_medium.json`. Beat flags use the matching persistence row for each dataset/history split.

| Dataset | Model | Variables | History | RMSE | Beats Persistence |
| --- | --- | --- | ---: | ---: | --- |
| small | Persistence | core4 | 3 | 2.355815142393112 | baseline |
| small | CNN | t2m | 1 | 5.2338986992836 | No |
| small | CNN | t2m | 3 | 4.504708349704742 | No |
| small | CNN | t2m | 6 | 5.41701078414917 | No |
| small | CNN | core4 | 1 | 4.496296465396881 | No |
| small | CNN | core4 | 3 | 3.4474770426750183 | No |
| small | CNN | core4 | 6 | 3.070897897084554 | Yes |
| small | ConvLSTM | t2m | 1 | 4.956881523132324 | No |
| small | ConvLSTM | t2m | 3 | 2.0036779940128326 | Yes |
| small | ConvLSTM | t2m | 6 | 2.991683403650919 | Yes |
| small | ConvLSTM | core4 | 1 | 4.246121346950531 | No |
| small | ConvLSTM | core4 | 3 | 1.5704300999641418 | Yes |
| small | ConvLSTM | core4 | 6 | 2.3035560051600137 | Yes |
| medium | Persistence | core4 | 3 | 2.966560184955597 | baseline |
| medium | CNN | t2m | 1 | 2.3131760358810425 | Yes |
| medium | CNN | t2m | 3 | 1.6414796262979507 | Yes |
| medium | CNN | t2m | 6 | 1.7203279733657837 | Yes |
| medium | CNN | core4 | 1 | 2.313338950276375 | Yes |
| medium | CNN | core4 | 3 | 2.1001143231987953 | Yes |
| medium | CNN | core4 | 6 | 2.3830468356609344 | Yes |
| medium | ConvLSTM | t2m | 1 | 2.561451241374016 | Yes |
| medium | ConvLSTM | t2m | 3 | 2.0794655084609985 | Yes |
| medium | ConvLSTM | t2m | 6 | 1.6947292536497116 | Yes |
| medium | ConvLSTM | core4 | 1 | 2.451022446155548 | Yes |
| medium | ConvLSTM | core4 | 3 | 1.581842489540577 | Yes |
| medium | ConvLSTM | core4 | 6 | 1.5877669304609299 | Yes |

## Analysis

1. **Does RMSE improve with more data?** Not for the single best ConvLSTM cell: small `core4_h3` is **1.5704**, while medium `core4_h3` is **1.5818**. That is essentially flat and slightly worse in absolute RMSE. The broader grid does improve: weak small-data cells, especially history 1 and CNN rows, become much better on medium.

2. **Does ConvLSTM benefit more than CNN?** Not universally. ConvLSTM remains best overall on medium (`core4_h3`, **1.5818**) and is clearly stronger for `core4_h3`/`core4_h6`, but CNN benefits a lot from the larger dataset and beats ConvLSTM on `t2m_h1`, `t2m_h3`, and `core4_h1`.

3. **Does temporal sensitivity stabilize?** Yes, partly. On small data, ConvLSTM history 1 fails badly and history 6 is mixed. On medium, every ConvLSTM history/input row beats its matching persistence baseline. History still matters: `core4_h3` and `core4_h6` are nearly tied, while `core4_h1` is worse.

4. **Are results still configuration-sensitive?** Yes. The best medium CNN uses `t2m_h3`, while the best medium ConvLSTM uses `core4_h3`. Extra variables help ConvLSTM but hurt CNN at histories 3 and 6. The ranking is more stable than small, but it is not architecture- or input-agnostic.

## Conclusion

Medium data makes the experiment more reliable and turns several failures into wins over persistence, but it does **not** prove that simply adding three months lowers the best achievable RMSE. The realistic next step is to keep the architecture fixed, add more calendar coverage, and repeat the same grid across at least one additional split seed.

## Stability Across Splits

Medium was rerun with `split_seed` / `seed` values **42**, **7**, and **123**, writing separate outputs under `results/experiments_medium_seed_<seed>/`. Values below are sample mean ± sample standard deviation across the three split/training seeds.

| Model | Variables | History | RMSE mean ± std |
| --- | --- | ---: | ---: |
| ConvLSTM | core4 | 6 | 1.4845 ± 0.1215 |
| ConvLSTM | core4 | 3 | 1.5289 ± 0.0806 |
| ConvLSTM | t2m | 6 | 1.6122 ± 0.2689 |
| CNN | t2m | 3 | 1.6345 ± 0.0451 |
| CNN | core4 | 3 | 1.8810 ± 0.2539 |
| ConvLSTM | t2m | 3 | 1.9028 ± 0.1547 |

The exact best row is **partially stable**, not invariant: ConvLSTM `core4_h3` wins seeds 42 and 7, while seed 123 wins with ConvLSTM `t2m_h6`. By mean RMSE, ConvLSTM `core4_h6` is best and ConvLSTM `core4_h3` is a close second. The main conclusion is stable at the model-family level: temporal ConvLSTM configurations are the strongest group, but history length and input set remain split-sensitive. ConvLSTM does **not** consistently beat CNN in every matched setting, and `core4` does **not** consistently beat `t2m` for CNN.
