# Results summary (archived experiment grid)

All numeric RMSE values below are **copied from** `docs/experiments/final_comparison.json` (no recomputation). Persistence RMSE is the single baseline value stored there (`improvement_vs_persistence.persistence_rmse`). **CNN** is not listed in that JSON; qualitative outcome for CNN on the same grid is noted under the table.

| Model | Variables | History | RMSE | Beats persistence |
| --- | --- | ---: | ---: | --- |
| Persistence | — | — | 2.355815142393112 | baseline |
| ConvLSTM | core4 | 1 | 4.246121346950531 | No |
| ConvLSTM | core4 | 3 | 1.5704300999641418 | Yes |
| ConvLSTM | core4 | 6 | 2.3035560051600137 | Yes |
| ConvLSTM | t2m | 1 | 4.956881523132324 | No |
| ConvLSTM | t2m | 3 | 2.0036779940128326 | Yes |
| ConvLSTM | t2m | 6 | 2.991683403650919 | No |

**CNN (same `run_core_experiments.py` grid, not in this JSON):** exact CNN rows live in `results/experiments/summary.csv` when the sweep has been run locally. In the current local summary, CNN is worse than persistence for most small-data cells; `core4_h6` is the exception.

**Readout**

ConvLSTM is best at **`core4` + history 3** in this archive. History 1 fails for both input sets, and history 6 does not improve on history 3. The history curve is unstable enough that this should be read as a small-split result, not a monotone rule about temporal context.
