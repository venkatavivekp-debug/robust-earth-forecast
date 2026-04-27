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

**CNN (same `run_core_experiments.py` grid, not in this JSON):** README and run logs for that sweep report CNN **worse than persistence** for every `(t2m \| core4) × (1, 3, 6)` cell; recover exact CNN RMSE from `results/experiments/<input>_h<h>/evaluation/` if needed.

**Interpretation**

ConvLSTM **helps most** at **`core4` + history 3**, where RMSE is lowest in this archive. It **fails badly at history 1** for both input sets (RMSE well above persistence). **`t2m` + history 6** also lands above persistence, while **`core4` + history 6** still beats persistence but is **worse than history 3**—so longer context is not automatically better. The pattern is **unstable across history**: large swing from h1 to h3, then partial loss at h6 for `core4` and full loss for `t2m` at h6, consistent with **very small validation *N*** and optimization noise rather than a monotone “more days always help” law.
