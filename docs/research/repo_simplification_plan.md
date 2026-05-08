# Repository simplification plan

This is a conservative audit after the spatial benchmark, boundary ablation, and undertraining diagnosis. The goal is to keep the repo focused without deleting experiment evidence.

## Keep

- `data_pipeline/`, `datasets/`, `models/`, `training/train_downscaler.py`, and `evaluation/evaluate_model.py`: core reproducibility path.
- `scripts/run_spatial_benchmark.py`, `scripts/run_spatial_benchmark_seed_stability.py`, `scripts/diagnose_boundary_artifacts.py`, `scripts/run_boundary_ablation.py`, and `scripts/run_undertraining_diagnosis.py`: current controlled spatial/diagnostic runners.
- `docs/experiments/*.md` and committed JSON summaries: evidence trail for the results shown in README and notebook.
- `docs/images/`: committed figures used by README/notebook.
- `notebooks/analysis.ipynb`: companion analysis record, as long as it stays concise.

## Merge later

- `docs/research/problem_structure.md` and `docs/research/problem_structure_notes.md`: overlapping problem-framing notes.
- `docs/research/repo_cleanup_plan.md`, `docs/research/final_repo_refactor_plan.md`, and this file: keep while the repo is evolving, then merge into one maintenance note.
- `docs/research/current_baseline_definition.md`, `docs/research/unet_transition_plan.md`, and `docs/research/spatial_benchmark_protocol.md`: related architecture notes that can eventually become a single controlled-spatial-baseline document.
- `docs/research/research_gap.md` and `docs/research/literature_notes.md`: useful background, but older temporal framing should be updated after the topography phase.

## Archive later

- `training/run_temporal_analysis.py` and `training/run_ablation.py`: older runners that do not match the current trainer/evaluator CLI. Archive rather than delete until old outputs are no longer referenced.
- `training/tune_downscaler.py`: useful only if hyperparameter tuning is revived; not part of the current controlled evidence path.
- `scripts/run_core_experiments.py`: still needed for archived PlainEncoderDecoder/ConvLSTM tables, but should be labeled as an archived-grid runner in future docs.
- `models/convlstm_downscaler.py`: keep for reproducibility, but temporal modeling is not the next active phase.

## Safe to remove

- Ignored scratch outputs with names like `results/tmp_eval`, `results/tmp_training`, and `results/_tmp_training`.
- `.DS_Store`, notebook checkpoints, and temporary rendered notebooks if they appear.
- Empty failed-run folders under ignored `results/`.

Removed in this pass: ignored local scratch folders `results/_tmp_training`, `results/tmp_training`, and `results/tmp_eval`.

## Do not touch

- `data_raw/`, `checkpoints/`, and validated `results/` folders unless explicitly archiving local data outside git.
- `docs/experiments/final_comparison*.json`, `stability_analysis.json`, and `spatial_benchmark_summary.json`.
- Committed images referenced by README or notebook.
- Historical `cnn` aliases and old checkpoint-compatible model names.
