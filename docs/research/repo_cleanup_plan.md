# Repository cleanup plan

This is an audit, not a deletion list. Do not remove files until the plan is reviewed against the current local data/results that are not tracked by git.

## Full repository audit

### A. Core research code

- `data_pipeline/`: ERA5 and PRISM download/validation entry points.
- `datasets/`: dataset path resolution and ERA5/PRISM alignment.
- `models/`: architecture implementations and baselines.
- `training/train_downscaler.py`: current trainer CLI.
- `evaluation/evaluate_model.py`: current evaluation script and metric/plot writer.
- `scripts/run_core_experiments.py`: current archived grid runner for plain encoder-decoder (`cnn`) / ConvLSTM summaries.
- `scripts/check_spatial_artifacts.py`: border/center diagnostic.
- `scripts/spatial_error_analysis.py`: gradient/error diagnostic.
- `scripts/validate_results.py`, `scripts/summarize_results.py`, `scripts/export_final_comparison.py`: result bookkeeping.

### B. Experimental artifacts

- `docs/experiments/final_comparison.json`
- `docs/experiments/final_comparison_medium.json`
- `docs/experiments/stability_analysis.json`
- `docs/experiments/error_analysis.json`
- `docs/experiments/t2m_sweep_summary.json`
- `docs/experiments/results_summary.md`
- `docs/experiments/data_scaling.md`
- `docs/experiments/underperformance_diagnosis.md`

These files are the committed experiment record. Keep them even if wording later changes.

### C. Generated outputs

- `docs/images/*.png`: committed figures referenced by README/notebook/docs.
- Ignored local `results/`: run outputs, diagnostics, plots, logs, and verification artifacts.
- Ignored local `checkpoints/`: model checkpoints.
- Ignored local `data_raw/`: downloaded ERA5/PRISM data.

Generated does not mean disposable. Some generated outputs are the evidence behind committed summaries.

### D. Duplicate or stale code

- `training/run_temporal_analysis.py`: appears stale. It passes `--forecast-horizon`, which the current trainer/evaluator do not define.
- `training/run_ablation.py`: appears stale. It passes `--era5-variable` and `--forecast-horizon`, which the current trainer/evaluator do not define.
- `docs/research/problem_structure_notes.md`: overlaps with `docs/research/problem_structure.md`. Keep for now, merge or retire after review.
- `docs/research/next_experiment_plan.md`: still leans temporal-first. It should be revised after the spatial baseline plan is accepted.

### E. Misleading naming

- `models/cnn_downscaler.py` / `CNNDownscaler`: better described as a plain spatial encoder/readout or encoder-decoder-style baseline without skip connections. Keep the CLI alias `cnn` for old results. Compatibility aliases now exist; a file/class rename can wait until old checkpoints and scripts are handled deliberately.
- README/docs should clarify whether an archived `cnn` label means the plain encoder-decoder baseline or a skip-connected U-Net baseline.
- `scripts/export_final_comparison.py`: output language is ConvLSTM-oriented; leave for archived summaries unless the result schema changes.

### F. Scratch/debug content

- `.DS_Store`, `__pycache__/`, `.pytest_cache/`, `.cache/`, notebook `.ipynb_checkpoints/`.
- Ignored local verification folders such as `results/_final_*`, `results/_harden_verify/`, `results/_tmp_training/`, `results/_verify_eval/`, `results/_verify_train/`, `results/tmp_eval/`, and `results/tmp_training/`.
- Ignored local verification logs under `results/evaluation_*` and `results/training_logs_*`.

### G. Important evidence to preserve

- `docs/experiments/*`: committed metrics and experiment interpretation.
- `docs/images/model_comparison.png`, `sample_prediction.png`, `error_map.png`, `spatial_model_check_unet_residual.png`, and gradient/error figures.
- `notebooks/analysis.ipynb`: main visual/interpretive notebook.
- Local `results/experiments*`, `results/diagnostics/`, and `results/experiments_medium_spatial/` until the corresponding committed docs/figures can be reproduced.

## Keep as core infrastructure

- Dataset download/alignment code.
- Current trainer/evaluator.
- Model files, including the current plain baseline, ConvLSTM baseline, and existing U-Net implementation.
- Baseline helpers.
- Validation, summarization, spatial artifact, and spatial error scripts.
- Committed experiment docs and figures.
- Notebook and README.

## Rename/refactor

- Decide later whether to rename `CNNDownscaler` and `models/cnn_downscaler.py`; keep current aliases until old checkpoints/results are migrated.
- Bring `training/run_temporal_analysis.py` and `training/run_ablation.py` in sync with the current CLI or archive them as legacy runners.
- Update `docs/research/next_experiment_plan.md` so spatial reconstruction comes before temporal complexity.
- Decide whether `docs/research/problem_structure_notes.md` should merge into `problem_structure.md`.
- Decide whether `scripts/run_core_experiments.py` remains the archived plain encoder-decoder / ConvLSTM grid runner or becomes a broader controlled-comparison runner.

## Archive later

- Old temporal/ablation/tuning local result folders if they are not referenced by docs or notebook.
- Legacy runner scripts after a replacement path exists.
- Duplicate research notes after the new problem-structure framing is reviewed.

Archiving means moving or clearly labeling; it does not mean deleting experiment evidence.

## Safe generated outputs

Safe to remove from the local working copy after review:

- `.DS_Store`, `.cache/`, `__pycache__/`, `.pytest_cache/`.
- Notebook checkpoints.
- Rendered notebooks that are not referenced by docs.
- Empty local legacy `data/raw/` directories if they remain empty and unused.
- Local verification output folders that are not referenced by docs/notebook.

## Needs confirmation before deletion

- `data_raw/`: raw ERA5/PRISM files are ignored by git but expensive to rebuild.
- `checkpoints/`: ignored local checkpoints may be needed to reproduce figures/diagnostics.
- `results/experiments/`, `results/experiments_medium/`, `results/experiments_medium_seed_*`, `results/experiments_medium_spatial/`, and `results/diagnostics/`.
- Any committed figure under `docs/images/`.
- Any committed JSON/markdown under `docs/experiments/`.
- `notebooks/analysis.ipynb`.

## Important experiment evidence to preserve

- Small and medium comparison JSONs.
- Multi-seed stability JSON.
- Spatial artifact diagnosis and residual U-Net comparison.
- Gradient-vs-error analysis figures and JSON.
- README figures.
- Notebook outputs that show the visual comparison and spatial checks.
