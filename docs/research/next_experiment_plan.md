# Next experiment plan

Three focused studies. Commands assume the repo root as working directory, dependencies from `requirements.txt`, and existing `data_raw/era5_georgia_multi.nc` plus PRISM rasters under `data_raw/prism`. Adjust paths if yours differ.

Use `scripts/run_core_experiments.py` for sweeps: it trains CNN and ConvLSTM, evaluates against persistence, and appends `results/experiments/summary.csv`. Add `--overwrite` to replace prior runs under `results/experiments/`.

---

## A. Temporal context scaling (history = 3, 6, 12)

**Hypothesis.** ConvLSTM RMSE versus persistence improves as history increases into the “enough dynamics, not too many parameters for *N*” band; beyond that, RMSE may flatten or worsen if the effective sample size drops (longer history removes the first days from the dataset) or optimization becomes harder.

**Command.**

```bash
python3 scripts/run_core_experiments.py \
  --input-sets core4 \
  --histories 3 6 12 \
  --split-seed 42 \
  --seed 42 \
  --device auto \
  --overwrite
```

**Expected behavior.** `results/experiments/summary.csv` lists rows per `(experiment, model)` with `delta_vs_persistence` negative when RMSE beats persistence. Compare ConvLSTM rows for `core4_h3`, `core4_h6`, `core4_h12`.

**Success condition.** ConvLSTM at one history value shows clearly lower RMSE than persistence and lower than the weaker history settings on the same split (e.g. best `delta_vs_persistence` among histories).

**Failure interpretation.** If longer history consistently hurts: likely too few samples relative to sequence length or increased alignment drops; shorten history or add months of data before changing the model.

---

## B. Data scaling (extend time range)

**Hypothesis.** With the same code and hyperparameters, more aligned days shrink validation uncertainty and tend to lower RMSE or at least stabilize ranking against persistence.

**Steps.**

1. Download additional ERA5 months and matching PRISM `tmean` days (see `data_pipeline/download_era5_georgia.py` and `data_pipeline/download_prism.py`). Rebuild or extend `era5_georgia_multi.nc` as you already do for multi-variable files.
2. Re-run the core grid on the enlarged corpus:

```bash
python3 scripts/run_core_experiments.py \
  --input-sets t2m core4 \
  --histories 3 6 \
  --split-seed 42 \
  --seed 42 \
  --device auto \
  --overwrite
```

**Expected behavior.** More rows in `summary.csv` correspond to more usable samples; persistence RMSE may shift slightly; ConvLSTM `delta_vs_persistence` ideally becomes more stable across seeds if you repeat with an alternate `--split-seed`.

**Success condition.** Validation RMSE variance decreases (e.g. across split seeds) or ConvLSTM improves over persistence in settings where it previously failed.

**Failure interpretation.** If metrics barely move, the bottleneck may be representation (coarse ERA5 vs PRISM detail) or misalignment, not raw count alone; document date gaps and variable availability.

---

## C. Variable ablation (t2m only vs wind + pressure vs extended)

**Hypothesis.** Extra channels help when they are physically informative and present in the NetCDF; `extended` helps only if all listed variables exist—otherwise the dataset loader raises a clear error.

**Commands (recommended path: explicit `input-set`, matches training API).**

*t2m only (baseline ablation):*

```bash
python3 training/train_downscaler.py \
  --model convlstm \
  --input-set t2m \
  --history-length 3 \
  --epochs 80 \
  --learning-rate 3e-4 \
  --weight-decay 1e-6 \
  --split-seed 42 \
  --seed 42 \
  --device auto \
  --checkpoint-out checkpoints/convlstm_ablate_t2m.pt \
  --training-results-dir results/training_ablation/t2m

python3 evaluation/evaluate_model.py \
  --models persistence era5_upsampled linear convlstm \
  --input-set t2m \
  --history-length 3 \
  --split-seed 42 \
  --convlstm-checkpoint checkpoints/convlstm_ablate_t2m.pt \
  --results-dir results/eval_ablation/t2m
```

*core4 (t2m + u10 + v10 + sp):*

```bash
python3 training/train_downscaler.py \
  --model convlstm \
  --input-set core4 \
  --history-length 3 \
  --epochs 80 \
  --learning-rate 3e-4 \
  --weight-decay 1e-6 \
  --split-seed 42 \
  --seed 42 \
  --device auto \
  --checkpoint-out checkpoints/convlstm_ablate_core4.pt \
  --training-results-dir results/training_ablation/core4

python3 evaluation/evaluate_model.py \
  --models persistence era5_upsampled linear convlstm \
  --input-set core4 \
  --history-length 3 \
  --split-seed 42 \
  --convlstm-checkpoint checkpoints/convlstm_ablate_core4.pt \
  --results-dir results/eval_ablation/core4
```

*extended (adds precipitation and humidity levels if variables exist in the file):*

```bash
python3 training/train_downscaler.py \
  --model convlstm \
  --input-set extended \
  --history-length 3 \
  --epochs 80 \
  --learning-rate 3e-4 \
  --weight-decay 1e-6 \
  --split-seed 42 \
  --seed 42 \
  --device auto \
  --checkpoint-out checkpoints/convlstm_ablate_extended.pt \
  --training-results-dir results/training_ablation/extended

python3 evaluation/evaluate_model.py \
  --models persistence era5_upsampled linear convlstm \
  --input-set extended \
  --history-length 3 \
  --split-seed 42 \
  --convlstm-checkpoint checkpoints/convlstm_ablate_extended.pt \
  --results-dir results/eval_ablation/extended
```

Train and pass a CNN checkpoint as well if you want CNN in `--models`; the commands above isolate ConvLSTM against baselines only.

**Expected behavior.** RMSE ordering `t2m` ≥ `core4` often holds for temperature downscaling when winds and pressure are available; `extended` may help or add noise.

**Success condition.** Monotonic or interpretable RMSE ordering with physically plausible gains from `core4`; no gain from `extended` implies redundancy or estimation noise at current *N*.

**Failure interpretation.** If extra variables hurt: check normalization, collinearity, or missing noisy channels; confirm variable names in the NetCDF match `datasets/prism_dataset.py` aliases.

---

## Note on other scripts

`training/run_temporal_analysis.py` and `training/run_ablation.py` pass CLI flags that the current `training/train_downscaler.py` / `evaluation/evaluate_model.py` do not define. Prefer the commands above until those runners are brought in sync with the trainer.
