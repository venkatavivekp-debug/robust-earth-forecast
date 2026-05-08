# Spatial benchmark: multi-seed stability

## Purpose

Test whether conclusions from the single-seed controlled spatial benchmark (`docs/experiments/spatial_benchmark.md`) hold when the **train/validation split seed** changes, with **no other protocol changes**.

## Fixed controls

- **Dataset:** `dataset-version=medium` (`datasets/medium/paths.json`)
- **Inputs:** `core4`, **history** 3, **target mode** direct
- **Models:** persistence, `PlainEncoderDecoder`, skip-connected U-Net
- **Training:** same CLI hyperparameters as `scripts/run_spatial_benchmark.py` defaults (epochs 80, batch 4, LR 3e-4, plateau scheduler, early stopping patience 12, etc.)
- **Evaluation:** all validation samples per split (`num_samples=0`), border width 8 px for border/center RMSE
- **Seeds:** split/train seeds **42, 7, 123** (each run uses `split_seed = seed` and `seed = seed`)

Outputs per seed: `results/spatial_benchmark_seed_stability/seed_<k>/`.

## Per-seed metrics (validation RMSE)

| Seed | persistence | PlainEncoderDecoder | U-Net |
| ---: | ---: | ---: | ---: |
| 42 | 2.8467 | 2.2313 | 1.8939 |
| 7 | 2.8739 | 1.7120 | 1.5257 |
| 123 | 2.4308 | 1.8081 | 1.8436 |

(Source: `results/spatial_benchmark_seed_stability/summary.csv`; values are the recorded `rmse` field.)

**Sample count:** 18 validation days per seed (full medium validation split for each run).

## Aggregate (mean ± sample std over seeds)

| Model | RMSE mean | RMSE std | Border RMSE mean | Center RMSE mean | Border/center ratio mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.7171 | 0.2484 | 3.3803 | 2.4608 | 1.372 |
| PlainEncoderDecoder | 1.9172 | 0.2763 | 2.1517 | 1.8337 | 1.174 |
| U-Net | 1.7544 | 0.1996 | 1.9848 | 1.6716 | 1.191 |

Full aggregation (including bias, correlation, gradient–error correlation): `results/spatial_benchmark_seed_stability/aggregate_summary.csv`.

## Interpretation (cautious)

- **Average ranking:** U-Net retains the **lowest mean RMSE** across these three seeds; PlainEncoderDecoder is intermediate; persistence is highest. This is **consistent improvement in the aggregate sense**, not a proof under all possible splits.
- **Seed 123 exception:** U-Net RMSE (**1.8436**) is **slightly worse** than PlainEncoderDecoder (**1.8081**) on that split. This shows **remaining instability** and supports the view that **further validation is required** before treating U-Net as uniformly dominant.
- **Border behavior:** **Border RMSE exceeds center RMSE** for every model and seed (`border_center_rmse_ratio > 1`). The gap **narrows for U-Net relative to persistence** on average, but **border–center imbalance remains**—consistent with **remaining spatial limitations** near the domain edge, not a resolved artifact.
- **Gradient–error correlation:** small positive mean for U-Net (~0.07) vs near-zero for PlainEncoderDecoder; interpretation should stay descriptive only.

## Limitations

- Three seeds is a small Monte Carlo sample; std over seeds is not a confidence interval for generalization.
- Same architecture capacity and training budget; a weaker seed may reflect optimization noise as well as split difficulty.
- Diagnostics are regional (Georgia box) and target-only (PRISM `tmean`).

## Reproduce aggregation

After all `seed_*` directories contain `summary.json`:

```bash
python3 scripts/run_spatial_benchmark_seed_stability.py \
  --aggregate-only \
  --output-dir results/spatial_benchmark_seed_stability \
  --seeds 42 7 123
```

Full reruns (destructive to `output-dir` unless using `--no-clear-root`):

```bash
python3 scripts/run_spatial_benchmark_seed_stability.py \
  --dataset-version medium \
  --input-set core4 \
  --history-length 3 \
  --target-mode direct \
  --seeds 42 7 123 \
  --epochs 80 \
  --device cpu \
  --output-dir results/spatial_benchmark_seed_stability \
  --overwrite
```

For incremental recovery after a partial failure, use `--no-clear-root` with `--overwrite` so completed seeds are kept.
