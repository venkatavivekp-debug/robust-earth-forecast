# Robust Earth Forecast

ERA5 -> PRISM daily temperature downscaling over Georgia. The working question is simple:

> Can a learned model recover PRISM-scale spatial structure that is missing from upsampled ERA5?

This is treated as a multi-scale spatial reconstruction problem, not a model leaderboard. Persistence and interpolation are strong because ERA5 already carries the broad daily temperature field. The hard part is local PRISM structure: terrain-driven gradients, clipped-domain borders, valleys/ridges, and other fine spatial detail that coarse ERA5 does not directly encode.

## Current Status

The repository now centers on controlled spatial evidence:

- `PlainEncoderDecoder` is the no-skip encoder-decoder baseline (`cnn` remains only as a historical alias).
- A skip-connected U-Net improves over `PlainEncoderDecoder` on average, but not on every split.
- Border RMSE remains higher than center RMSE for persistence, `PlainEncoderDecoder`, and U-Net.
- Padding/upsampling ablations changed the error profile but did not remove blur or boundary degradation.
- A fixed-budget undertraining check found only mild improvement at 300 epochs.
- Spatial sharpness diagnostics show U-Net lowers RMSE while still losing much of the PRISM-scale gradient/detail structure.
- Temporal modeling and ConvLSTM are archived, not the next active direction.

The current grounded phase is static spatial context: real DEM-derived terrain channels under the same controlled U-Net evaluation.

## Result Snapshot

Archived full-grid result:

| Dataset | Best archived config | Best RMSE | Persistence RMSE |
| --- | --- | ---: | ---: |
| small | ConvLSTM `core4_h3` | 1.5704300999641418 | 2.355815142393112 |
| medium | ConvLSTM `core4_h3` | 1.581842489540577 | 2.966560184955597 |

Current controlled spatial benchmark, medium `core4_h3`, direct target, seed 42:

| Model | RMSE | MAE | Border RMSE | Center RMSE |
| --- | ---: | ---: | ---: | ---: |
| persistence | 2.8467 | 1.9428 | 3.5826 | 2.5596 |
| PlainEncoderDecoder | 2.2313 | 1.7827 | 2.5038 | 2.1344 |
| U-Net | 1.8939 | 1.4901 | 2.1607 | 1.7978 |

![Model comparison across baselines and learned models](docs/images/model_comparison.png)

![Controlled spatial benchmark prediction panel](docs/images/spatial_benchmark_prediction_panel.png)

![Controlled spatial benchmark error maps](docs/images/spatial_benchmark_error_maps.png)

Full write-ups:

- [`docs/experiments/spatial_benchmark.md`](docs/experiments/spatial_benchmark.md)
- [`docs/experiments/spatial_benchmark_seed_stability.md`](docs/experiments/spatial_benchmark_seed_stability.md)
- [`docs/experiments/boundary_artifact_diagnosis.md`](docs/experiments/boundary_artifact_diagnosis.md)
- [`docs/experiments/boundary_ablation_results.md`](docs/experiments/boundary_ablation_results.md)
- [`docs/experiments/undertraining_diagnosis_results.md`](docs/experiments/undertraining_diagnosis_results.md)
- [`docs/experiments/spatial_sharpness_diagnosis.md`](docs/experiments/spatial_sharpness_diagnosis.md)
- [`docs/experiments/topography_context_results.md`](docs/experiments/topography_context_results.md)
- [`docs/experiments/topography_seed_stability.md`](docs/experiments/topography_seed_stability.md)
- [`docs/experiments/topography_residual_stability.md`](docs/experiments/topography_residual_stability.md)
- [`docs/experiments/detail_preserving_loss_results.md`](docs/experiments/detail_preserving_loss_results.md)

## Research Progression

1. **Problem framing:** ERA5 -> PRISM is not just interpolation. PRISM contains local terrain-aware structure that coarse ERA5 cannot directly resolve.
2. **Baselines:** persistence and `PlainEncoderDecoder` establish what large-scale carryover and no-skip decoding can do.
3. **Architecture:** U-Net skip connections improve spatial reconstruction, but outputs remain smooth.
4. **Diagnostics:** seed stability, boundary artifacts, padding/upsampling, and undertraining were tested.
5. **Conclusion:** the remaining limitation likely needs physically meaningful static spatial context before temporal modeling.

## Why Topography Next

Professor Hu suggested topography after the blurred-output and boundary checks. That is physically plausible: elevation, slope, aspect, and terrain gradients affect local temperature and are part of what PRISM is designed to represent. The current ERA5 variables (`t2m`, `u10`, `v10`, `sp`) do not provide PRISM-scale terrain structure.

The sharpness diagnosis supports this direction: U-Net improves RMSE, but its gradient magnitude and local contrast remain well below PRISM. That points to missing fine-scale spatial information, not just an undertrained decoder.

The first controlled topography run uses a real USGS 3DEP Elevation ImageServer export aligned to the PRISM grid. On medium seed 42, U-Net `core4 + elevation` improved RMSE from **1.7995** to **1.5146**. Across seeds 42, 7, and 123, both terrain variants beat the no-topography U-Net; full topo had the lowest mean RMSE (**1.4481 +/- 0.1428**). High-frequency detail is still weak, so the result supports terrain context without solving blur.

A seed-42 residual topo check improved RMSE from **1.5886** to **1.3791** and recovered more gradient/detail signal, but it also kept border degradation. That should be repeated across seeds before it becomes the next main result.

That repeat is now done. Across seeds 42, 7, and 123, residual topo improves mean RMSE from **1.4481 +/- 0.1428** to **1.3858 +/- 0.0564**, raises gradient ratio from **0.3509** to **0.5665**, and raises high-frequency ratio from **0.0005** to **0.0302**. Seed 7 is a small RMSE regression, and border/center ratio remains above 1.0.

A detail-preserving loss check tested MSE, MSE+L1, MSE+gradient, and MSE+L1+gradient for the same residual topo setup. Gradient terms slightly improved detail ratios, but the previous MSE+L1 objective still had the best mean RMSE (**1.3858 +/- 0.0564**). The objective is not changed by default.

The comparison is:

1. persistence / upsampled ERA5;
2. U-Net `core4_h3`;
3. U-Net `core4_h3 + topography`;

with the same split, normalization, target mode, metrics, diagnostics, and seed protocol.

Plan: [`docs/research/topography_context_plan.md`](docs/research/topography_context_plan.md).

## Data and Models

- **Predictors:** ERA5 over Georgia, mainly `t2m` and `core4` (`t2m`, `u10`, `v10`, `sp`), with optional DEM-derived `core4_elev` / `core4_topo` static context.
- **Target:** PRISM daily mean temperature (`tmean`).
- **Histories:** 1, 3, and 6 days in archived grids; current spatial work uses `core4_h3`.
- **Baselines:** persistence, upsampled ERA5, linear baseline, `PlainEncoderDecoder`, U-Net.
- **Archived temporal model:** ConvLSTM remains for evidence, but temporal modeling is postponed.

Default small data is January 2023. Medium data is 2023-01-01 through 2023-03-31.

## Repository Structure

- `data_pipeline/`: ERA5 and PRISM download/validation entry points.
- `datasets/`: ERA5/PRISM alignment and dataset path resolution.
- `models/`: architecture implementations and baselines.
- `training/`: trainer CLI and checkpoint writing.
- `evaluation/`: metrics, plots, and baseline evaluation.
- `scripts/`: experiment runners and diagnostics.
- `docs/experiments/`: result summaries and diagnosis notes.
- `docs/research/`: problem framing, protocols, and next-step plans.
- `notebooks/`: companion analysis notebook.

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

Medium checks:

```bash
python3 scripts/validate_results.py --dataset-version medium
python3 scripts/summarize_results.py --dataset-version medium
```

Current diagnostic runners:

```bash
python3 scripts/run_spatial_benchmark.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/run_boundary_ablation.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/run_undertraining_diagnosis.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/diagnose_spatial_sharpness.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/download_dem_data.py --bbox=-85,30,-80,35 --source usgs_3dep_image_service
python3 scripts/prepare_topography_context.py --dem-path data_raw/static/source_dem/<dem>.tif --dataset-version medium
python3 scripts/run_topography_experiment.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc
python3 scripts/run_topography_seed_stability.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc --seeds 42 7 123
python3 scripts/run_topography_residual_stability.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc --seeds 42 7 123
python3 scripts/run_detail_preserving_loss.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc --seeds 42 7 123
```

## Limitations

- Short regional sample, especially in the default small run.
- Georgia-only bounding box; no transfer claim.
- PRISM and ERA5 differ in grid, observation influence, and physical representation.
- DEM-derived terrain context is Georgia-only and still early.
- U-Net improves reconstruction but remains smooth, especially in high-frequency detail.
- Boundary behavior remains unresolved.
- Residual topography improves mean metrics but still loses most fine-scale PRISM detail.
- Gradient-aware loss gives only a small detail signal and is not the new default.
- RMSE rankings are split-sensitive; seed tables matter.

## Notes

- Notebook: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)
- Repository philosophy: [`docs/research/repository_philosophy.md`](docs/research/repository_philosophy.md)
- Simplification plan: [`docs/research/repo_simplification_plan.md`](docs/research/repo_simplification_plan.md)
- Next phase reasoning: [`docs/research/next_phase_reasoning.md`](docs/research/next_phase_reasoning.md)
- Detail-loss reasoning: [`docs/research/detail_preserving_loss_reasoning.md`](docs/research/detail_preserving_loss_reasoning.md)
- Topography plan: [`docs/research/topography_context_plan.md`](docs/research/topography_context_plan.md)
- Spatial benchmark protocol: [`docs/research/spatial_benchmark_protocol.md`](docs/research/spatial_benchmark_protocol.md)
- Failure mode catalog: [`docs/research/failure_mode_catalog.md`](docs/research/failure_mode_catalog.md)
