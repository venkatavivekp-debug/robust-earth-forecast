# Robust Earth Forecast

ERA5 -> PRISM daily temperature downscaling over Georgia.

The current research question is:

> What limits fine-scale spatial reconstruction in terrain-aware ERA5 -> PRISM downscaling?

This repository is now organized around that question. The active work is not model chasing. It studies why learned reconstructions remain smooth, where boundary artifacts enter, and how terrain-aware residual reconstruction behaves under controlled diagnostics.

## Problem

ERA5 carries the broad daily atmospheric state: near-surface temperature, wind, and surface pressure on a coarse grid. PRISM contains finer spatial structure informed by station observations and terrain-aware interpolation. Upsampling ERA5 is therefore a strong baseline, but it cannot create missing local structure such as ridges, valleys, sharp terrain gradients, or clipped-domain boundary behavior.

The hard part is not resizing an image. It is recovering PRISM-scale spatial detail from coarse atmospheric predictors plus static geographic context.

## Current Direction

The active phase is boundary-aware spatial reconstruction:

- diagnose blur and high-frequency detail loss;
- compare padding and decoder reconstruction behavior;
- measure edge and center errors separately;
- use real DEM-derived terrain channels;
- keep temporal modeling archived until spatial behavior is understood.

`PlainEncoderDecoder` is the no-skip encoder-decoder baseline. The old `cnn` name remains only as a historical CLI alias. ConvLSTM results are kept as archived evidence, but temporal modeling is not the active direction.

## Main Evidence

| Finding | Evidence | Read |
| --- | --- | --- |
| U-Net helps over the no-skip baseline | medium `core4_h3`, seed 42: U-Net RMSE `1.8939` vs PlainEncoderDecoder `2.2313` | skip connections matter, but do not solve blur |
| Border degradation is real | border RMSE stays above center RMSE for persistence, PlainEncoderDecoder, and U-Net | boundary behavior is not only a learned-model artifact |
| Padding/decoder choices matter | replicate padding gives best RMSE in the current ablation; ConvTranspose2d changes edge/detail behavior but worsens RMSE | reconstruction operations are part of the failure mode |
| Topography helps | full topo U-Net direct RMSE `1.4481 +/- 0.1428` across seeds | static terrain context is physically useful |
| Residual topography helps more | residual topo RMSE `1.3858 +/- 0.0564`; gradient ratio `0.5665` | learning PRISM correction over ERA5 is better grounded |
| Gradient-aware loss is not enough | best RMSE remains `mse_l1`; gradient losses only slightly improve detail ratios | objective mismatch is not the whole bottleneck |
| Training sanity is not clean | one-sample residual overfit stalls at `0.2249` RMSE after 600 epochs | debug training/reconstruction before adding more modeling |

Visual examples:

![Model comparison across baselines and learned models](docs/images/model_comparison.png)

![Controlled spatial benchmark prediction panel](docs/images/spatial_benchmark_prediction_panel.png)

![Controlled spatial benchmark error maps](docs/images/spatial_benchmark_error_maps.png)

## Boundary-Aware Evaluation

Global RMSE is reported, but it is not sufficient. Current diagnostics also track:

- border RMSE and center RMSE;
- border/center RMSE ratio;
- per-edge and corner RMSE;
- prediction/target variance ratio;
- gradient magnitude ratio;
- high-frequency detail ratio;
- local contrast ratio;
- error vs distance from boundary;
- prediction, error, gradient, high-frequency, and local-patch panels.

This matters because a model can lower RMSE while still producing overly smooth maps.

## Reconstruction Diagnostics

Controlled experiments currently cover:

- persistence / upsampled ERA5;
- no-skip `PlainEncoderDecoder`;
- proper skip-connected U-Net;
- U-Net padding modes: zero, reflection, replicate;
- decoder reconstruction: bilinear interpolation + convolution vs ConvTranspose2d;
- DEM-derived terrain context: elevation, slope, aspect, terrain-gradient magnitude;
- direct vs residual topography;
- MSE/L1/gradient-aware training losses.

The strongest current interpretation is:

> Terrain-conditioned residual U-Net improves the reconstruction, but most fine PRISM-scale detail is still missing. The remaining limitation is tied to reconstruction behavior, boundary context, decoder smoothing/artifacts, and limited spatial information, not just undertraining or loss choice.

## Training Sanity Checks

The latest debug pass does not show an obvious date/unit/residual-scaling bug, but the overfit checks are not clean. A terrain residual U-Net reaches `0.2249` RMSE on one fixed sample after 600 epochs, and the 4/8-sample subset checks also fail to memorize strongly. Doubling U-Net width helps the tiny-subset fit (`0.3660` best train RMSE) but does not resolve it.

This points back to training/reconstruction behavior before adding ConvLSTM, more inputs, or another architecture family. See [`docs/experiments/training_sanity_checks.md`](docs/experiments/training_sanity_checks.md).

## Result Docs

Core reconstruction diagnostics:

- [`docs/experiments/spatial_benchmark.md`](docs/experiments/spatial_benchmark.md)
- [`docs/experiments/spatial_benchmark_seed_stability.md`](docs/experiments/spatial_benchmark_seed_stability.md)
- [`docs/experiments/boundary_artifact_diagnosis.md`](docs/experiments/boundary_artifact_diagnosis.md)
- [`docs/experiments/boundary_ablation_results.md`](docs/experiments/boundary_ablation_results.md)
- [`docs/experiments/spatial_sharpness_diagnosis.md`](docs/experiments/spatial_sharpness_diagnosis.md)
- [`docs/experiments/topography_context_results.md`](docs/experiments/topography_context_results.md)
- [`docs/experiments/topography_seed_stability.md`](docs/experiments/topography_seed_stability.md)
- [`docs/experiments/topography_residual_stability.md`](docs/experiments/topography_residual_stability.md)
- [`docs/experiments/detail_preserving_loss_results.md`](docs/experiments/detail_preserving_loss_results.md)
- [`docs/experiments/training_sanity_checks.md`](docs/experiments/training_sanity_checks.md)

Archived or supporting context:

- [`docs/experiments/undertraining_diagnosis_results.md`](docs/experiments/undertraining_diagnosis_results.md)
- [`docs/archive/legacy_experiments.md`](docs/archive/legacy_experiments.md)

Research framing:

- [`docs/research/problem_structure.md`](docs/research/problem_structure.md)
- [`docs/research/spatial_reconstruction_focus.md`](docs/research/spatial_reconstruction_focus.md)
- [`docs/research/spatial_benchmark_protocol.md`](docs/research/spatial_benchmark_protocol.md)
- [`docs/research/current_baseline_definition.md`](docs/research/current_baseline_definition.md)
- [`docs/research/failure_mode_catalog.md`](docs/research/failure_mode_catalog.md)
- [`docs/research/topography_context_plan.md`](docs/research/topography_context_plan.md)
- [`docs/research/detail_preserving_loss_reasoning.md`](docs/research/detail_preserving_loss_reasoning.md)
- [`docs/research/training_data_sanity_audit.md`](docs/research/training_data_sanity_audit.md)
- [`docs/research/paper_alignment_notes.md`](docs/research/paper_alignment_notes.md)

## Reproduce

Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Small data:

```bash
python3 data_pipeline/download_era5_georgia.py --year 2023 --month 1 --overwrite
python3 data_pipeline/download_prism.py --start-date 20230101 --days 30 --variable tmean
```

Medium validation:

```bash
python3 scripts/validate_results.py --dataset-version small
python3 scripts/validate_results.py --dataset-version medium
```

Current reconstruction diagnostics:

```bash
python3 scripts/run_spatial_benchmark.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/run_boundary_ablation.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/diagnose_spatial_sharpness.py --dataset-version medium --input-set core4 --history-length 3
python3 scripts/download_dem_data.py --bbox=-85,30,-80,35 --source usgs_3dep_image_service
python3 scripts/prepare_topography_context.py --dem-path data_raw/static/source_dem/<dem>.tif --dataset-version medium
python3 scripts/run_topography_residual_stability.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc --seeds 42 7 123
python3 scripts/run_detail_preserving_loss.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc --seeds 42 7 123
python3 scripts/overfit_single_sample.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc
python3 scripts/overfit_small_subset.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc
python3 scripts/run_training_capacity_diagnosis.py --dataset-version medium --static-covariate-path data_processed/static/georgia_prism_topography.nc
```

## Repository Layout

- `data_pipeline/`: ERA5, PRISM, and DEM acquisition/preparation entry points.
- `datasets/`: ERA5/PRISM alignment and static covariate loading.
- `models/`: architecture implementations and baselines.
- `training/`: training CLI and checkpoint writing.
- `evaluation/`: reusable metrics/evaluation helpers.
- `scripts/`: controlled experiment runners and diagnostics.
- `docs/research/`: problem framing, protocols, and reasoning.
- `docs/experiments/`: experiment results and diagnosis notes.
- `docs/archive/`: historical or de-emphasized experiment context.
- `notebooks/`: companion analysis notebook.

## Limitations

- Short regional sample, especially in the default small run.
- Georgia-only bounding box; no transfer claim.
- PRISM and ERA5 differ in grid, observation influence, and physical representation.
- DEM-derived context is early and only tested for this regional setup.
- Boundary behavior remains unresolved.
- Residual topography improves mean metrics but still loses most fine PRISM detail.
- Gradient-aware loss gives only a small detail signal and is not the new default.
- The current U-Net does not pass a strict tiny-sample memorization check.
- Temporal modeling is intentionally postponed.

## Next Work

Stay with spatial reconstruction:

1. error by elevation, slope, and terrain-gradient bins;
2. boundary-distance diagnostics for terrain residuals;
3. decoder/skip feature diagnostics for where detail is lost;
4. data coverage checks before any temporal model is revived.
