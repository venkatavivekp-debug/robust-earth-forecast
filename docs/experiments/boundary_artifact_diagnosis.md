# Boundary artifact diagnosis

## Why this was run

Professor Hu flagged two related issues in the prediction panels: outputs remain blurred, and border values look suspicious. He also asked whether padding and convolution/deconvolution parameters had been checked. This phase audits those settings and measures boundary error directly, without changing model architecture or retraining.

Inputs come from the existing multi-seed spatial benchmark:

- dataset: medium;
- input set: `core4`;
- history: 3;
- target mode: direct;
- seeds: 42, 7, 123;
- models: persistence, `PlainEncoderDecoder`, U-Net.

Script:

```bash
.venv/bin/python scripts/diagnose_boundary_artifacts.py \
  --dataset-version medium \
  --input-set core4 \
  --history-length 3 \
  --seeds 42 7 123 \
  --benchmark-root results/spatial_benchmark_seed_stability \
  --output-dir results/boundary_artifact_diagnosis \
  --device cpu
```

Outputs:

- `results/boundary_artifact_diagnosis/boundary_metrics.csv`
- `results/boundary_artifact_diagnosis/error_by_distance.csv`
- `results/boundary_artifact_diagnosis/boundary_summary.json`
- `results/boundary_artifact_diagnosis/error_vs_boundary_distance.png`
- `results/boundary_artifact_diagnosis/border_mask.png`
- `results/boundary_artifact_diagnosis/mean_abs_error_border_map.png`

## Padding and upsampling settings checked

| Model | Padding / boundary handling | Upsampling | Deconvolution |
| --- | --- | --- | --- |
| persistence | none | bilinear latest ERA5 `t2m` -> PRISM grid | no |
| PlainEncoderDecoder | zero padding in `3x3` convs | bilinear feature upsampling | no |
| U-Net | reflection padding in `3x3` conv blocks | avg-pool encoder, bilinear decoder upsampling, skip concatenation | no |
| ConvLSTM, archived | zero padding in ConvLSTM gate and readout | bilinear hidden-state upsampling plus latest-`t2m` base | no |

Training/evaluation pass the PRISM tensor shape as `target_size`; prediction/target shape mismatches raise errors. There is no final crop. PRISM rasters are clipped to ERA5 bounds and matched to a common PRISM template.

## Boundary metrics

Mean +/- sample std over seeds.

| Model | Full RMSE | Border RMSE | Center RMSE | Border/Center | Top RMSE | Bottom RMSE | Left RMSE | Right RMSE | Corner RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.7171 +/- 0.2484 | 3.3803 +/- 0.3489 | 2.4608 +/- 0.2075 | 1.372 +/- 0.030 | 2.1387 | 4.0646 | 2.0401 | 4.7354 | 4.4244 |
| PlainEncoderDecoder | 1.9172 +/- 0.2763 | 2.1517 +/- 0.3073 | 1.8337 +/- 0.2656 | 1.174 +/- 0.014 | 1.8611 | 2.4615 | 1.8123 | 2.5559 | 2.7891 |
| U-Net | 1.7544 +/- 0.1996 | 1.9848 +/- 0.1936 | 1.6716 +/- 0.2062 | 1.191 +/- 0.055 | 1.9674 | 2.3622 | 1.6218 | 1.9287 | 2.0695 |

Per-seed border/center ratios:

| Seed | persistence | PlainEncoderDecoder | U-Net |
| ---: | ---: | ---: | ---: |
| 42 | 1.3996 | 1.1731 | 1.2018 |
| 7 | 1.3771 | 1.1877 | 1.2397 |
| 123 | 1.3398 | 1.1604 | 1.1308 |

## Read

Border degradation is confirmed in this diagnostic: border RMSE is higher than center RMSE for every model and seed. That supports Professor Hu's concern that boundary behavior is a real failure mode.

U-Net reduces full-image RMSE and absolute border RMSE relative to `PlainEncoderDecoder` on average. It does not fully remove boundary degradation; the mean border/center ratio remains above 1.0, and it is slightly higher than the no-skip baseline on average.

The learned models do not show stronger border/center degradation than persistence. That matters: persistence has no convolution or padding, so the boundary pattern is not proof that zero padding alone is responsible. The evidence is more consistent with a mixture of domain clipping, missing outside-context information, coarse-to-fine interpolation, and model boundary handling.

Edge errors are asymmetric. The persistence and `PlainEncoderDecoder` runs have larger bottom/right errors than top/left errors. U-Net reduces the right-edge error substantially, but bottom-edge error remains elevated. This suggests alignment, regional clipping, and boundary context should be checked before adding topography or temporal complexity.

## What should not be concluded

- This does not prove padding is the only cause.
- This does not prove U-Net fixed the boundary issue.
- This does not justify topography yet.
- This does not justify ConvLSTM or temporal modeling yet.
- This does not replace seed stability; it uses the same three-seed benchmark as evidence.

## Next-phase decision

- If border/center RMSE ratio is consistently above 1 across models and seeds, boundary degradation is confirmed.
- If learned models show stronger border degradation than persistence, architecture/padding/upsampling may be contributing.
- If U-Net lowers full-image RMSE but border/center ratio remains high, skip connections help but do not fully solve boundary behavior.
- If edge-specific errors are asymmetric, inspect resizing/cropping/alignment.
- Only if these conditions are supported should the next phase test controlled padding/upsampling ablations.
- Topography should come after boundary handling is understood, not before.

Decision for the current run: boundary degradation is confirmed, U-Net helps absolute error, and padding/upsampling remains a plausible contributor. Because persistence also has strong border error, the next phase should be a controlled padding/upsampling/alignment ablation rather than a new model or new predictor.
