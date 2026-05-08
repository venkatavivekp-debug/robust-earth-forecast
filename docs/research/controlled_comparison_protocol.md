# Controlled comparison protocol

Future architecture comparisons should isolate one change at a time. If the dataset, split, normalization, loss, target mode, and model architecture all change together, the result is not interpretable.

The clean spatial comparison is:

1. persistence / upsampled ERA5;
2. `PlainEncoderDecoder` (`cnn` alias, no skip connections);
3. proper U-Net with skip connections.

Do not add temporal complexity to this comparison.

## Fixed experimental conditions

Use identical settings for every learned spatial model:

- same dataset version and raw files;
- same input variables;
- same history length;
- same train/validation split;
- same split seed and training seed;
- same input normalization procedure;
- same target scaling and target grid;
- same target mode (`direct` or `residual`, but not mixed within one comparison);
- same optimizer, learning rate schedule, patience, and epoch budget unless explicitly testing optimization;
- same evaluation samples;
- same metrics and plots.

The persistence, upsampled ERA5, and linear baselines should be evaluated on the same validation dates.

## Why this matters

Changing multiple variables at once makes attribution weak. If U-Net improves while the split, target mode, and input set also changed, the improvement could come from any of those. A controlled comparison should answer one question:

Does adding real skip connections improve spatial reconstruction relative to the current plain encoder-decoder baseline?

## Required metrics

Keep the existing scalar metrics:

- RMSE;
- MAE;
- bias;
- correlation;
- prediction variance;
- target variance;
- variance ratio.

These are necessary but not sufficient.

## Required diagnostics

A model should not only be evaluated by RMSE. Also check:

- spatial sharpness in prediction panels;
- edge preservation;
- terrain-gradient reconstruction;
- border artifacts;
- oversmoothing / low variance;
- absolute error maps;
- gradient-vs-error behavior;
- center-vs-boundary degradation;
- stability across seeds;
- prediction realism compared with PRISM and upsampled ERA5.

## Evaluation consistency audit

Current code checks:

- Metrics are computed through the same evaluation loop for learned models and baselines in `evaluation/evaluate_model.py`.
- Learned checkpoints store train/validation split metadata; evaluation uses the stored split when available.
- Input normalization is computed from training indices in `training/train_downscaler.py`, then stored in the checkpoint and reused in evaluation. This avoids validation-stat leakage for learned models.
- The linear baseline is fitted on the training split used for evaluation.
- Persistence and upsampled ERA5 use the same latest `t2m` frame and target grid as learned models.
- Target tensors come from the same `ERA5_PRISM_Dataset`, so target scaling is shared across models.
- Prediction/target shape mismatches raise an error.
- Comparison plots use shared color ranges across ERA5, prediction, and target inside each saved panel.

Current risks:

- Historical checkpoints still use the CLI alias `cnn`; new commands can use `plain_encoder_decoder` for the same no-skip baseline.
- `--target-mode residual` changes the learning problem. Direct and residual runs should not be mixed as if they were only architecture changes.
- ConvLSTM already adds an upsampled latest-`t2m` base internally, so generic residual mode is not comparable for ConvLSTM.
- `num_samples` can restrict evaluation to the first validation samples. For architecture comparisons, use the same full validation set or document the subset.
- The first spatial benchmark summary is committed as `docs/experiments/spatial_benchmark_summary.json`, but it is still a single split.
- Plot ranges are consistent within each panel, but separate panels may not share identical ranges unless explicitly generated together.

## Minimum report for the next comparison

For each model:

- RMSE / MAE / bias / correlation;
- border RMSE / center RMSE / ratio;
- prediction variance / target variance / variance ratio;
- one shared model-comparison panel;
- absolute error map;
- gradient-vs-error analysis;
- three-seed stability summary if compute allows.

If a model wins RMSE but still blurs edges or worsens border artifacts, report that directly.
