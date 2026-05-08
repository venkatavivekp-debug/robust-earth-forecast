# Spatial benchmark protocol

This benchmark is meant to answer one narrow question:

> Does a proper skip-connected U-Net improve ERA5 -> PRISM spatial reconstruction beyond persistence and the current no-skip `PlainEncoderDecoder` baseline?

It should not be mixed with temporal modeling, new predictors, or target-mode changes. The point is to isolate the spatial architecture change.

## Models compared

1. **Persistence / upsampled ERA5**
   - latest ERA5 `t2m` field bilinearly upsampled to the PRISM grid;
   - no learned parameters;
   - strong baseline because daily temperature has high persistence.

2. **PlainEncoderDecoder**
   - historical `cnn` alias;
   - stacks history and variables as channels;
   - convolves on the coarse ERA5 grid;
   - bilinearly upsamples features to PRISM resolution;
   - no skip connections.

3. **Skip-connected U-Net**
   - same input convention as `PlainEncoderDecoder`;
   - encoder, bottleneck, decoder;
   - decoder reuses encoder feature maps through skip concatenations;
   - still a spatial baseline, not a temporal model.

## Fixed conditions

The comparison must use the same:

- dataset version;
- ERA5/PRISM alignment;
- input variables;
- history length;
- train/validation split;
- seed;
- target mode;
- input normalization computed on the training split only;
- PRISM target resolution;
- evaluation indices;
- metrics and plot settings.

The first controlled run uses `medium`, `core4`, `history=3`, `split_seed=42`, `seed=42`, and `target_mode=direct`.

## Metrics

Report at least:

- RMSE;
- MAE;
- bias;
- correlation;
- prediction/target variance ratio;
- border RMSE;
- center RMSE;
- border/center RMSE ratio.

## Diagnostics

Keep the visual diagnostics with the scalar metrics:

- prediction-vs-target panels;
- absolute error maps;
- border-vs-center error;
- gradient-vs-error relation;
- prediction variance / oversmoothing check;
- seed stability before making a stronger claim.

## What would count as real improvement?

Lower RMSE alone is not enough. A useful spatial baseline should show at least some of:

- lower or comparable RMSE/MAE;
- reduced bias;
- better edge and gradient preservation;
- less oversmoothing;
- better border behavior;
- more realistic prediction-vs-target panels;
- stable behavior across seeds.

If U-Net lowers RMSE but keeps the same blur and border error pattern, the spatial problem is still not solved.

## Why ConvLSTM is postponed

ConvLSTM is useful only after the spatial reconstruction problem is understood. If the no-skip baseline loses spatial detail before decoding, adding temporal recurrence can hide the real failure mode. The controlled sequence is:

1. persistence / interpolation check;
2. no-skip spatial encoder-decoder;
3. skip-connected spatial U-Net;
4. only then revisit temporal models under the same diagnostics.
