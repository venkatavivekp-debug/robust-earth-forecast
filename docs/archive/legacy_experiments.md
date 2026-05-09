# Archived Experiment Context

This repository started as a broader ERA5 -> PRISM model comparison. That history is kept because it explains how the current spatial-reconstruction focus was reached, but it is no longer the active research direction.

## Preserved but de-emphasized

- ConvLSTM experiments;
- early `cnn` terminology for the no-skip encoder-decoder baseline;
- broad small/medium data-scaling summaries;
- early result-grid comparisons that rank models mostly by RMSE;
- exploratory loss checks that did not become the default objective.

## Why these are archived

The current failure mode is spatial, not temporal:

- prediction maps remain smooth;
- high-frequency detail remains weak;
- border RMSE remains above center RMSE;
- padding and decoder choices change reconstruction behavior;
- terrain-aware residual reconstruction is the strongest controlled path so far.

ConvLSTM may be revisited later, but only after the terrain-aware spatial residual baseline is understood. Adding temporal recurrence now would mix spatial reconstruction failure with temporal-context questions.

## Naming note

Historical scripts and checkpoints may still use `cnn`. In the current research framing, that model is described as `PlainEncoderDecoder`: a no-skip encoder-decoder baseline, not a proper U-Net and not the active architecture.

## Active direction

The active work is boundary-aware, terrain-aware spatial reconstruction diagnostics:

1. preserve useful evidence;
2. avoid deleting validated results;
3. keep old baselines reproducible;
4. make new experiments answer what limits fine-scale reconstruction.
