# Repository philosophy

This repository studies ERA5 -> PRISM temperature downscaling as a multi-scale spatial reconstruction problem. The task is not only to lower a scalar error score; it is to understand what PRISM-scale structure can be recovered from coarse ERA5 predictors, and where the reconstruction fails.

Architecture choices should follow the problem structure. A model is useful here only if it can be compared cleanly against persistence, interpolation, and the current no-skip encoder-decoder baseline. Adding capacity without isolating the cause of improvement makes the result hard to interpret.

Diagnostics matter as much as RMSE. Prediction panels, absolute error maps, border-vs-center checks, variance ratios, gradient/error analysis, and seed stability are part of the evaluation, not afterthoughts.

Comparisons should be controlled and reproducible:

- same aligned ERA5/PRISM samples;
- same train/validation split;
- same normalization;
- same target mode;
- same metrics and plots;
- same seeds when testing architecture changes.

The near-term goal is spatial structure preservation: sharper reconstruction, fewer boundary artifacts, better local gradients, and more physically plausible residual fields. Temporal modeling should be revisited only after the spatial baseline is stable enough to interpret.

## Research progression

- Problem framing: ERA5 -> PRISM is multi-scale spatial reconstruction, not only interpolation.
- Baseline: persistence and `PlainEncoderDecoder` establish what coarse-field carryover and no-skip decoding can do.
- Architecture: skip-connected U-Net improves reconstruction, but outputs remain smooth and boundary error remains.
- Diagnostics: seed stability, boundary artifacts, padding/upsampling, and training budget have been tested under controlled settings.
- Current conclusion: the remaining limitation likely needs physically meaningful static spatial context, especially topography, before temporal modeling is revisited.
