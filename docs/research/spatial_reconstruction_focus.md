# Spatial Reconstruction Focus

The project is now organized around one question:

> What limits fine-scale spatial reconstruction in terrain-aware ERA5 -> PRISM downscaling?

The experiments so far suggest that the bottleneck is not a single missing model component. U-Net skip connections help. Topography helps. Residual target mode helps. Gradient-aware loss gives only a small detail signal. Yet high-frequency PRISM structure remains weak and border errors persist.

## Working explanation

The remaining limitation is likely a mix of:

- coarse ERA5 inputs lacking PRISM-scale terrain-conditioned information;
- decoder pathways that reconstruct smooth low-frequency fields more easily than sharp local structure;
- boundary effects from clipped regional grids;
- imperfect preservation of high-resolution features through the encoder/decoder path;
- objective functions that still reward average spatial correctness more than local detail.

## What future work must test

Future changes should be justified by a reconstruction failure mode:

- padding changes should target boundary discontinuities;
- decoder changes should target smoothing or checkerboard artifacts;
- terrain features should target missing static spatial context;
- residual formulation should target coarse-field correction rather than full-field relearning;
- evaluation changes should expose spatial detail loss, not only global RMSE.

## What remains postponed

Temporal modeling, ConvLSTM, attention, transformers, GANs, and diffusion-style models are not the next step. They may become useful later, but they do not directly answer the current reconstruction bottleneck.

## Required diagnostics

Any future reconstruction experiment should report:

- RMSE and MAE;
- bias and correlation;
- border RMSE, center RMSE, and border/center ratio;
- per-edge and corner RMSE where relevant;
- gradient ratio;
- high-frequency ratio;
- local contrast ratio;
- variance ratio;
- prediction, error, gradient, detail, and local-patch panels.

The goal is to understand whether the model is reconstructing local structure or smoothing toward a plausible low-frequency field.
