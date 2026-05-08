# ERA5 -> PRISM problem structure

This project is a supervised downscaling problem: use coarse ERA5 atmospheric fields over Georgia to estimate the finer PRISM daily `tmean` grid. The target is not a higher-resolution ERA5 field. PRISM is observation-informed and terrain-aware, so the model is trying to learn a correction between two different products as well as a change in spatial scale.

## Not just interpolation

Bilinear interpolation can put ERA5 on the PRISM grid, but it cannot add information that was absent from the coarse field. Upsampling preserves broad synoptic temperature structure and smooths everything else. It does not know local elevation, slope exposure, cold-air pooling, land/water transitions, station-network effects, or product-specific PRISM corrections.

That is why persistence is strong but incomplete. Latest ERA5 `t2m`, upsampled to PRISM resolution, is already a good first guess for daily temperature. The learning problem is mostly the residual: what needs to be added or removed so the coarse state looks like PRISM?

## Multi-scale spatial reconstruction

ERA5 -> PRISM needs information at several scales at once:

- coarse temperature patterns from ERA5;
- mesoscale gradients from winds, pressure, and recent history;
- fine local structure tied to terrain and PRISM interpolation;
- sharp transitions at regional borders, elevation changes, and local gradients.

A model that only optimizes pixelwise RMSE can get the broad field roughly right while producing a blurred map. That can still score decently when persistence is strong, but it misses the part of the problem that makes downscaling useful.

## Why edges and details matter

Climate/downscaling outputs are used as spatial fields, not just collections of pixels. Edge definition matters when the target has terrain boundaries, local gradients, or sharp transitions. If border values collapse or gradients blur, the model may look like a smoothed interpolation rather than a reconstruction of PRISM-scale structure.

The current diagnostics already show this failure mode: border RMSE is higher than center RMSE, and visual outputs are smoother than the PRISM target. This does not prove one architectural cause, but it points away from only adding temporal complexity.

## Encoder-decoder limits

The archived `CNNDownscaler` is a plain spatial baseline: it stacks history as channels, applies convolution blocks on the ERA5 grid, upsamples features, and reads out the PRISM field. It does not carry high-resolution encoder features into the decoder with skip connections. Calling it a proper U-Net would be misleading.

That kind of plain encoder-decoder can lose fine spatial information because the decoder only sees compressed/smoothed feature maps. Near edges, padding can also create artificial context. The expected symptoms are oversmoothing, low border values, and weak recovery of sharp gradients.

## Why U-Net is the right spatial baseline

A proper U-Net is the next spatial baseline because it explicitly joins encoder features back into the decoding path. Skip connections let the decoder reuse spatial features from earlier layers instead of reconstructing everything from a bottleneck representation. In this project, the question is not "does U-Net lower one RMSE number?" It is:

- does it preserve edge definition better than the plain encoder-decoder?
- does the error map become less border-heavy?
- does it recover local PRISM gradients better?
- does residual prediction still help once skip connections are present?
- does the result stay stable across split seeds?

This matters beyond ERA5/PRISM. Climate downscaling, satellite imagery, and drone or 5D spatial data all share the same basic issue: coarse/global context and fine local structure have to be combined without washing out boundaries.

## Expected failure modes

- **Oversmoothing:** prediction variance too low; maps look interpolation-like.
- **Boundary artifacts:** border RMSE higher than center RMSE; edge means biased low/high.
- **Loss of sharp gradients:** target gradients are not matched by prediction gradients.
- **Systematic bias:** mean prediction error persists across samples or regions.
- **Interpolation-like outputs:** learned model barely differs from upsampled ERA5.
- **Split sensitivity:** rankings change across train/validation seeds.
- **Limited-data instability:** model improves one metric but fails visually or across seeds.

## Diagnostics to keep

- RMSE and MAE for basic error scale.
- Bias for systematic warm/cold offsets.
- Correlation for field-shape agreement.
- Persistence, upsampled ERA5, and linear baselines.
- Prediction-vs-target panels for visual blur and local structure.
- Absolute error maps for spatially organized failure.
- Border/center RMSE for edge artifacts.
- Gradient-vs-error analysis for sharp-transition behavior.
- Seed stability tables before treating a result as reliable.

No single metric is enough. A U-Net that wins RMSE but keeps the same border artifact has not solved the spatial reconstruction problem.

## Reference paper relevance

Hu et al. (2023), "Deep Learning Forecast Uncertainty for Precipitation over the Western United States" ([doi:10.1175/MWR-D-22-0268.1](https://doi.org/10.1175/MWR-D-22-0268.1)), used a U-Net framework for weather forecast postprocessing and uncertainty estimation over a large western U.S. precipitation dataset. Their task is not this repo's ERA5 -> PRISM temperature downscaling task, and their data scale is much larger.

The useful connection is narrower: U-Net-style spatial architectures are a reasonable baseline when the target is a structured weather/climate field, and evaluation should look beyond one deterministic RMSE number. Spatial structure, uncertainty/failure modes, benchmark comparisons, and data scale all matter.

## Next research step

The next implementation step should be a clean spatial U-Net baseline with real skip connections, compared directly against the current plain encoder-decoder baseline and persistence on the same split. If the existing `UNetDownscaler` is kept, first verify that its skip connections and naming match the intended baseline and document it as such.

The goal is not model-chasing. The goal is to test whether skip connections improve spatial reconstruction, edge definition, terrain-gradient preservation, and error-map behavior relative to the current encoder-decoder baseline.

## What not to do yet

- Do not add ConvLSTM or more temporal complexity until the spatial baseline is diagnosed.
- Do not claim success from one RMSE value.
- Do not call the current plain encoder-decoder a proper U-Net if it lacks skip connections.
- Do not remove files until `repo_cleanup_plan.md` has been reviewed.
