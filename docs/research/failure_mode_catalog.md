# Failure mode catalog

Use this as a checklist when comparing persistence, `PlainEncoderDecoder`, and the skip-connected U-Net. Most failures should be diagnosed with both scalar metrics and maps.

## Oversmoothing

- **Visual symptoms:** prediction field looks too smooth; local PRISM structure is muted; variance is low.
- **Likely architectural causes:** shallow decoder, bilinear upsampling of coarse features, pixelwise loss encouraging averages, missing skip connections.
- **Likely data causes:** limited samples, noisy residual signal, missing terrain/static predictors.
- **Diagnostics:** prediction-vs-target panel, prediction variance / target variance, absolute error map, gradient-vs-error analysis.

## Blurry terrain transitions

- **Visual symptoms:** gradients near elevation or regional transitions are washed out.
- **Likely architectural causes:** compressed coarse representation, weak high-resolution decoder, no feature reuse from early layers.
- **Likely data causes:** no DEM/elevation channel, ERA5 grid too coarse to identify local terrain effects.
- **Diagnostics:** target-gradient maps, gradient-vs-error scatter, local visual panels, terrain-stratified checks once DEM exists.

## Checkerboard / reconstruction artifacts

- **Visual symptoms:** repeated grid-like texture or alternating high/low artifacts in predictions.
- **Likely architectural causes:** transposed convolution, uneven upsampling kernels, decoder instability.
- **Likely data causes:** usually less data-driven than architecture-driven, but sparse samples can amplify unstable artifacts.
- **Diagnostics:** prediction panels, residual maps, spectral/texture inspection if artifacts appear.

Current note: the existing models use bilinear interpolation, not `ConvTranspose2d`, so checkerboard artifacts are not the leading diagnosis right now.

## Interpolation-like outputs

- **Visual symptoms:** learned prediction barely differs from upsampled ERA5; local corrections are weak.
- **Likely architectural causes:** residual too constrained, decoder underfits, excessive smoothness from interpolation/readout.
- **Likely data causes:** persistence is genuinely strong, limited PRISM samples, missing local predictors.
- **Diagnostics:** compare prediction against persistence/upsampled ERA5, residual maps, variance ratio, RMSE delta vs persistence.

## Center-vs-boundary degradation

- **Visual symptoms:** border values look too low/high; errors concentrate near clipped regional edges.
- **Likely architectural causes:** zero padding, regional clipping, weak decoder near boundaries, missing surrounding context.
- **Likely data causes:** target grid clipped to domain, edge pixels have less spatial context, PRISM/ERA5 alignment differences.
- **Diagnostics:** border RMSE, center RMSE, border/center ratio, border prediction mean/min/max, border mask panel.

Benchmark note: in the controlled medium `core4_h3` run, U-Net reduced absolute border RMSE from `2.5038` to `2.1607` relative to `PlainEncoderDecoder`, but border RMSE stayed above center RMSE (`1.7978`). The edge issue is reduced, not removed.

## Instability across runs

- **Visual symptoms:** same configuration gives different quality across seeds.
- **Likely architectural causes:** parameter count high relative to sample size, optimization sensitivity, early stopping noise.
- **Likely data causes:** small validation set, limited temporal coverage, nonrepresentative split.
- **Diagnostics:** multi-seed RMSE mean/std, best/worst seed, rank stability, shared visual sample across seeds.

## Split sensitivity

- **Visual symptoms:** model ranking changes when validation dates change.
- **Likely architectural causes:** model overfits date-specific residual patterns.
- **Likely data causes:** short time range, unusual weather days in one split, seasonal coverage too narrow.
- **Diagnostics:** split-seed grid, date-level error table, distribution of target/persistence RMSE per split.

## Poor extreme reconstruction

- **Visual symptoms:** warm/cold extremes are damped; high-error maps align with extreme days or local maxima/minima.
- **Likely architectural causes:** MSE/L1 losses favor central tendency, decoder smooths peaks, no uncertainty modeling.
- **Likely data causes:** rare extremes, short archive, missing terrain/static controls.
- **Diagnostics:** error by target quantile, max/min prediction comparison, event-day panels, bias by temperature regime.

## Current benchmark read

The controlled spatial benchmark is useful mainly because it separates the spatial architecture question from temporal recurrence. U-Net improves the direct spatial baseline on one split, while gradient/error correlation remains weak (`r ~= 0.049` for U-Net). That leaves terrain/static predictors, split stability, and gradient-aware diagnostics as open work.
