# Recoverable Scale Analysis

ERA5 -> PRISM downscaling is not ordinary image super-resolution. ERA5 carries
the broad atmospheric state on a coarse grid. PRISM is station-informed and
terrain-aware, with structure near 4 km that is only partly implied by ERA5 and
static topography.

That means global RMSE is not enough. A model can lower RMSE while still losing
the wavelengths that make the PRISM field useful: local gradients, terrain
transitions, and boundary-region detail. The better question is which spatial
scales are recoverable from the current inputs.

The expected pattern is simple:

- large wavelengths should be easiest because ERA5 contains synoptic-scale
  structure;
- intermediate wavelengths may be helped by topography and skip connections;
- near-PRISM wavelengths are hardest because they depend on local observations,
  terrain effects, outside-domain context, and atmospheric details not fully
  present in coarse ERA5.

The previous diagnostics already point in this direction. PixelShuffle improves
4-8 km retention on controlled targets and improves gradient alignment, so the
decoder path matters. But PixelShuffle does not lower full-validation RMSE, so
decoder changes alone are not the answer. The daily residual is still weakly
terrain-predictable in the January Georgia sample, with terrain geometry
explaining about 5.4% of residual variance.

This is the scale-level version of Professor Hu's feedback: diagnose what is
missing before adding model complexity. It also explains why a later Prithvi
WxC-style direction is reasonable. Larger pretrained weather representations
may help if they encode atmospheric regimes and context that a short regional
ERA5/PRISM sample cannot learn locally. That is a later research direction, not
something this repository should claim from the current results.
