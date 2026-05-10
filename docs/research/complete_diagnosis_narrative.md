# Complete Diagnosis Narrative

The central question is no longer whether a larger model can lower RMSE. The question is what limits fine-scale spatial reconstruction in terrain-aware ERA5 -> PRISM downscaling.

ERA5 gives the broad atmospheric state on a coarse grid. PRISM is a different data product: station-informed, terrain-aware, and much sharper at regional scales. The target is therefore not simple super-resolution. The model is being asked to infer PRISM-scale structure that is only partly implied by ERA5 and static topography.

The physics baselines set the first boundary. A fixed lapse-rate correction barely improves bilinear ERA5 (`3.4525` vs `3.4580` RMSE), and linear terrain features explain only `5.4%` of the daily `PRISM - ERA5_bilinear` residual variance. That means most of the daily residual is not a clean elevation/slope/aspect correction. The information ceiling is real.

The next diagnostic checked whether the residual U-Net was simply collapsing to zero residual. It was not. The predicted/target residual magnitude ratio was `0.8938`, and the mean predicted residual map correlated strongly with the mean target residual map (`r = 0.9632`). The model learns a stable spatial residual pattern. The visual similarity to bilinear ERA5 comes from the residual being smooth relative to the full temperature field, not from a dead residual head.

The static-bias test made that clearer. When the target is changed from daily residual to the temporal mean residual map, the same U-Net learns the spatial pattern well (`0.2343` RMSE, `0.9861` correlation). But the high-pass/local-detail retention is only `0.2050`. In plain terms: the model can learn the broad static residual, but it still drops most fine spatial detail.

This phase located the drop more directly. The bilinear decoder retained only `0.0693` of target power in the 4-8 km band, `0.7503` in the 8-16 km band, and `0.7783` in the 16-32 km band. The first failure is therefore at the near-grid scale. Intermediate decoder maps are dominated by 32 km and larger structure, especially after the final high-resolution reconstruction stage.

The upsampling comparison was the cleanest controlled check. On the same stable static-bias target, bilinear upsampling reached `0.2343` RMSE and `0.7515` banded 4-32 km retention. ConvTranspose2d improved that to `0.1441` RMSE and `0.9148` retention. PixelShuffle improved it further to `0.0781` RMSE and `0.9480` retention. This does not make PixelShuffle the new production model. It does show that the low-pass behavior is not inevitable; it is strongly tied to the reconstruction pathway.

Skip features were not empty. The encoder maps are spatially structured, and skip tensors retain substantial 8-32 km spectral content after z-scored comparison. They are weaker in the 4-8 km band, which is expected because the skip tensors originate on the ERA5-resolution input grid, not the PRISM grid. So the skip path helps preserve coarse spatial layout, but it cannot directly carry native 4 km PRISM detail into the decoder.

## Decoder Reconstruction Bottleneck: Quantified and Addressed

The near-grid-scale number is now the clearest diagnostic. With the bilinear decoder, 4-8 km retention is `0.0693` on the static-bias target. PixelShuffle raises the same static-bias 4-8 km retention to `0.5397`. On the fixed daily residual overfit sample, PixelShuffle also improves 4-8 km retention from `0.0129` to `0.0642`, lowers RMSE from `0.1875` to `0.1375`, and reduces border RMSE from `0.2090` to `0.1455`.

This does not mean the problem is solved. ERA5-resolution skip features cannot carry PRISM-native 4-8 km information because ERA5 has no sub-grid atmospheric signal at that scale. The skips carry useful 8-32 km spatial structure and help preserve coarse layout, but the decoder still has to synthesize the 4 km field.

The boundary context issue is separate but related. If ERA5 is downloaded exactly to the PRISM target boundary, encoder convolutions at the edge operate without real atmospheric neighbors outside the box. That degrades boundary features before the skip path sees them. A buffered ERA5 download flag has been added, but the local data were not redownloaded in this pass.

PixelShuffle changed the reconstruction pathway. It did not change the input information, residual target, training split, or loss. The controlled overfit result says learned sub-pixel upsampling can reconstruct more of the learnable detail than bilinear interpolation. It does not create terrain information that ERA5/topography never contained.

The honest remaining limits are unchanged:

- the ERA5 sub-grid information deficit cannot be fixed by upsampling alone;
- the `5.4%` terrain-linear residual ceiling still constrains daily predictability;
- PixelShuffle improves learnable reconstruction detail but does not make the daily residual fully predictable;
- extending data into seasons with stronger terrain signal remains the main path to meaningful scientific improvement.

The answer to the research question is now more specific:

> Fine-scale reconstruction is limited by both input information and decoder reconstruction. ERA5/topography do not determine most daily PRISM residual variance, but even for the stable learnable residual map, the current bilinear decoder suppresses near-grid detail.

That is why loss changes and longer training only helped mildly. The target contains low-SNR daily variation, and the reconstruction path is low-pass. The encoder is not obviously dead, skip connections help, and residual mode is not simply returning zero.

The next controlled model-side test, if one is run, should be narrow: replace only the upsampling pathway under the same U-Net, split, target, normalization, and diagnostics. PixelShuffle is the strongest candidate from this diagnostic because it improved static-bias RMSE and high-frequency retention without changing the broader architecture family. Multi-scale supervision could also be tested later, but only after the upsampling result is checked on daily validation, not just the static target.

The data limitation remains. The current runs cover a small regional sample, mostly winter. January residuals appear weakly terrain-predictable. A summer or shoulder-season extension should be evaluated before making broad claims about terrain-aware downscaling. If terrain R2 increases in that season, decoder improvements become more consequential. If it does not, the dominant limit is still the ERA5/PRISM information gap.
