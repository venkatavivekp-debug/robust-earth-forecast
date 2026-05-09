# Boundary ablation results

This phase tested padding and decoder-upsampling choices in the existing U-Net baseline. It keeps the data, split, seed, normalization, target mode, loss, optimizer, and evaluation metrics fixed.

Command:

```bash
.venv/bin/python scripts/run_boundary_ablation.py \
  --dataset-version medium \
  --input-set core4 \
  --history-length 3 \
  --target-mode direct \
  --seed 42 \
  --split-seed 42 \
  --overwrite
```

Outputs are under `results/boundary_ablation/`.

## Configurations

| Variant | Padding | Upsampling |
| --- | --- | --- |
| `reflection_bilinear` | reflection | bilinear + conv |
| `zero_bilinear` | zero | bilinear + conv |
| `replicate_bilinear` | replicate | bilinear + conv |
| `reflection_convtranspose` | reflection | ConvTranspose2d decoder, final target-size alignment |

The optional valid-convolution crop experiment was not run. It would change the prediction/target support and would not be a clean comparison with the fixed PRISM grid used elsewhere.

## Metrics

Medium `core4_h3`, direct target, seed 42, 18 validation samples.

| Variant | RMSE | MAE | Border RMSE | Center RMSE | Border/Center | Top | Bottom | Left | Right | Corner | Variance ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reflection + bilinear | 1.8939 | 1.4901 | 2.1607 | 1.7978 | 1.202 | 2.0566 | 2.7184 | 1.6838 | 2.0949 | 2.3039 | 0.911 |
| zero + bilinear | 1.9572 | 1.6000 | 2.1242 | 1.8992 | 1.118 | 2.2752 | 2.4007 | 1.7762 | 2.0402 | 2.3017 | 0.871 |
| replicate + bilinear | 1.7995 | 1.4031 | 2.0285 | 1.7178 | 1.181 | 1.9286 | 2.6647 | 1.5102 | 1.8285 | 2.0145 | 0.934 |
| reflection + ConvTranspose2d | 1.9232 | 1.5461 | 2.0715 | 1.8720 | 1.107 | 2.0661 | 2.5640 | 1.7380 | 1.8117 | 2.0537 | 0.934 |

Expanded reconstruction diagnostics from the same checkpoints:

| Variant | Gradient ratio | HF ratio | Local contrast ratio | Border variance ratio | Center variance ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| reflection + bilinear | 0.2855 | 0.0004 | 0.3292 | 0.8948 | 0.9170 |
| zero + bilinear | 0.6457 | 0.3917 | 0.6016 | 0.8933 | 0.8616 |
| replicate + bilinear | 0.2975 | 0.0002 | 0.3430 | 0.9106 | 0.9417 |
| reflection + ConvTranspose2d | 0.4998 | 0.0586 | 0.5221 | 0.9310 | 0.9344 |

## Read

Padding affects the result, but this single split does not isolate padding as the dominant cause. Replicate padding gives the lowest full-image RMSE and lowest absolute border RMSE here, while zero padding and ConvTranspose2d reduce the border/center ratio mostly by changing both border and center errors.

Reflection padding is not clearly best in this run. It remains a reasonable default, but the ablation suggests that boundary handling is not neutral and should be carried into the next controlled test.

ConvTranspose2d changes the error profile. Its border/center ratio is lowest in this run, and right/corner errors improve relative to reflection + bilinear. It also retains more gradient/high-frequency signal than the bilinear U-Net variants. Full RMSE is worse than the replicate + bilinear result, so it does not support replacing bilinear upsampling on this evidence alone.

Zero padding has high gradient and high-frequency ratios, but that should not be read as better reconstruction by itself. The full RMSE is worse, and the extra high-frequency energy may include edge discontinuities rather than useful PRISM detail.

Blur is not resolved. Replicate padding gives the best RMSE but still has very low high-frequency retention. Decoder choices can change detail metrics, but none of these variants gives a clean combined win across RMSE, border behavior, and high-frequency reconstruction.

Border degradation persists for every variant. The bottom edge remains high across configurations, which is consistent with the earlier diagnosis that regional clipping, outside-domain context, and target-grid effects may be involved.

## What remains unresolved

- This is one seed, not a stability result.
- The ablation does not separate model boundary behavior from missing outside-region context.
- The ConvTranspose2d variant still needs final target-size alignment because the PRISM grid is not an integer multiple of all U-Net feature maps.
- No topography was used, so terrain-gradient reconstruction is still physically under-specified.

## Next step

Run the best-supported boundary variants across the same three benchmark seeds before adding topography. A reasonable small set is `replicate_bilinear`, `reflection_bilinear`, and `reflection_convtranspose`, plus the existing persistence reference. If the border/center gap stays above 1.0 across those runs, the next research step should shift toward outside-domain context and real static terrain predictors rather than more decoder variants.
