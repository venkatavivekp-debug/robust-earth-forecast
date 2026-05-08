# Spatial Sharpness Diagnosis

This diagnosis checks whether the current spatial models are recovering PRISM-scale detail or mainly matching the broad temperature field. It reuses the existing medium `core4_h3` seed-stability benchmark checkpoints and validation splits. No models were retrained.

Outputs were written locally under `results/spatial_sharpness_diagnosis/`.

## Metrics

Mean across seeds 42, 7, and 123:

| Model | RMSE | Variance Ratio | Gradient Ratio | High-Frequency Ratio | Local Contrast Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.7171 | 0.999 | 0.615 | 0.031 | 0.714 |
| PlainEncoderDecoder | 1.9172 | 0.959 | 1.359 | 2.063 | 1.065 |
| U-Net | 1.7544 | 0.973 | 0.277 | 0.0003 | 0.323 |

Ratios compare prediction statistics against the PRISM target. A value below 1 means the prediction contains less of that measured structure than PRISM.

## Reading

U-Net still has the best RMSE in this controlled comparison, but the sharpness diagnostics support the visual impression of oversmoothing. Its prediction variance is close to PRISM, yet its mean gradient magnitude is only about 28% of the target and its local contrast is about 32% of the target. The high-frequency detail energy is nearly absent with this 7-pixel local-mean diagnostic.

Persistence also misses fine structure, which is expected from upsampled coarse ERA5. PlainEncoderDecoder shows higher gradient and high-frequency energy than PRISM, but that does not mean it is better: its RMSE is worse than U-Net and the gradient-magnitude RMSE is higher, so the extra detail is not clearly target-aligned.

## Conclusion

The evidence supports the current interpretation: U-Net improves the broad spatial reconstruction but still smooths away much of the PRISM-scale detail. The remaining problem is not explained well by training time, padding, or upsampling alone.

Topography/static spatial covariates are justified as the next controlled input test. They are physically relevant to PRISM temperature structure and directly target the missing fine-scale gradients that ERA5-only inputs do not contain.

## Generated Diagnostics

- `summary.csv`
- `aggregate_summary.csv`
- `summary.json`
- `boundary_distance_error.csv`
- `gradient_comparison.png`
- `high_frequency_detail_maps.png`
- `zoomed_patch_comparison.png`
- `error_vs_boundary_distance.png`
