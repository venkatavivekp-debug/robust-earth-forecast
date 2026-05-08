# Controlled spatial benchmark

Run:

```bash
.venv/bin/python scripts/run_spatial_benchmark.py \
  --dataset-version medium \
  --input-set core4 \
  --history-length 3 \
  --target-mode direct \
  --split-seed 42 \
  --seed 42 \
  --epochs 80 \
  --device cpu \
  --overwrite
```

This is a single controlled spatial comparison. It keeps the dataset, split, normalization, target mode, and evaluation samples fixed.

| Model | RMSE | MAE | Bias | Border RMSE | Center RMSE | Border/Center | Gradient/Error r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.8466506862243577 | 1.942786613336184 | 0.8325659299076611 | 3.5825521603086234 | 2.559619644749001 | 1.3996423912662743 | -0.04906254992828221 |
| PlainEncoderDecoder | 2.2313389357026714 | 1.7826750643192193 | -0.19393645438069573 | 2.5037854530924526 | 2.134421349139709 | 1.1730511663508323 | 0.0023686957690537635 |
| U-Net | 1.8938983473045878 | 1.4901164816111936 | 0.10235609315184656 | 2.1606650140879413 | 1.7978037822151207 | 1.2018358374047526 | 0.048938473694331665 |

![Spatial benchmark prediction panel](../images/spatial_benchmark_prediction_panel.png)

![Spatial benchmark error maps](../images/spatial_benchmark_error_maps.png)

## Read

The skip-connected U-Net improves RMSE and MAE over the no-skip `PlainEncoderDecoder` on this split. It also lowers absolute border RMSE, but the border/center ratio stays above 1.0. That means the U-Net is a better spatial baseline here, not a complete fix for edge behavior.

The gradient/error relation remains weak. The model is learning useful corrections beyond upsampled ERA5, but this run does not prove that terrain-gradient structure is solved. A multi-seed repeat is still needed before making a stronger architecture claim.
