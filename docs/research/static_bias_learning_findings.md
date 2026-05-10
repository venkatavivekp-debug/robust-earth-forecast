# Static Bias Learning Findings

This test removes temporal variability from the residual target. The model sees
one ERA5/topography input and learns the training-split temporal mean of
`PRISM - ERA5_bilinear`.

- Epochs: `500`
- Final RMSE: `0.2343` deg C
- Final MAE: `0.1553` deg C
- Pixelwise correlation: `0.9861`
- Target std: `1.4101` deg C
- Predicted std: `1.3906` deg C
- High-frequency retention: `0.2050`

The model can learn a stable residual map once day-to-day variability is removed.

Panel: `docs/images/static_bias_learning_result.png`.

If this test succeeds while daily residual prediction remains smooth or noisy,
the failure mode is more consistent with data/target variability than a broken decoder.
