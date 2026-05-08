# Underperformance diagnosis

Checked `models/`, `training/`, `evaluation/`, `datasets/prism_dataset.py`, `notebooks/analysis.ipynb`, README figures, and committed experiment JSONs. No model code was changed.

## Architecture notes

- CNN decoder: `CNNDownscaler` applies same-size `Conv2d(..., padding=1)` blocks on the ERA5 grid, bilinearly upsamples feature maps to PRISM size, then applies a padded `3x3` conv and final `1x1` conv.
- ConvLSTM decoder: `ConvLSTMCell` uses `padding=kernel_size//2`; the hidden state is bilinearly upsampled, passed through a padded readout conv, and added to an upsampled latest-`t2m` base when `out_channels == 1`.
- `ConvTranspose2d` is not used.
- Output size is forced by `target_size=(y.shape[-2], y.shape[-1])`; evaluation raises on prediction/target shape mismatch.
- PRISM rasters are clipped to ERA5 bounds and later rasters are `rio.reproject_match`ed to the first PRISM template. This gives consistent tensor shape, but edge pixels still deserve attention because the target itself is a clipped regional window.

## Border diagnostic

Command:

```bash
.venv/bin/python scripts/check_spatial_artifacts.py --device cpu
.venv/bin/python scripts/check_spatial_artifacts.py --device cpu --model cnn \
  --output-json results/diagnostics/spatial_artifacts_core4_h3_cnn.json \
  --output-panel results/diagnostics/spatial_artifacts_core4_h3_cnn.png
```

Border width: 8 PRISM pixels. Validation split comes from the stored `core4_h3` checkpoints.

| Model | RMSE | MAE | Border RMSE | Center RMSE | Border/Center | Border pred mean | Center pred mean | Target border mean | Target center mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvLSTM `core4_h3` | 1.571948 | 1.196809 | 2.347021 | 1.214406 | 1.932649 | 10.274251 | 11.965181 | 11.162066 | 11.637097 |
| CNN `core4_h3` | 3.497680 | 2.731013 | 4.325209 | 3.179704 | 1.360255 | 9.357196 | 12.690310 | 11.162066 | 11.637097 |

Border artifacts are confirmed for the current checkpoints. Both learned models predict lower border means than the PRISM border mean; ConvLSTM has the larger border/center RMSE ratio, while CNN is worse overall.

## Likely source

This does not look like a deconvolution artifact because there is no transposed convolution. The more likely source is the combination of zero-padded convolutions near image edges, bilinear feature upsampling, and weak spatial decoding. CNN is especially exposed because it predicts the full PRISM field without a residual path or skip connections. ConvLSTM is better because it adds an upsampled latest-`t2m` base, but its residual readout still uses padded convs after a smooth upsample.

Undertraining may contribute for CNN, but the ConvLSTM result is not just a missing-epochs story: the saved best epoch is late in training, and the visual issue is spatially structured. Missing context and limited data remain real, but the first thing to test is the base spatial decoder.

## Next experiment plan

1. Keep the same data, split, metrics, and `core4_h3` setup.
2. Add one spatial baseline at a time: first a small U-Net-style CNN with skip connections, then a residual CNN that predicts a correction to upsampled latest `t2m`.
3. Try reflection padding or explicit valid-region scoring only as a controlled ablation; do not mix it with architecture changes.
4. Add topography/static fields after the spatial decoder is fixed enough to separate architecture artifacts from missing terrain signal.
5. Re-run this border diagnostic for every new spatial model. Only lean on ConvLSTM if history 3/6 still beats history 1 after the spatial decoder issue is addressed.
