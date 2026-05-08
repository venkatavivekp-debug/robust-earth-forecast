# Current baseline definition

This note freezes the current historical `cnn` model as a baseline for future comparisons. The name `cnn` remains in checkpoints and CLIs for compatibility, but the model is better described as `PlainEncoderDecoder` or `EncoderDecoderBaseline`.

## What the architecture is

The current baseline is a small plain spatial encoder-decoder:

1. ERA5 history is stacked as channels: `[T, C, H, W] -> [T*C, H, W]`.
2. Several same-resolution `Conv2d + ReLU` layers operate on the coarse ERA5 grid.
3. The encoded feature map is bilinearly upsampled to the PRISM target size.
4. A shallow convolutional readout predicts the PRISM field.

It is not a proper U-Net. It has no skip connections from earlier encoder features into the decoder.

## Encoder path

The encoder is three padded `3x3` convolutions:

- input channels -> `base_channels`
- `base_channels` -> `2 * base_channels`
- `2 * base_channels` -> `2 * base_channels`

The spatial size stays at the ERA5 grid resolution throughout this path. There is no multi-level downsampling hierarchy and no retained feature pyramid for later decoding.

## Bottleneck behavior

The model's "bottleneck" is not a deep latent pyramid. It is simply the final coarse-grid feature tensor before interpolation. All PRISM-scale structure must be reconstructed after this coarse representation is upsampled.

This is useful as a controlled baseline because it tests how far a shallow coarse-grid representation plus bilinear feature upsampling can go.

## Decoder / reconstruction path

The decoder receives the upsampled feature tensor at PRISM resolution, then applies:

- one padded `3x3` convolution;
- `ReLU`;
- one `1x1` output convolution.

The decoder has no direct access to earlier encoder activations. It reconstructs the target from the upsampled coarse feature state.

## No skip connections

There are no concatenations or additions from encoder layers into decoder layers. This matters because skip connections are the main mechanism that makes a U-Net a U-Net. Without them, local spatial detail can be compressed away before reconstruction.

## Why outputs can look smooth

The model upsamples coarse feature maps with bilinear interpolation and then applies a shallow readout. That encourages smooth fields unless the readout learns strong local corrections. Near regional borders, padded convolution also sees artificial context. The expected behavior is interpolation-like reconstruction with limited fine-gradient recovery.

## Expected strengths

- Compact and reproducible.
- Fast to train.
- Useful sanity check against persistence and linear baselines.
- Tests whether stacked temporal channels and coarse spatial convolutions contain enough signal.
- Provides a stable "no skip connections" baseline for U-Net comparison.

## Expected weaknesses

- Weak preservation of local gradients.
- Smooth or blurry outputs.
- Border artifacts from padding and clipped regional windows.
- No explicit residual base unless `--target-mode residual` is used.
- No terrain/static information.
- Can look better by RMSE while still missing spatial structure.

## Why keep it

This model is now valuable as a controlled baseline. A proper U-Net should beat it not only in RMSE, but also in spatial diagnostics: sharper panels, lower border/center error, better gradient behavior, and more realistic residual structure.
