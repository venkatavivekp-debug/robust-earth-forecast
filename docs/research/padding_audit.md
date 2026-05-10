# Padding Audit

This audit checks `models/unet_downscaler.py` after the decoder diagnostics.

## Convolution Layers

| Layer group | Conv2d layers | Spatial kernel | Padding behavior with default U-Net |
| --- | ---: | --- | --- |
| `enc1` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `enc2` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `bottleneck` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `dec2` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `dec1` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `high_res` | 2 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |
| `out` | 1 | 1x1 | no spatial padding; does not sample outside the pixel |
| PixelShuffle expanders | 3 | 3x3 | `ReflectionPad2d(1)` before each Conv2d |

No default 3x3 spatial Conv2d uses zero padding when `padding_mode="reflection"`.

## Upsampling and Skip Order

The order is:

1. encode ERA5-resolution features;
2. average-pool into lower-resolution encoder levels;
3. upsample decoder tensor to the matching encoder feature size;
4. concatenate the skip feature;
5. run the decoder ConvBlock;
6. upsample to PRISM target size;
7. run `high_res`;
8. apply the 1x1 output projection.

This is the correct U-Net ordering for skip fusion. The skip tensor is concatenated after upsampling to the matching encoder-grid size, not before.

## Findings

Padding is consistent end-to-end for the default reflection-padding U-Net. No layer had to be changed from zero padding to reflection padding. The one caveat is that the final `out` layer is a 1x1 projection, so reflection padding is not applicable there.

The remaining boundary issue is therefore more likely from missing atmospheric context outside the domain and from the reconstruction/upsampling path, not from hidden zero padding in default 3x3 convolutions.
