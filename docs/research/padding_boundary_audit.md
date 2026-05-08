# Padding and boundary audit

## Why this audit exists

Professor Hu pointed out that the predictions still look blurred and that border values appear low. He specifically asked whether padding is being used and whether convolution/deconvolution settings had been checked. This audit records the architecture settings before any padding, upsampling, residual, topography, or temporal changes are tested.

This is a diagnosis step, not a model-improvement step.

## Architecture findings

| Model | Convolution settings | Padding | Upsampling / decoding | Deconvolution | Output sizing |
| --- | --- | --- | --- | --- | --- |
| `PlainEncoderDecoder` / historical `cnn` | four `3x3` convs plus final `1x1`; stride defaults to 1 | `padding=1` in all `3x3` convs, so zero padding | coarse-grid features are bilinearly interpolated to PRISM size, then decoded by shallow conv readout | none | `target_size=(y.shape[-2], y.shape[-1])` during training/eval |
| U-Net | repeated two-conv blocks; `3x3` convs, stride 1 | `ReflectionPad2d(1)` before each `3x3` conv | avg-pool encoder, bilinear decoder upsampling, skip concatenations, final PRISM-size interpolation | none | exact target size is passed from PRISM tensor |
| ConvLSTM | gate conv uses `kernel_size=3`, `padding=1`; readout has one `3x3` conv plus `1x1` | zero padding in recurrent gate and readout conv | hidden state bilinearly interpolated to PRISM size; for one-channel output it adds an upsampled latest-`t2m` base | none | exact target size is passed from PRISM tensor |
| Persistence / upsampled ERA5 | no conv | none | latest ERA5 `t2m` bilinearly interpolated to PRISM size | none | exact target size is passed from PRISM tensor |

No model uses `ConvTranspose2d`. The checkerboard-style deconvolution failure mode is therefore not the leading explanation for the current artifacts.

## Target grid and resizing

Training and evaluation call each model with the PRISM tensor shape as `target_size`. Shape mismatches raise an error. There is no explicit output crop after prediction.

The dataset clips PRISM rasters to the ERA5 bounds and uses `rio.reproject_match` so later PRISM rasters match the first PRISM template. That gives consistent tensors, but the regional border is still a clipped physical/statistical boundary: edge pixels have less surrounding spatial context than interior pixels.

## Likely boundary-risk points

- `PlainEncoderDecoder` uses zero padding at every `3x3` convolution. Near image borders, that injects artificial context.
- ConvLSTM also uses zero padding in the recurrent gate and readout, though it has a residual-style upsampled latest-`t2m` base.
- U-Net uses reflection padding, which is safer than zero padding, but still depends on reflected in-domain values rather than real context outside the Georgia crop.
- All models and persistence use bilinear interpolation with `align_corners=False`. This is consistent, but it can still smooth coarse-to-fine transitions.
- The output field is reconstructed directly at PRISM size. No static topography, land/water mask, or surrounding-area context is available.

## What the audit can and cannot conclude

The code audit supports Professor Hu's concern that boundary behavior is worth testing. It shows real padding choices, exact target resizing, and no deconvolution.

It does not prove that padding alone causes the border artifacts. Persistence has no convolution but still shows high border error, so domain clipping, ERA5/PRISM alignment, missing surrounding context, and the coarse-to-fine reconstruction problem may also contribute.

## Evidence needed for the next phase

A controlled padding/upsampling ablation is justified only if the diagnostic results show:

- border RMSE is consistently higher than center RMSE across models and seeds;
- edge-specific errors are asymmetric enough to motivate alignment/boundary checks;
- U-Net lowers absolute error but does not remove border degradation;
- learned models do not simply inherit all boundary behavior from persistence.

If those conditions hold, the next phase should test padding/upsampling choices under the same dataset, split, normalization, and evaluation diagnostics.
