# U-Net transition plan

The next step is not a new model hunt. It is a controlled test of whether a proper spatial U-Net fixes failure modes that the current plain baseline cannot handle well.

## Why the current model is not a proper U-Net

The archived `CNNDownscaler` stacks the ERA5 history as channels, applies several convolution layers on the coarse ERA5 grid, bilinearly upsamples the features, and applies a shallow readout at PRISM size. It has an encoder-like feature extractor and a decoder/readout, but it does not pass encoder features into the decoder through skip connections.

That makes it closer to a plain encoder-decoder-style spatial baseline than a U-Net. Calling it a U-Net would hide the exact point Professor Hu raised: the decoder is being asked to reconstruct fine spatial detail from compressed/smoothed features.

## What skip connections preserve

Skip connections move feature maps from earlier encoder layers directly into matching decoder stages. In a downscaling problem, that matters because early layers can retain local spatial structure while deeper layers represent broader context.

For ERA5 -> PRISM, the useful split is:

- coarse atmospheric state from ERA5;
- local gradients and transitions learned from the target;
- high-resolution spatial cues reused during decoding;
- residual corrections to the upsampled latest `t2m` field.

The decoder should not have to rebuild all local structure from the bottleneck alone.

## Why U-Net should help here

A proper U-Net baseline should be tested because the current failure modes are spatial:

- border error is higher than center error;
- prediction panels look smoother than PRISM;
- fine gradients are not well explained by the current model;
- residual prediction helps, suggesting the coarse field should stay explicit while the model learns local corrections.

Skip connections are a direct architectural response to these issues. They should help the decoder keep spatial continuity, preserve local gradients, and avoid treating the output as a blurred interpolation problem.

## Expected improvements

These are hypotheses, not claims:

- sharper spatial reconstruction;
- better continuity across local gradients;
- improved edge preservation near regional borders and terrain transitions;
- less blurry output than the plain encoder-decoder baseline;
- lower border/center error ratio;
- error maps that are less dominated by the same edge artifacts.

## Expected remaining limitations

A U-Net will not solve everything:

- PRISM contains terrain/station information not explicitly present in ERA5.
- Without DEM/static fields, terrain effects can only be inferred indirectly.
- Pixelwise losses still encourage smooth averaging.
- Extreme warm/cold events may remain hard with limited samples.
- Split sensitivity will remain until there is more temporal coverage.
- A single validation split can still make one architecture look better than it really is.

## What success should actually mean

Success is not only lower RMSE. A better spatial baseline should also show:

- cleaner spatial structure in prediction panels;
- better edge preservation;
- lower boundary artifacts;
- improved gradient reconstruction;
- smaller systematic bias;
- more physically realistic output fields;
- stable behavior across split seeds.

If RMSE improves but the model still blurs gradients or keeps the same border artifact, the spatial problem is not solved.

## Controlled comparison to run later

The first controlled comparison now exists for medium `core4_h3`, direct target mode, seed 42:

| Model | RMSE | MAE | Border RMSE | Center RMSE |
| --- | ---: | ---: | ---: | ---: |
| persistence | 2.8466506862243577 | 1.942786613336184 | 3.5825521603086234 | 2.559619644749001 |
| PlainEncoderDecoder | 2.2313389357026714 | 1.7826750643192193 | 2.5037854530924526 | 2.134421349139709 |
| U-Net | 1.8938983473045878 | 1.4901164816111936 | 2.1606650140879413 | 1.7978037822151207 |

The U-Net improves RMSE and MAE over the no-skip baseline on this split. Border RMSE also drops in absolute terms, but the border/center ratio remains above 1.0. The result supports the skip-connected U-Net as the next spatial baseline, not as a finished solution.

The next comparison should repeat:

1. persistence / upsampled ERA5 baseline;
2. current plain encoder-decoder baseline;
3. proper U-Net with skip connections.

All three should use the same:

- dataset version;
- input variables;
- history length;
- normalization;
- train/validation split;
- evaluation samples;
- RMSE, MAE, bias, correlation;
- prediction panels;
- absolute error maps;
- border/center RMSE;
- gradient-vs-error analysis;
- seed-stability check.

Only after this comparison should temporal complexity be reconsidered.
