# Research Summary

The working question is:

> What limits fine-scale spatial reconstruction in terrain-aware ERA5 -> PRISM
> downscaling?

This project started as an ERA5-to-PRISM daily temperature downscaling study
over Georgia. The useful result is not a final production downscaler. It is a
controlled diagnosis of why the learned maps remain too smooth.

## Problem Structure

ERA5 provides the broad atmospheric state on a coarse grid. PRISM is a different
data product: station-informed, terrain-aware, and much sharper in local
spatial structure. Downscaling here is therefore conditional reconstruction,
not image resizing. ERA5 can explain large-scale day-to-day temperature
changes, but it does not directly contain ridgeline, valley, land-water, or
station-network structure at PRISM scale.

That distinction matters because a model trained with pointwise losses can
learn the expected PRISM correction while still suppressing detail. If one
coarse ERA5 state is compatible with many possible local PRISM patterns, the
MSE-optimal prediction is smooth.

## Diagnostic Sequence

The first physics check showed that the daily residual is weakly terrain
predictable. Bilinear ERA5 has `3.4580` deg C RMSE, and a simple lapse-rate
correction only reaches `3.4525` deg C. Linear terrain features explain
`R2 = 0.0543` of the `PRISM - ERA5_bilinear` residual variance. In this January
Georgia sample, most daily residual variance is not a simple elevation/slope
correction.

The residual-collapse check ruled out the easiest failure mode. The residual
U-Net did not just predict zeros. The predicted/target residual magnitude ratio
was `0.8938`, and the mean predicted residual map had Pearson `r = 0.9632` with
the mean target residual map. The model learns the broad spatial pattern, even
when the final temperature map still looks close to bilinear ERA5.

The static-bias test separated learnable spatial structure from daily noise.
When the target was changed to the temporal mean residual map, the U-Net learned
it well: `r = 0.9861`. But high-frequency retention was only `0.2050`, meaning
the decoder lost about 80% of the fine spatial detail even on a stable target.

Decoder PSD analysis located the loss. With bilinear upsampling, 4-8 km
retention was only `0.0693`; PixelShuffle raised the same static-bias 4-8 km
retention to `0.5397`. Skip features also showed the expected limit: enc1
4-8 km content was `0.0974`, because ERA5 has no true PRISM-native sub-grid
signal to pass through the skip path.

PixelShuffle helped the fixed overfit diagnostic. On the same residual sample,
it reduced RMSE from `0.1875` to `0.1375`, raised 4-8 km PSD retention from
`0.0129` to `0.0642`, reduced border RMSE from `0.2090` to `0.1455`, and raised
Sobel-gradient correlation with PRISM from about `0.75` to `0.90`.

## Full-Training PixelShuffle Check

The full medium-data check is more conservative:

| Decoder | RMSE mean +/- std | Gradient ratio | HF ratio | Border RMSE |
| --- | ---: | ---: | ---: | ---: |
| Bilinear residual-topo reference | 1.3858 +/- 0.0564 | 0.5665 | 0.0302 | 1.5828 |
| PixelShuffle residual-topo | 1.4104 +/- 0.0958 | 0.5979 | 0.0507 | 1.5986 |

PixelShuffle transfers to detail metrics but not to validation RMSE. It keeps
more gradient, high-frequency, and local-contrast structure, but the RMSE is
slightly worse and more variable across seeds. That is an important negative
result: improving the reconstruction pathway does not by itself overcome the
daily residual information limit.

## What Remains Open

The remaining limits are now fairly specific:

- daily residual variance is mostly not explained by linear terrain geometry;
- ERA5 has no 4 km atmospheric signal for skip connections to preserve;
- the decoder can improve detail retention, but validation skill remains
  constrained by data scale and target noise;
- January may be a weak season for terrain-driven temperature structure.

## Next Meaningful Step

The next data-side step should be seasonal extension, especially summer or
shoulder-season months where terrain heating, slope/aspect effects, and local
gradients should be stronger. The point is not just more rows. The point is to
test whether the terrain-predictable residual component increases. If terrain
R2 stays near `0.054`, architecture changes will mostly reorganize a weak
signal. If terrain R2 rises, then decoder improvements such as PixelShuffle
become more scientifically consequential.
