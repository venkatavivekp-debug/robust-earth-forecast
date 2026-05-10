# PixelShuffle Overfit Results

This is a fixed-sample diagnostic, not a new benchmark. It uses the same sample, residual target, reflection padding, fixed LR `1e-3`, no scheduler, seed `42`, and `1000` epochs as the corrected bilinear U-Net overfit run. The only intended change is the decoder upsampling mode.

| Decoder | RMSE @100 | RMSE @500 | RMSE @1000 | 4-8 km retention | Border RMSE | Interior RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bilinear | 0.6143 | 0.2508 | 0.1875 | 0.0129 | 0.2090 | 0.1745 |
| PixelShuffle | 0.6830 | 0.1688 | 0.1375 | 0.0642 | 0.1455 | 0.1329 |

PixelShuffle starts slower at epoch 100 but fits the sample better by epoch 500 and epoch 1000. It also raises 4-8 km PSD retention on the overfit sample from `0.0129` to `0.0642`. That is still low in absolute terms, but it is a clear improvement over the bilinear decoder.

Boundary behavior also improves on this fixed sample: border RMSE drops from `0.2090` to `0.1455`, and the border/interior ratio drops from `1.1982` to `1.0951`.

The three-panel comparison is saved at `docs/images/pixelshuffle_overfit_panel.png`. Visually, the PixelShuffle prediction has sharper local structure and less muted edge contrast than the bilinear overfit panel, but it still does not reproduce all PRISM-scale texture. This supports a narrow conclusion: learned upsampling improves reconstruction of learnable detail, but it does not create missing sub-grid information.
