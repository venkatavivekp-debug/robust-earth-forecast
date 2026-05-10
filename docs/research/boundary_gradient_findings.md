# Boundary Gradient Findings

This diagnostic compares Sobel gradient magnitude maps for the fixed overfit sample.
It uses saved prediction arrays from the bilinear and PixelShuffle single-sample runs.

| Method | Border grad mean | Interior grad mean | Border/interior | Pearson r vs PRISM grad |
| --- | ---: | ---: | ---: | ---: |
| PRISM target | 0.1519 | 0.1487 | 1.0210 | 1.0000 |
| ERA5 bilinear | 0.1063 | 0.0928 | 1.1458 | -0.0466 |
| U-Net bilinear | 0.1171 | 0.1171 | 1.0001 | 0.7487 |
| U-Net PixelShuffle | 0.1442 | 0.1344 | 1.0729 | 0.9042 |

PixelShuffle improves gradient-map alignment over the bilinear decoder.

Figure: `docs/images/boundary_gradient_comparison.png`.
