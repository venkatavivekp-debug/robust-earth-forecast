# Skip Feature Quality Findings

This diagnostic asks whether the U-Net skip tensors carry fine-scale content
toward the decoder. The maps are z-scored before PSD comparison, so the
numbers describe spectral distribution rather than physical temperature units.

- Encoder skip features classified as high-HF: `yes`
- Interpretation: skip features retain substantial near-grid spectral content

| Map | spatial std | 4-8km retention | 8-16km retention | 16-32km retention |
| --- | ---: | ---: | ---: | ---: |
| skip_enc1 | 0.1065 | 0.0974 | 1.0773 | 1.0187 |
| skip_enc2 | 0.1207 | 0.3103 | 1.5681 | 1.4781 |

PSD figure: `docs/images/skip_feature_psd.png`.

Because the model input is ERA5 resolution, the skip tensors are not
PRISM-resolution skip connections. They preserve coarse spatial layout and
terrain-conditioned features, but they cannot directly carry native 4 km
PRISM detail into the final decoder.
