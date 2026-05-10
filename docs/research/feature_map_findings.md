# Feature Map Findings

This diagnostic visualizes mean and channel-spread activation maps from a trained
skip-connected U-Net on one validation sample.

- Checkpoint: `results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt`
- Sample index: `0`
- Encoder spatial structure: `yes`

The encoder activations are spatially structured, so the network is not ignoring spatial inputs.

Layer summaries:

| Layer | mean-map spatial std | mean-map range | mean channel std |
| --- | ---: | ---: | ---: |
| enc1 | 0.108797 | 0.556467 | 0.191942 |
| enc2 | 0.124566 | 0.628789 | 0.154961 |
| bottleneck | 0.029482 | 0.104471 | 0.073231 |
| decoder_first_up | 0.345607 | 1.869559 | 0.866992 |
| decoder_second_up | 0.836086 | 4.483751 | 2.094811 |
| skip_enc1 | 0.108797 | 0.556467 | 0.191942 |
| skip_enc2 | 0.124566 | 0.628789 | 0.154961 |

Panel: `docs/images/feature_map_visualization.png`.

Spatially structured activations do not mean the final residual is predictable.
They only rule out a simple failure where the encoder is producing uniform maps.
