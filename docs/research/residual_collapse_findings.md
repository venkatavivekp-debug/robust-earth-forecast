# Residual Collapse Findings

This diagnostic checks whether the trained residual U-Net is learning a meaningful
`PRISM - ERA5_bilinear` correction or mostly returning the bilinear ERA5 baseline.

- Checkpoint: `results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt`
- Validation samples: `18`
- Mean |predicted residual|: `1.7364` deg C
- Mean |target residual|: `1.9428` deg C
- Residual magnitude ratio: `0.8938`
- Pearson r between mean predicted and target residual maps: `0.9632`
- Collapse rule (`ratio < 0.2`): `not confirmed`

The residual branch is not fully collapsed by the 0.2 magnitude-ratio rule.

The image compares the mean predicted residual map with the mean target residual map:
`docs/images/residual_collapse_diagnosis.png`.

This does not prove that the model is incorrectly implemented. If the daily residual
is weakly predictable from ERA5/topography, MSE training should move the predicted
residual toward its conditional mean, which can be close to zero.
