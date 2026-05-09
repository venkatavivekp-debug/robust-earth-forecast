# Paper Alignment Notes

Professor Hu's MWR U-Net weather paper is useful here less as a recipe for one architecture and more as a reminder about benchmark design. The paper uses a much stronger weather-postprocessing setting than this repository currently has: larger historical data, careful baseline comparison, reliability/skill checks, and verification beyond a single deterministic RMSE.

This project is narrower:

- regional ERA5 -> PRISM daily temperature reconstruction over Georgia;
- deterministic output, not a probabilistic forecast product;
- limited temporal coverage compared with reforecast-style training;
- active focus on spatial reconstruction failure modes, not operational forecast skill.

The important lesson is that a U-Net is not enough by itself. The data scale, target formulation, split design, verification metrics, and failure-mode diagnostics all matter. The current visual weakness should therefore be treated as a training/data/reconstruction problem first, not as a reason to add temporal complexity or a larger architecture.

## Current Fit to the Paper Direction

| Paper-aligned idea | Current repo status |
| --- | --- |
| Strong baseline comparison | Persistence, PlainEncoderDecoder, U-Net, topography, residual, boundary diagnostics |
| Spatial verification | RMSE/MAE plus gradients, high-frequency detail, border/center metrics, error maps |
| Data scale awareness | Medium split and seed stability are present, but still small for weather/climate learning |
| Reliability/extreme verification | Not yet covered; this is a limitation |
| Forecast uncertainty | Not covered; current repo is deterministic |

## Practical Takeaway

Before adding ConvLSTM or other complexity, the repository needs to prove that the terrain-conditioned U-Net can fit tiny controlled samples and preserve local PRISM structure. If it cannot, larger model families will only obscure the failure.
