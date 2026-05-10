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

## Decoder Diagnostics Compared With Hu et al.

Hu et al.'s MWR U-Net work is a stronger weather-postprocessing setting than this repository:

- long reforecast history rather than a short regional sample;
- a forecast-correction target with a different relationship to the input fields than ERA5 -> PRISM product residuals;
- probabilistic/reliability and skill verification, including event-oriented evaluation;
- enough data to separate model behavior from sampling noise.

This repository is smaller and deterministic, so it should not claim the same kind of skill. The useful comparison is methodological: the paper treats U-Net as one part of a controlled verification system, not as a magic architecture.

The current study now has one concrete, falsifiable finding in that spirit. For the stable static-bias target, the bilinear U-Net decoder keeps only `0.0693` of target power in the 4-8 km band and `0.7515` banded 4-32 km high-frequency retention, while PixelShuffle reaches `0.9480` in the same static-bias diagnostic. Skip connections reduce border RMSE on the fixed overfit sample by about `26%` and improve high-frequency retention (`0.7834` vs `0.7491`), but they do not fully solve near-grid detail loss.

The PixelShuffle follow-up makes the decoder point more concrete. On the same fixed residual overfit sample, PixelShuffle lowers RMSE from `0.1875` to `0.1375`, increases 4-8 km PSD retention from `0.0129` to `0.0642`, and lowers border RMSE from `0.2090` to `0.1455`. Boundary gradient alignment also improves: Sobel-gradient Pearson r with PRISM is `0.9042` for PixelShuffle versus `0.7487` for the bilinear decoder.

The boundary-context issue is not fully tested yet. The ERA5 downloader now supports a one-grid-cell buffer (`--era5-buffer-deg 0.25`), but the local ERA5 data were not redownloaded. A proper buffered experiment needs the target PRISM domain to remain fixed while the ERA5 input retains outside-domain context.

That is scientifically meaningful at small data scale because it is controlled and reproducible. It does not show the final downscaler is strong. It shows where the current reconstruction path loses detail and which next ablation is justified.
