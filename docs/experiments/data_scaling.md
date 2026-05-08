# Data scaling experiment (small vs medium)

Same Georgia bbox, ERA5/PRISM pipeline, archived `cnn`/ConvLSTM code, metrics, and `scripts/run_core_experiments.py` defaults. The only change is the date range and dataset paths. The archived `cnn` rows are the plain encoder-decoder baseline, not a skip-connected U-Net.

**Small** is the January 2023 reference overlap. **Medium** is January 1 through March 31, 2023 under `data_raw/medium/`.

## Sample counts

| Dataset | History 1 | History 3 | History 6 |
| --- | ---: | ---: | ---: |
| small | 20 | 18 | 15 |
| medium | 90 | 88 | 85 |

## RMSE comparison

Values come from `results/experiments/summary.csv`, `results/experiments_medium/summary.csv`, `docs/experiments/final_comparison.json`, and `docs/experiments/final_comparison_medium.json`. Beat flags use the matching persistence row for each dataset/history split.

| Dataset | Model | Variables | History | RMSE | Beats Persistence |
| --- | --- | --- | ---: | ---: | --- |
| small | Persistence | core4 | 3 | 2.355815142393112 | baseline |
| small | PlainEncoderDecoder (`cnn`) | t2m | 1 | 5.2338986992836 | No |
| small | PlainEncoderDecoder (`cnn`) | t2m | 3 | 4.504708349704742 | No |
| small | PlainEncoderDecoder (`cnn`) | t2m | 6 | 5.41701078414917 | No |
| small | PlainEncoderDecoder (`cnn`) | core4 | 1 | 4.496296465396881 | No |
| small | PlainEncoderDecoder (`cnn`) | core4 | 3 | 3.4474770426750183 | No |
| small | PlainEncoderDecoder (`cnn`) | core4 | 6 | 3.070897897084554 | Yes |
| small | ConvLSTM | t2m | 1 | 4.956881523132324 | No |
| small | ConvLSTM | t2m | 3 | 2.0036779940128326 | Yes |
| small | ConvLSTM | t2m | 6 | 2.991683403650919 | Yes |
| small | ConvLSTM | core4 | 1 | 4.246121346950531 | No |
| small | ConvLSTM | core4 | 3 | 1.5704300999641418 | Yes |
| small | ConvLSTM | core4 | 6 | 2.3035560051600137 | Yes |
| medium | Persistence | core4 | 3 | 2.966560184955597 | baseline |
| medium | PlainEncoderDecoder (`cnn`) | t2m | 1 | 2.3131760358810425 | Yes |
| medium | PlainEncoderDecoder (`cnn`) | t2m | 3 | 1.6414796262979507 | Yes |
| medium | PlainEncoderDecoder (`cnn`) | t2m | 6 | 1.7203279733657837 | Yes |
| medium | PlainEncoderDecoder (`cnn`) | core4 | 1 | 2.313338950276375 | Yes |
| medium | PlainEncoderDecoder (`cnn`) | core4 | 3 | 2.1001143231987953 | Yes |
| medium | PlainEncoderDecoder (`cnn`) | core4 | 6 | 2.3830468356609344 | Yes |
| medium | ConvLSTM | t2m | 1 | 2.561451241374016 | Yes |
| medium | ConvLSTM | t2m | 3 | 2.0794655084609985 | Yes |
| medium | ConvLSTM | t2m | 6 | 1.6947292536497116 | Yes |
| medium | ConvLSTM | core4 | 1 | 2.451022446155548 | Yes |
| medium | ConvLSTM | core4 | 3 | 1.581842489540577 | Yes |
| medium | ConvLSTM | core4 | 6 | 1.5877669304609299 | Yes |

## Readout

- Best single-split ConvLSTM RMSE does **not** improve: small `core4_h3` is **1.5704**, medium `core4_h3` is **1.5818**.
- Medium improves the weak parts of the grid. History-1 ConvLSTM and plain encoder-decoder rows are much less brittle than in the small run.
- In the seed-42 medium run, ConvLSTM is best overall at `core4_h3`; the plain encoder-decoder still wins some matched settings (`t2m_h1`, `t2m_h3`, `core4_h1`).
- The ranking is more stable than the small run, but it still depends on model, input set, history, and split.

## Bottom line

Medium data makes the experiment more reliable, but it does **not** show that three months lowers the best achievable RMSE. Keep the architecture fixed, add calendar coverage, and keep reporting split variability.

## Stability Across Splits

Medium was rerun with `split_seed` / `seed` values **42**, **7**, and **123**, writing separate outputs under `results/experiments_medium_seed_<seed>/`. Values below are sample mean ± sample standard deviation across the three split/training seeds.

| Model | Variables | History | RMSE mean ± std |
| --- | --- | ---: | ---: |
| ConvLSTM | core4 | 6 | 1.4845 ± 0.1215 |
| ConvLSTM | core4 | 3 | 1.5289 ± 0.0806 |
| ConvLSTM | t2m | 6 | 1.6122 ± 0.2689 |
| PlainEncoderDecoder (`cnn`) | t2m | 3 | 1.6345 ± 0.0451 |
| PlainEncoderDecoder (`cnn`) | core4 | 3 | 1.8810 ± 0.2539 |
| ConvLSTM | t2m | 3 | 1.9028 ± 0.1547 |

Winner varies by seed: ConvLSTM `core4_h3` wins seeds 42 and 7; seed 123 wins with ConvLSTM `t2m_h6`. By mean RMSE, ConvLSTM `core4_h6` is best and `core4_h3` is close. The useful takeaway is narrower: temporal ConvLSTM configurations are the strongest group, but history length and input set remain split-sensitive. ConvLSTM does **not** beat the plain encoder-decoder in every matched setting, and `core4` does **not** consistently beat `t2m` for the plain encoder-decoder.
