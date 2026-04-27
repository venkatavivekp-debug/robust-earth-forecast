# Research gap (this repository vs mainstream weather ML)

## What this repository already demonstrates

- A reproducible path from downloaded ERA5 NetCDF and PRISM rasters to aligned daily samples, with explicit checks on date overlap, history windows, and finite values.
- Controlled train/validation splits with split metadata carried in checkpoints so evaluation matches training.
- CNN and ConvLSTM baselines that map a short ERA5 history to a higher-resolution PRISM temperature field, plus persistence, upsampled ERA5, and a global linear baseline.
- Scripted sweeps over input channel sets (`t2m` vs `core4`) and history length, with RMSE/MAE logged against persistence.
- Documented behavior where learned models sometimes fail to beat persistence when history is too short or the setup is unfavorable.

## Why it often struggles to beat persistence

- **Persistence is strong when the target day is close to the last coarse input in Kelvin space.** Upsampling the latest ERA5 frame already tracks synoptic-scale temperature; gains require predicting corrections at PRISM resolution and correcting reanalysis–observation differences.
- **Small validation sets** (on the order of a handful of days in the default configuration) make RMSE differences noisy; a model can look worse than persistence by chance.
- **Optimization sensitivity** matters at small *N*: learning rate, weight decay, early stopping, and L1 weight on outputs interact; under-training leaves the model between persistence and truth.

## Role of temporal context

- ConvLSTM uses consecutive days before the target; too little history removes dynamical signal; too much history on a tiny dataset can increase parameters relative to samples and hurt generalization.
- Recorded runs in `docs/experiments/final_comparison.json` show history=3 outperforming history=1 and, in that grid, history=6 not improving further—consistent with temporal information helping only when the window matches data support.

## Role of multi-variable input

- Wind and surface pressure carry information about advection and mass patterns not visible in 2 m temperature alone; `core4` improves over `t2m` in the logged grid when ConvLSTM is adequately trained.
- Variables missing from the NetCDF cannot help; the pipeline selects channels that exist in the file.

## Why small datasets break deep models

- Convolutional recurrent stacks have many weights relative to tens of days of data; variance is high and validation metrics swing with split seed.
- The model can fit training quirks without learning stable cross-resolution structure; regularization and early stopping help but cannot invent information that is not in the sample.

## Why large models (Prithvi WxC, GraphCast, FourCastNet) succeed on their tasks

- They train on long multivariate records (years to decades) and global or near-global spatial context, so parameters are constrained by massive evidence.
- Architectures are sized to match that evidence; optimization noise averages down.
- Their benchmarks are forecasting or broad downscaling at reanalysis-like scales, not a single U.S. state with a handful of PRISM days.

## The exact gap between this repo and those models

- **Data volume and coverage:** This project aligns PRISM files to ERA5 for specific date ranges; operational-scale models use orders of magnitude more space–time volume.
- **Task formulation:** Here the target is station-quality gridded observations at fine resolution; global models predict full atmospheric states on a fixed grid at coarser resolution or specialize with different targets.
- **Compute and personnel:** Foundation and global forecasting models assume clusters and curated data pipelines; this repo is a laptop-scale controlled study.

## Most realistic next step

Extend aligned ERA5/PRISM coverage in time (more months or years), keep architecture fixed, and re-run the same history-length and input-set sweeps so improvements (or lack thereof) reflect sample growth rather than new layers. Until the sample count rises, treat any architecture change as poorly identified.
