# Research gap (this repository vs mainstream weather ML)

## What this repository already shows

- A reproducible path from downloaded ERA5 NetCDF and PRISM rasters to aligned daily samples, with explicit checks on date overlap, history windows, and finite values.
- Controlled train/validation splits with split metadata carried in checkpoints so evaluation matches training.
- A plain encoder-decoder baseline (`cnn` alias) and a ConvLSTM baseline that map a short ERA5 history to a higher-resolution PRISM temperature field, plus persistence, upsampled ERA5, and a global linear baseline.
- Scripted sweeps over input channel sets (`t2m` vs `core4`) and history length, with RMSE/MAE logged against persistence.
- Documented behavior where learned models sometimes fail to beat persistence when history is too short or the setup is unfavorable.

## What the experiments actually confirm

These points follow only from the archived metrics in `docs/experiments/final_comparison.json` and the companion error map analysis, not from external literature.

- **Persistence is a strong baseline** at the recorded persistence RMSE (~2.356): upsampled latest ERA5 is already informative for daily temperature on this domain when data are scarce.
- **ConvLSTM uses temporal structure**: at **history 1** its RMSE is far above persistence for both `t2m` and `core4`; at **history 3** it drops below persistence for both input sets—so a short multi-day window matters for this model class here.
- **Multi-variable input gives a clear gain at the best history**: `core4` at history 3 beats `t2m` at history 3 (lower RMSE), consistent with winds and surface pressure carrying usable signal beyond 2 m temperature alone.
- **Performance is unstable across history length**: history 6 is worse than history 3 for ConvLSTM RMSE on both input sets even where it still beats persistence (`core4`) or fails (`t2m`)—not a smooth “more context is better” curve.
- **Spatial learning is weak in the sense tested**: gradient–error correlation on the best ConvLSTM run is small in `docs/experiments/error_analysis.json`, so residual error is not dominated by a single “steep slopes” story; fine-grid structure remains hard.

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

## Gap to foundation-scale models

- **Data volume and coverage:** This project aligns PRISM files to ERA5 for specific date ranges; operational-scale models use orders of magnitude more space–time volume.
- **Task formulation:** Here the target is station-quality gridded observations at fine resolution; global models predict full atmospheric states on a fixed grid at coarser resolution or specialize with different targets.
- **Compute and personnel:** Foundation and global forecasting models assume clusters and curated data pipelines; this repo is a laptop-scale controlled study.

## Current next step

This note predates the controlled U-Net, boundary, padding, and undertraining diagnostics. Those later checks suggest that another temporal or architecture-first sweep is not the cleanest next move. The current next step is to add real static spatial context, especially DEM-derived topography, to the fixed U-Net spatial benchmark and test whether it improves local structure, blur, and border behavior under the same split/metrics.
