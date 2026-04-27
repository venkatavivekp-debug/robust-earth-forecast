# Literature notes (ERA5 → PRISM context)

Short summaries tied to this repository: regional statistical downscaling with small aligned samples, CNN/ConvLSTM baselines, and comparison to persistence. No new methods are implied here—only what is useful to read next.

---

## 1. Prithvi WxC: Foundation Model for Weather and Climate

- **Link:** https://arxiv.org/abs/2409.13598  
- **Code:** https://github.com/NASA-IMPACT/prithvi-wxc  

**Summary.** IBM and NASA describe a large weather-and-climate foundation model pretrained on long records of atmospheric fields, with encoder–decoder attention and reported downstream use cases including forecasting and downscaling-style tasks. Training uses broad multivariate spatiotemporal coverage at reanalysis-like resolution, not a single small region.

**Why it matters here.** It is a concrete example of how modern practice couples huge data volume, many variables, and long temporal context before specialization.

**What can be adopted.** Treat multivariate inputs and temporal windows as first-class design choices; keep evaluation against simple baselines; document data limits explicitly.

**What cannot be adopted at this scale.** Model size, compute budget, and global/long-record pretraining are out of scope for this repo’s current data volume.

---

## 2. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

- **Link:** https://arxiv.org/abs/1506.04214 (Shi et al., 2015; NeurIPS proceedings)  

**Summary.** ConvLSTM replaces fully connected LSTM gates with convolutions so hidden state preserves spatial structure. The paper focuses on sequence-to-sequence prediction of radar echo maps—closely related to using a short history of gridded fields to predict the next field.

**Why it matters here.** The repo’s ConvLSTM baseline is the same family of idea: temporal recurrence with spatially local mixing.

**What can be adopted.** History length as a controlled knob; monitoring for instability when sequences are short or data are scarce.

**What cannot be adopted wholesale.** Radar nowcasting setups use dense high-resolution sequences; this project has far fewer days and a reanalysis-to-observation gap.

---

## 3. GraphCast: Learning skillful medium-range global weather forecasting

- **Link:** https://arxiv.org/abs/2212.12794 (Lam et al.; see also *Science* publication and https://github.com/google-deepmind/graphcast)  

**Summary.** GraphCast learns global medium-range forecasting from reanalysis (ERA5) using a graph-based architecture on a multi-scale mesh, with training at a scale that covers decades of data and many variables.

**Why it matters here.** It shows what becomes possible when the model can see full spatial context and long climate statistics; persistence is not the main competitor in that regime.

**What can be adopted.** Emphasis on honest baselines, held-out time, and reporting against operational or numerical references where applicable.

**What cannot be adopted.** Global mesh, parameter count, and training data volume are not transferable to a Georgia-only PRISM slice without a different project scope.

---

## 4. FourCastNet: A Global Data-driven High-resolution Weather Model Using Adaptive Fourier Neural Operators

- **Link:** https://arxiv.org/abs/2202.11214 (Pathak et al.; NVIDIA; code: https://github.com/NVlabs/FourCastNet)  

**Summary.** FourCastNet uses adaptive Fourier neural operators in a ViT-like stack for fast global forecasting from ERA5-like inputs at 0.25° scale, trained on large historical corpora.

**Why it matters here.** It is another reference for “large model + massive reanalysis” rather than “small regional supervised downscaling.”

**What can be adopted.** Thinking in terms of operator/view that matches grid structure; careful normalization and multi-channel inputs at the *conceptual* level.

**What cannot be adopted.** Training infrastructure and global 6-hourly archives at full resolution are not available in this prototype.

---

## 5. Repeatable high-resolution statistical downscaling through deep learning

- **Link:** https://doi.org/10.5194/gmd-15-7353-2022 (Hill et al., *Geoscientific Model Development*, 2022)  

**Summary.** The paper presents a deep-learning downscaling workflow with attention to reproducibility, validation design, and high-resolution targets—closer in spirit to regional SDM than to global forecasting models.

**Why it matters here.** It aligns with the repo’s actual problem class: learn a mapping from coarse predictors to a finer grid with documented procedures.

**What can be adopted.** Clear experiment logging, held-out evaluation, and reporting relative to simpler mappings.

**What cannot be adopted without more data.** Their setups still assume substantially more data and often broader domains than a short PRISM window over one state.

---

## 6. The ERA5 global reanalysis

- **Link:** https://doi.org/10.1002/qj.3803 (Hersbach et al., *Quarterly Journal of the Royal Meteorological Society*, 2020)  

**Summary.** ERA5 is produced by a numerical weather prediction system constrained by observations; it is not a direct observational product. Biases and representation error vary by variable, height, and region.

**Why it matters here.** Coarse ERA5 2 m temperature is a plausible predictor for PRISM, but the two products differ in physics, grid, and observation influence.

**What can be adopted.** Treating misalignment and systematic bias as expected, and reserving strong claims until sample size grows.

**What cannot be adopted.** Reanalysis cannot be “fixed” by a small neural net alone; any learned correction is data-limited.

---

## 7. Bias correction, quantile mapping, and downscaling: linking separate approaches

- **Link:** https://doi.org/10.1002/wcc.372 (Maraun, *WIREs Climate Change*, 2013)  

**Summary.** Reviews connections between bias correction, quantile mapping, and statistical downscaling, and stresses that naïve application can distort small-scale variability and cross-variable consistency.

**Why it matters here.** Supervised ERA5 → PRISM is a form of learned mapping across scales; the same structural cautions apply when *N* is small.

**What can be adopted.** Careful interpretation of improvements over persistence; checking spatial and marginal properties beyond a single RMSE.

**What cannot be adopted as a turnkey fix.** Full multivariate calibration theory needs more data and often explicit probabilistic goals than this baseline repo targets.

---

## 8. The analog method as a simple statistical downscaling technique

- **Link:** https://doi.org/10.1175/1520-0442(1999)012<2474:TAMASS>2.0.CO;2 (Zorita & von Storch, *Journal of Climate*, 1999)  

**Summary.** Pairs large-scale circulation states with historical analog days to infer local climate; compares favorably in that study with more elaborate SDM variants for some targets.

**Why it matters here.** It is the classical counterpart to learned deep mappings: similarity in predictor space replaces millions of parameters when long archives exist.

**What can be adopted.** Treating persistence and linear maps as mandatory sanity checks echoes the analog idea of “do not ignore cheap structure.”

**What cannot be adopted without data.** A meaningful analog pool needs a long, consistent archive; a month-scale PRISM slice is the opposite regime.

---

## Quick index

| Reference | Role in this project |
|-----------|----------------------|
| Prithvi WxC | Scale and pretraining reference |
| Shi et al. ConvLSTM | Justifies temporal conv recurrent baseline |
| GraphCast / FourCastNet | Large-scale ERA5 ML reference |
| Hill et al. GMD | Downscaling + reproducibility |
| Hersbach ERA5 | Predictor product limitations |
| Maraun WIREs | Scale/bias/mapping caveats |
| Zorita & von Storch | Analog / cheap-structure baselines when data are limited |
