# Undertraining diagnosis results

This run checks whether the current U-Net setup is mainly limited by training budget. It uses the strongest current spatial/boundary setup:

- model: skip-connected U-Net
- dataset: medium
- input set: `core4`
- history: 3
- target mode: direct
- padding: replicate
- upsampling: bilinear
- seed/split: 42
- early stopping: disabled, so the budgets are actual epoch counts

Command:

```bash
.venv/bin/python scripts/run_undertraining_diagnosis.py \
  --dataset-version medium \
  --input-set core4 \
  --history-length 3 \
  --target-mode direct \
  --padding-mode replicate \
  --upsampling-mode bilinear \
  --budgets 80 160 300 \
  --seed 42 \
  --split-seed 42 \
  --overwrite
```

Outputs are under `results/undertraining_diagnosis/`.

## Metrics

| Budget | Best epoch | Best val loss | Final train loss | Final val loss | RMSE | MAE | Border RMSE | Center RMSE | Border/Center | Variance ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 64 | 3.0510 | 1.0900 | 3.1026 | 1.7291 | 1.3406 | 1.9405 | 1.6538 | 1.173 | 0.949 |
| 160 | 160 | 3.0445 | 1.0447 | 3.0445 | 1.7295 | 1.3404 | 1.9422 | 1.6538 | 1.174 | 0.928 |
| 300 | 290 | 2.9897 | 1.0101 | 3.0056 | 1.7135 | 1.3272 | 1.9299 | 1.6363 | 1.179 | 0.930 |

## Read

Longer training helps a little, but not much. The 300-epoch run gives the best RMSE and best validation loss, but the gain from 80 epochs is about `0.016` RMSE. The 160-epoch run is essentially tied with 80 epochs on evaluation RMSE.

The curves suggest mild remaining optimization headroom, not a clear undertraining failure. Training loss keeps decreasing, while validation loss moves slowly and stays near the same range after the early improvement.

Blur is not clearly reduced. Prediction variance remains below target variance for every budget, and it is lower at 160/300 epochs than at 80 epochs. The saved panels should still be inspected visually, but the scalar diagnostics do not show a strong sharpening effect.

Border degradation remains. Border RMSE improves slightly at 300 epochs, but the border/center ratio stays above 1 and does not improve with longer training.

## Decision

Undertraining is only weakly supported. There is a small validation improvement at 300 epochs, so the model was not fully saturated at 80 epochs. But the small magnitude of the gain, persistent blur indicators, and unchanged border ratio suggest that training budget is not the main cause of the remaining artifacts.

The next phase should not be another large epoch sweep. The more useful next step is to keep the 300-epoch budget as a reference and test missing physical/context information, especially real terrain/topography, under the same controlled evaluation.

Concise conclusion: longer training gave only mild improvement, so undertraining alone is not the dominant explanation. That points the next phase toward physically meaningful static spatial context, especially terrain, before adding temporal complexity.
