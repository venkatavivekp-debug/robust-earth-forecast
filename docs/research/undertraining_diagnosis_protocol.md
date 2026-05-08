# Undertraining Diagnosis Protocol

Professor Hu noted that the prediction panels still look blurred and that the models may be undertrained. Before adding topography, changing the loss, or adding temporal structure, this phase checks whether the current best spatial setup simply needs more optimization time.

## Question

Is the current U-Net spatial baseline undertrained, or are the remaining errors more consistent with missing context, loss/data limits, or the coarse-to-fine ERA5 -> PRISM problem itself?

## Fixed setup

The diagnosis uses the strongest current spatial setup from the boundary ablation:

- model: skip-connected U-Net
- input set: `core4`
- history: 3 days
- target mode: direct
- dataset version: medium
- seed and split seed: 42
- padding: replicate
- decoder upsampling: bilinear
- optimizer, learning rate, normalization, loss, batch size, and evaluation pipeline: unchanged

The only planned change is training budget. Early stopping is disabled for this runner so the budget labels are actual training durations rather than upper bounds.

## Budgets

- 80 epochs
- 160 epochs
- 300 epochs

Each run writes its own checkpoint, training curve, prediction panel, absolute error map, and boundary diagnostics under `results/undertraining_diagnosis/`.

## Evidence for undertraining

Undertraining is supported if:

- training loss and validation loss both keep improving with longer budgets;
- validation RMSE/MAE improve materially at 160 or 300 epochs;
- prediction panels look less smoothed, not just numerically different;
- border and center errors both improve without clear overfitting.

Overfitting is more likely if training loss continues improving while validation loss or validation RMSE worsens.

Missing context, loss design, or data limitations are more likely if longer training changes training loss but does not meaningfully improve validation metrics, output sharpness, or boundary behavior.

This test should not be used to claim the model is solved. It only decides whether longer training is worth treating as the next lever before adding new predictors or losses.
