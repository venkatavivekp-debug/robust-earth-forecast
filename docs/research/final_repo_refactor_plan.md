# Final repository refactor plan

This is a conservative organization plan. It flags future work; it is not a deletion request.

## Keep unchanged

- `data_pipeline/`: ERA5 and PRISM download/validation scripts.
- `datasets/`: path resolution and ERA5/PRISM sample construction.
- `models/`: architecture implementations and compatibility aliases.
- `training/train_downscaler.py`: current trainer CLI.
- `evaluation/evaluate_model.py`: current metrics and plot writer.
- `scripts/check_spatial_artifacts.py`: border/center diagnostic.
- `scripts/spatial_error_analysis.py`: gradient/error diagnostic.
- `docs/experiments/*.json`: committed result records.
- `docs/images/*.png`: figures referenced by README/docs/notebook.
- `notebooks/analysis.ipynb`: main analysis notebook and committed outputs.

## Rename later

- `models/cnn_downscaler.py`: keep for compatibility now, but the implementation is better named `plain_encoder_decoder.py`.
- `CNNDownscaler`: keep as a checkpoint/script alias now; prefer `PlainEncoderDecoder` or `EncoderDecoderBaseline` in new research text.
- `--model cnn` and `--cnn-checkpoint`: keep while archived results depend on the name; consider adding nonbreaking `plain_encoder_decoder` CLI aliases later.
- `scripts/run_core_experiments.py`: currently an archived encoder-decoder/ConvLSTM grid runner. A future spatial comparison runner may deserve a more explicit name.

## Archive later

- `training/run_temporal_analysis.py`: appears out of sync with the current trainer/evaluator flags.
- `training/run_ablation.py`: appears out of sync with the current trainer/evaluator flags.
- `training/tune_downscaler.py`: keep until the main comparison protocol is stable, then decide whether it still belongs in the active path.
- `docs/research/next_experiment_plan.md`: useful history, but it is temporal-first and should be superseded by the controlled spatial comparison protocol.

## Review before removal

- Any local `results/experiments*` folder.
- Any local `results/diagnostics` folder.
- Any local `checkpoints` folder.
- Any local `data_raw` folder.
- Any committed figure in `docs/images`.
- Any committed experiment table or JSON in `docs/experiments`.

These files may be generated, but they are also the evidence trail for the current results.

## Remaining clutter risks

- Research notes overlap: `problem_structure.md`, `problem_structure_notes.md`, `unet_transition_plan.md`, and `current_baseline_definition.md` cover related ideas. Keep them for now, then merge after the next controlled U-Net comparison is complete.
- The historical `cnn` alias is still present in code and result paths. This is intentional for reproducibility, but all new text should call the concept `PlainEncoderDecoder` or `EncoderDecoderBaseline`.
- The existing U-Net result is a preliminary spatial check. The repository still needs one controlled comparison table that freezes split, normalization, target mode, and diagnostics across persistence, `PlainEncoderDecoder`, and proper skip-connected U-Net.
