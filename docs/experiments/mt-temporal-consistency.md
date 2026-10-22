# MT with training-only temporal consistency

`MT_TC_SE_adLIF` is an MT training-objective variant, supported by the official
`runner.py` workflow. Its inference architecture and checkpoint keys match MT;
its training loss adds an explicitly recorded auxiliary objective:

```
training loss = original dataset classification loss + 0.1 * temporal JS
```

Temporal JS is the mean Jensen-Shannon divergence between adjacent overlapping
five-frame windows of class probabilities. Only pairs fully inside the same
valid target class are included. Burn-in, ignored labels, batch padding and
ECG class transitions are excluded. When no pair is eligible the penalty is
zero with a valid zero gradient. This is a hypothesis, not a demonstrated
improvement: smoothing may suppress useful temporal changes or have too little
effect. Window 5 and weight 0.1 are frozen local experiment choices.

Temporal regularization is an established research direction. This variant
is not an exact TET reproduction or an established novelty claim. Related
primary sources: [TET](https://arxiv.org/abs/2202.11946),
[Temporal-Spatial Self-Distillation](https://arxiv.org/abs/2406.07862), and
[Selective Alignment Knowledge Distillation](https://www.ijcai.org/proceedings/2026/410).

## Shared runner and fair ablation

The new configs inherit MT architecture and the dataset comparison group,
which inherits `config/paper_baseline.yaml`. Compare MT and MT_TC with matched
training seeds 42, 123 and 456; supervised loss aggregation, LR, batch,
preprocessing, split, epoch budget and stopping rules match. SHD* retains a
20% training-derived validation split with split seed 42. SSC uses mean loss;
ECG uses its timestep labels. The auxiliary loss is the intended experimental
difference; the complete training objectives are not identical.

The main training entrypoint selects the consistency training module only
when `training_objective: mt_temporal_consistency_v1` is explicit. Ordinary
reference, MT, DTH and AMT runs keep their existing training behavior. The
runner records the objective, weight and window in baseline deviations and
appends `mt_temporal_consistency_v1` to the dataset-specific protocol identity.
Resolved configs and checkpoint hyperparameters preserve the settings.

## Commands

Run from the repository root. Non-training preflight for all nine jobs:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_TC_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/mt_consistency_runner_v1 --dry-run
```

The user launches fresh training, final testing and automatic cost measurement:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_TC_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/mt_consistency_runner_v1
```

Only the new variant is scheduled; reuse existing matched MT records for
comparison. Outputs are under
`results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/<dataset>/seed_<seed>/`.
Fresh runs start with random weights and a fresh optimizer at epoch zero.
Repeating the same command resumes each interrupted job from its own
`last.ckpt` and skips matching completions. Final test evaluation uses the
validation-selected best checkpoint. The runner provides aggregation and
per-seed `COST_RESULTS.json`/`.md`; cost failures do not require retraining.

On different PCs, select disjoint datasets or seeds and keep the same relative
output layout. Git does not transfer datasets or checkpoints. Paths resolve
against the repository root even when Hydra changes the working directory.
Changing the objective or other guarded settings requires a new directory.

This is a fixed follow-up configuration, not a test-selected search. Select
any later tuning using validation in separate outputs before evaluating its
final test. Previous test scores have already been observed; retain that
history. Log supervised `train_loss`, `train_temporal_js`, `train_total_loss`
and ordinary validation/test metrics. Report mean and sample SD over matching
seeds, use `(Δ paper)` labels where appropriate, and retain the historical ECG
reference stopping/budget caveat if comparing against that reference.

## Previous validation-only prototype

`train_mt_consistency.py` and the earlier `*_MT_consistency` configs remain
as the separate validation-only prototype record. Use the main-runner commands
above for new comparisons. Their old `results/experiments/mt_consistency_v1/`
outputs, if any, must not be reused or relabeled as the runner's completed
final-test evidence. Do not change an active prototype process or its artifacts.
