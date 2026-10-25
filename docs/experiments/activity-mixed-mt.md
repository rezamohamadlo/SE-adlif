# Activity-dependent MT mixture (AMT)

AMT_SE_adLIF extends MT with a causal, activity-dependent mixture of its
existing fast and slow adaptation currents. This is a research hypothesis,
not a demonstrated improvement or a verified novelty claim.

For each sample and layer, update two traces from the previous timestep's
mean hidden spike activity:

```
r_fast[t] = 0.9 * r_fast[t-1] + 0.1 * mean(z[t-1])
r_slow[t] = 0.99 * r_slow[t-1] + 0.01 * mean(z[t-1])
g[t] = sigmoid(mix_logits + mix_activity_gain * (r_fast[t] - r_slow[t]))
adaptation[t] = g[t] * w_fast[t-1] + (1 - g[t]) * w_slow[t-1]
```

Both traces start at zero for each sequence. The trace rates are fixed local
architecture choices, not paper-derived settings. The intercept and activity
gain are neuron-specific learned parameters; the gain is initialized to zero.
Consequently, copying MT weights into AMT reproduces MT's initial dynamics.
The proposed experiment uses **fresh random initialization**, with fresh
optimizer/scheduler state and epoch zero; it does not load an MT checkpoint.
Thresholds, adaptation drives, and MT's time-constant ranges are unchanged.
Activity is isolated between samples, and no future frames enter the gate.
The additional activity gain adds one parameter per hidden neuron. The two
activity states add memory and operations; actual inference costs must be
measured rather than inferred from parameter count.

## Comparison and outputs

Compare reference SE_adLIF, MT_SE_adLIF, and AMT_SE_adLIF on SHD*, SSC and
ECG with training seeds 42, 123 and 456. The AMT experiment configs inherit
the corresponding MT architecture and dataset-specific comparison group;
the latter inherits `config/paper_baseline.yaml`. No optimization, budget,
preprocessing, loss, or stopping overrides are proposed. SHD* keeps split
seed 42 and a 20% validation reserve; SSC uses the runner's mean loss.
The historical ECG reference's budget/stopping mismatch remains a limitation
when comparing existing records. Do not describe that row as fully controlled.

All nine runs were imported on 25 October 2026 and are retained under
`results/not_improved/AMT_SE_adLIF/<dataset>/seed_<seed>/`.
Existing reference and MT results remain at their recorded locations. Results
must retain baseline identity, fully resolved configs and architecture choices.
Do not merge artifacts from different devices into one seed directory or run
multiple launchers writing the same directory. Distribute disjoint datasets
or seeds across devices and preserve the relative output layout when copying.

The retention category reflects SHD* accuracy; SSC/ECG outcomes remain separate.
The old output root resolves here through an explicit path alias. Saved files
retain their original bytes. See the [outcome record](../../results/not_improved/AMT_SE_adLIF/README.md)
and [transfer audit](../transfers/reza-amt-import-2026-10-25.md).

## User-launched commands

Run from the repository root using the project's `.venv` Python. Preflight:

```powershell
.\.venv\Scripts\python.exe runner.py --models AMT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/not_improved --dry-run
```

Completed jobs are skipped; an interrupted job's own `last.ckpt` is resumed if
the same command is repeated, and matching completed jobs are skipped:

```powershell
.\.venv\Scripts\python.exe runner.py --models AMT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/not_improved
```

These matching nine AMT runs are already complete and will be skipped.
Reuse recorded reference/MT evidence for comparison, retaining the historical
ECG reference budget/stopping caveat. Do not retrain those baselines as part
of this follow-up.

The runner performs final testing of validation-selected checkpoints and
measures completed runs automatically. Cost failures do not require retraining.
For multiple devices, replace `--datasets SHD SSC ECG` with one disjoint
dataset on each device. Transfer datasets/checkpoints explicitly if needed;
Git does not carry ignored artifacts. Paths above are repository-relative,
and the runner resolves them against its repository root despite Hydra chdir.

Freeze these architecture choices before reviewing the new test results.
If tuning is needed, use separate outputs and validation-only selection;
do not select architecture or gate settings using the test set. Previous test
scores have already been observed, so retain the follow-up disclosure.

Report per-seed accuracy, mean and sample SD, plus costs. In paper comparison
tables use `(Δ paper)` headings and signed differences in percentage points
from unrounded local metrics. The gate should also be inspected for time
variation: a constant learned mixture would weaken the intended explanation.
