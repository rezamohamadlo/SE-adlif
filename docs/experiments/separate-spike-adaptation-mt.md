# MT with separate spike-adaptation strengths

The fourth candidate, `SA_MT_SE_adLIF`, changes MT's adaptation drives. Ordinary
MT feeds the same expression into both its fast and slow memories:

```
fast_drive = (a * post_reset_membrane + b * current_spike) * q
slow_drive = (a * post_reset_membrane + b * current_spike) * q
```

SA-MT retains shared membrane coupling `a`, but gives the slow memory its own
per-neuron spike coefficient:

```
fast_drive = (a * post_reset_membrane + b      * current_spike) * q
slow_drive = (a * post_reset_membrane + b_slow * current_spike) * q
```

Both coefficients use the inherited `b_range`. `b_slow` is initialized by
copying the freshly initialized `b` without sharing its storage. The model
therefore starts with MT's exact dynamics for matching random initialization,
but the two strengths receive separate gradients. No trained checkpoint is
loaded. The inherited post-reset update order, time constants, static mixture
gate, thresholds, reset, recurrent connections, subthreshold coupling and
surrogate gradient stay unchanged. Resetting parameters copies the new fast
coefficient into the slow coefficient again; constraints clamp both strengths.

This tests whether fast and slow adaptation benefit from different strengths
of spike-triggered feedback. Ordinary MT couples their response to membrane
and spike signals through the same drive; the new parameter releases part of
that restriction. It is distinct from AMT, which changes the mixture over time,
MT temporal consistency, which changes the training objective, and LT-DTH,
which learns activity-trace timescales in DTH.

## Capacity and interpretation

The model adds one trainable parameter per hidden neuron: 720 for SHD*, 1,440
for SSC, and 72 for ECG under the current two-layer architectures. It adds no
recurrent states, but computing two drives adds operations. Measure actual
activity, compute and timing; parameter counts do not establish inference cost.

This is a local experimental hypothesis, not a demonstrated improvement or a
verified novelty claim. An accuracy gain alone would not separate the proposed
mechanism from the additional parameter capacity. Examine the slow/fast
coefficients, time constants and mixture jointly. The gate also scales the
currents, so effective weighted spike strengths (`mix*b` and
`(1-mix)*b_slow`) are more informative than comparing raw coefficients alone.
No biological specialization or causal explanation follows just from different
learned coefficient values. A later tied-coefficient control or capacity control
would need a separate, validation-selected experiment.

## Fair comparison through the main runner

Compare primarily against MT on matching training seeds 42, 123 and 456.
The three experiment configs inherit MT architecture and dataset-specific
comparison groups, which inherit `config/paper_baseline.yaml`. Only the hidden
cell identifier changes. The supervised loss, LR, batch, preprocessing, split,
epoch budget, stopping rule, and optimizer remain inherited. SHD* uses 20% of
the official training set for validation with split seed 42. SSC keeps mean
loss. ECG keeps its per-timestep targets. The historical ECG reference budget
caveat still applies if including that reference in a broader comparison.

No new launcher is needed. `runner.py` saves resolved configs and model-variant
deviations, selects checkpoints on validation, performs final tests, aggregates
matching seeds, and measures newly completed jobs automatically.

Non-training preflight from the repository root:

```powershell
.\.venv\Scripts\python.exe runner.py --models SA_MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/sa_mt_v1 --dry-run
```

The user launches the nine jobs locally:

```powershell
.\.venv\Scripts\python.exe runner.py --models SA_MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/sa_mt_v1
```

Outputs are under
`results/experiments/sa_mt_v1/SA_MT_SE_adLIF/<dataset>/seed_<seed>/`.
Fresh jobs start with random weights and a fresh optimizer at epoch zero.
Repeating the same command resumes interrupted jobs from their own `last.ckpt`
and skips matching completions; final testing uses the best validation-selected
checkpoint. Cost-measurement failures stay separate from training completion.
The checkpoints retain `b_slow` and the registered neuron identity, so evaluation
must reconstruct this model from its saved config rather than substitute MT.

Keep this configuration fixed before reviewing test results. Put any tuning
in separate configs/output directories and select it with validation. Retain
the disclosure that prior test results were already observed. Report per-seed
accuracy, mean and sample SD, and cost measurements. In paper comparison tables
use `(Δ paper)` headings and signed percentage-point deltas from unrounded
metrics. Preserve existing evidence and the other devices' active runs.
