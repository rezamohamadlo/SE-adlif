# MT with training-only temporal consistency

This is the second proposed direction, separate from AMT. It keeps MT's
neurons and inference unchanged and adds an auxiliary training objective:

```
training loss = original dataset classification loss + 0.1 * temporal JS
```

Temporal JS is the mean Jensen-Shannon divergence between adjacent overlapping
five-frame windows of class probabilities. Each window averages the per-frame
softmax outputs. Only pairs fully inside the same valid target class are
included. Burn-in, ignored labels, batch padding and ECG class transitions
are excluded. The statistic weights eligible window pairs equally. When no
pair is eligible it is zero, with a valid zero gradient.

The hypothesis is that weak temporal smoothness can reduce unstable predictions
without adding a second excitability feedback mechanism or inference parameters.
It may also suppress useful fast changes or have too little effect; improvement
is not assumed. Five frames and weight 0.1 are local starting choices, not
paper-derived values. Log `train_temporal_js`, `train_total_loss`, supervised
`train_loss`, and ordinary validation accuracy to assess the effect.

This is **not a reproduction of TET**. TET changes temporal training objectives;
our proposed pilot uses same-label window consistency while retaining the
original supervised aggregation. Temporal regularization and self-distillation
are established research directions, so this experiment is not an established
novel contribution. Relevant primary sources:

- [TET, ICLR 2022](https://arxiv.org/abs/2202.11946).
- [Temporal-Spatial Self-Distillation](https://arxiv.org/abs/2406.07862).
- [Selective Alignment Knowledge Distillation, IJCAI 2026](https://www.ijcai.org/proceedings/2026/410).

## Protocol and isolation

The configs inherit MT architecture and the corresponding dataset comparison
group, which inherits `config/paper_baseline.yaml`. The original supervised
loss, LR, batch, epoch budget, split and stopping rule remain inherited.
The training objective differs, so metadata labels this separate protocol
`mt_temporal_consistency_v1`; do not describe it as an unchanged paper baseline.
SHD* uses split seed 42 with 20% held out. SSC uses the existing mean loss.
ECG keeps its timestep labels and does not smooth across class boundaries.

Only `train_mt_consistency.py` applies this auxiliary objective. Do not launch
these configs with `run.py` or `runner.py`, which instantiate the ordinary
classification training module. The separate script reuses the established
trainer within its own process without editing either main runner or the MT
neuron implementation. This is an objective experiment, not a replacement
for the official SSC reference/DTH/MT comparison workflow.

Runs start with fresh random weights and a fresh optimizer at epoch zero.
Repeating a matching command resumes only its own `ckpt/last.ckpt`; completed
full-budget validation runs are skipped. Configuration/source/environment
changes require a new output directory. A manifest mismatch is rejected before
the saved Hydra config is rewritten. Per-seed manifests record fully resolved
settings, baseline identity, deviations, seeds, source hashes and environment.
Locks prevent concurrent training in the same seed directory. Preserve the
relative output layout across PCs; Git does not transfer datasets or weights.

## Non-training preflight

From the repository root, resolve each dataset's config without training:

```powershell
.\.venv\Scripts\python.exe train_mt_consistency.py experiment=SHD_MT_consistency --cfg job --resolve
.\.venv\Scripts\python.exe train_mt_consistency.py experiment=SSC_MT_consistency --cfg job --resolve
.\.venv\Scripts\python.exe train_mt_consistency.py experiment=ECG_MT_consistency_2layer --cfg job --resolve
```

## User-launched validation pilot

These three commands run sequential seeds within each dataset. They can run
on different PCs because their dataset directories are disjoint:

```powershell
.\.venv\Scripts\python.exe train_mt_consistency.py --multirun experiment=SHD_MT_consistency random_seed=42,123,456
.\.venv\Scripts\python.exe train_mt_consistency.py --multirun experiment=SSC_MT_consistency random_seed=42,123,456
.\.venv\Scripts\python.exe train_mt_consistency.py --multirun experiment=ECG_MT_consistency_2layer random_seed=42,123,456
```

Outputs: `results/experiments/mt_consistency_v1/<dataset>/seed_<seed>/`.
Use saved validation results to assess the pilot against matched MT validation
metrics. Different datasets' raw validation scores are not directly comparable.
If tuning is needed, put every changed configuration in a separate output and
record its deviations. A weight-zero control tests loss equivalence; it must
have its own directory and is not an additional independent seed.

Official test evaluation is **disabled** in this pilot. Freeze the design using
validation before preparing standalone final testing and inference costs for
the selected checkpoints. Base MT inference loads these checkpoints with the
same state dictionary; no auxiliary objective runs during validation or test.
Retain the prior-test-observation disclosure when reporting later test results.

For subsequent paper comparisons, use `(Δ paper)` headings and signed
parenthesized differences from unrounded local metrics, in percentage points.
Report training-objective changes, matching seeds, mean and sample SD; do not
infer final test improvements from validation gains.
