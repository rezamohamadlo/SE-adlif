# LT-DTH versus DTH: full SHD* validation

Status, 12 November 2026: user execution completed; all 63 recorded output
hashes verified, with full sample coverage, six clean anchors, prediction
pairing, summary calculations and six trace-parameter records checked.
Runtime: approximately 5 minutes 45 seconds. See the
[post-run review](../../results/experiments/adaptation_lt_dth_full_validation_v1/post_run_review/VERIFICATION.md).
No evaluation rerun is required. LT-DTH-specific adjusted statistics are now
implemented and preflight checked, awaiting user execution; see the
[statistics guide](adaptation-lt-dth-statistics-v1.md).
This is the next planned model pair in the supervisor roadmap. No training,
new checkpoint selection, official-test access, TTA or automatic statistical
analysis is performed. Short evaluations run locally without a PC-count question.

The [study definition](../../config/studies/adaptation_lt_dth_full_validation_v1.yaml)
inherits the full-validation study and its original pilot definitions, with
`config/paper_baseline.yaml` as the training-setting authority. It pins six
existing validation-selected checkpoints using SHA-256 and selected epochs:

| Model | Seed 42 epoch | Seed 123 epoch | Seed 456 epoch |
|---|---:|---:|---:|
| DTH | 178 | 247 | 98 |
| LT-DTH | 223 | 253 | 192 |

All 1,631 validation samples (20% of official training, split seed 42), training
seeds 42/123/456, features, clean condition, time scales .9/1.1 and event deletion
.1 with three realizations are unchanged. The saved settings are checked against
the paper baseline and each other, allowing only the registered neuron type and
LT-DTH trace bounds to differ. Historical path aliases resolve the same recorded
checkpoints; a missing checkpoint is never replaced with another file.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_pairs/lt_dth.py --check-only
.\.venv\Scripts\python.exe scripts/adaptation_pairs/lt_dth.py
```

The assigned command is also in [command.txt](../../command.txt). Use the
project's installed `.venv`. Preflight authenticates local checkpoints, saved
configs, original metrics, raw training H5 and full validation membership without
inference or output writes. Transfer missing ignored data/weights/metrics in the
same relative layout; no downloads are attempted. The launcher resolves project
artifacts from its own checkout rather than the current/Hydra working directory.

Outputs use `results/experiments/adaptation_lt_dth_full_validation_v1/`, with
shared `diagnostic_rows/` and `analyses/clean`, `analyses/time_scaling` and
`analyses/spike_deletion`. Hardware is provenance only. The clean predictions
and input features are shared between analyses. Full clean anchors must agree
with saved validation evidence before interpreting interventions. Progress is
visible in the terminal and `PROGRESS.json`. Runtime is not guaranteed from
the preceding pair's approximately six-minute evaluation.

`parameters/<model>_seed_<seed>.json` additionally records the checkpoint identity
and both layers' fast/slow trace decays. DTH rates are fixed; LT-DTH rates are
learned. E-folding descriptors are `-1/log(rho)` frames, multiplied by the saved
4-ms input bin for physical milliseconds. Neuron `dt=1` is not used as a physical
millisecond conversion. These are layer/model properties, not thousands of
independent sample-level measurements. No parameter–data correlation or causal
claim is established merely by recording them.

Identical interrupted runs reuse hash- and identity-checked prediction chunks.
This is continuation of evaluation, not training resume. Changed settings/code
or environment require a new output root; never bypass guards. The existing
AMT/MT evaluator and completed records are unchanged. This versioned runtime
reuses its numerical helpers and report reductions while adding LT-DTH labels
and trace descriptors.

After completion:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_pairs/lt_dth.py --verify-only
```

Share the complete result directory, including ignored `diagnostic_rows/`.
Completion hashes cover predictions, feature cache, reports, anchors and six
parameter records. Summaries show per-seed results and sample SD; perturbation
replicates remain distinct and are averaged within seed for the additional
effect table. They are not extra training seeds. The statistical-stage extension
for LT-DTH must be specified separately after this evaluation is reviewed;
the completed AMT/MT statistics specification is not silently reused.
