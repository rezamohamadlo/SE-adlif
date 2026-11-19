# SSC AMT/MT full-validation diagnostics v1

Prepared, user-executed and reviewed on 2026-11-12. **Full-validation evaluation
is complete and verified; test execution remains unverified.** See the
[saved-evidence review](../../results/experiments/adaptation_ssc_amt_full_validation_v1/POST_RUN_REVIEW.md).
This extends the [completed 256-record SSC pilot](../../results/experiments/adaptation_ssc_amt_pilot_v1/POST_RUN_REVIEW.md)
to **all 9,981 official validation records under every condition**. The pilot's
implementation, manifests and results are preserved. Roadmap stage 6 remains
in progress; this does not start stage 7 gradients.

## Frozen scope

The [versioned study](../../config/studies/adaptation_ssc_amt_full_validation_v1.yaml)
inherits `config/paper_baseline.yaml`'s SSC section. It freezes the same six
existing validation-selected AMT/MT checkpoints at seeds 42, 123 and 456. No
training or checkpoint/hyperparameter selection occurs. The new
[entrypoint](../../scripts/adaptation_ssc_full/evaluate.py), runtime and report
writer live in a separate folder so completed pilot source hashes stay valid.

SSC keeps its official validation split, LR .006, completed 40-epoch training
budget, batch 256, two 720-neuron hidden layers, 140 input channels, 35 classes,
four-ms bins, minimum 300 frames and ten-frame burn-in. Evaluation uses
`mean_membrane_potentials` under `ssc_mean_loss_v1`, without an auxiliary training
penalty or TTA. This does not copy SHD holdout, budget or loss settings.

Six conditions retain the pilot definitions: clean, count-preserving time scales
.9 and 1.1, and 10% per-event deletion with three deterministic realizations.
Every condition contains the original validation IDs in file order, including
the original class frequencies. The pilot's class-balanced sampling is not used;
there is no `--samples` flag. The original deterministic perturbation seed formula
is retained, making matching recordings' inputs comparable with the pilot.

The validation-only loader authenticates the same H5 and never opens train/test
H5 files, constructs a dataset or downloads data. Timestamp widening occurs
before conversion to microseconds. Saved downsampling, framing, minimum padding,
burn-in and batch boundaries are preserved. Features use unpadded physical count
support; mean-membrane classification keeps the saved treatment of minimum
padding and block masks. Those are distinct contracts, not a change in loss.

Clean features are extracted first in batches. Each checkpoint then performs
**one full clean forward evaluation**, serving both the clean diagnostic condition
and its compatibility anchor. All selected checkpoints must match their saved
selected-epoch validation accuracy/loss within 1e-6 (fraction) and .001 before any
perturbation features or inference begin. Anchor failure stops the workflow.
Inspect preprocessing, masks, configs and aggregation; preserve the evidence and
do not replace checkpoints or relax tolerances to obtain a preferred result.

For each remaining condition, input features are shared across all model/seed
evaluations. Input hashes enforce identical reconstructed inputs. Preprocessing
and inference retain count tensors for one batch at a time. Paired CSV records
stream to disk, while only report fields remain in memory; result metadata and
reports still grow with the number of recordings. No result is imported from
the pilot, and no extra clean forward pass duplicates the new clean condition.

## Preflight and command interface

Agents may execute the first command only; users launch evaluation and tests.
**The user selected one PC on 2026-11-12.** Run one foreground launcher on
`cuda:0` for both models and all three training seeds (42, 123, 456), sequentially
within each condition. Use the main study root, without seed-cohort splitting.
Results keep their study/model/seed identities independently of this assignment.

```powershell
# Preflight only: no model forward, features, statistics, tests or result writes.
.\.venv\Scripts\python.exe scripts/adaptation_ssc_full/evaluate.py --check-only

# PC1: user launches both models and all three seeds on the single local GPU.
.\.venv\Scripts\python.exe scripts/adaptation_ssc_full/evaluate.py --device cuda:0

# Read-only saved-output verification after execution.
.\.venv\Scripts\python.exe scripts/adaptation_ssc_full/evaluate.py --verify-only

# Contract tests: user launches; the agent has not run these.
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ssc tests.test_adaptation_ssc_full
```

The pilot's measured execution was 412.344 seconds on an RTX 4090. Its saved
file timestamps show most execution time was spent in the six 9,981-record clean
anchor evaluations; timestamps are approximate local provenance, not a benchmark.
This extension performs 359,316 sample-forward evaluations across six conditions,
and extracts features for 59,886 condition/record combinations. **Budget roughly
30–45 minutes on one comparable idle RTX 4090, allowing up to an hour.** This
estimate has not been measured for the new workflow; disk speed, hardware and
concurrent workloads can change it. No additional timing experiment was run.

**Recorded completion:** the user's one-PC RTX 4090 run took 2,175.671 seconds
(36 minutes 16 seconds), including execution after startup preflight. All 57
output hashes, six clean anchors and 179,658 paired records passed review. The
planning estimate above is retained as historical context, not a new benchmark.

The PC count is settled for this study: **one PC**, with the single command
above. No cohort assembly is required for this plan. As an optional interface,
`--seeds` can select a unique
subset of 42/123/456, leaving the full validation sample set unchanged. Automatic
partial-cohort roots are siblings of the main study directory:
`results/experiments/adaptation_ssc_amt_full_validation_v1_seed_cohorts/<sorted-seed-IDs>/`.
This identity uses seeds, not PC labels. Partial reports explicitly list missing
members of the intended three-seed cohort. Assembly of separately executed seed
cohorts is not implemented yet; if multiple PCs are selected, prepare and verify
that assembly workflow before issuing a distributed execution plan. Do not pool
or copy summary values as a substitute for verified per-record aggregation.

To resume an interrupted unchanged evaluation, the user repeats the same command.
Complete cached chunks are reused only after manifest and checksum checks;
an incomplete chunk restarts. Each destination has an OS-owned analysis lock.
Do not remove a lock file to bypass an active process or launch two writers in
the same destination. Changing sources, config or environment requires a new
explicit `--output-root results/experiments/<new-study-name>` when the manifest
differs. The pilot root must never be reused for full validation.

## Results and interpretation

For all three seeds, the default destination is
`results/experiments/adaptation_ssc_amt_full_validation_v1/`:

- `MANIFEST.json`: resolved SSC config, all official validation IDs, run/metric/
  checkpoint/data identities, normalized source hashes and environment.
- `diagnostic_rows/`: six shared feature chunks, 36 prediction chunks and
  `paired_samples.csv` with **179,658 unique paired records**. Clean prediction
  chunks also supply the anchors; there are no duplicate full-clean chunks.
- `VALIDATION_ANCHORS.json`: six compatibility results against saved training
  metrics, with differences and sample counts.
- `analyses/{clean,time_scaling,spike_deletion}/`: full-validation JSON/MD
  summaries and PNG/SVG figures, clearly identified as SSC full validation.
- `PROGRESS.json`: current condition/model/seed and elapsed time, plus completed
  condition timings. `COMPLETE.json` records phase times, scope, seed-cohort
  completeness and hashes for **57 evidence outputs** in the three-seed case.

Reports retain per-training-seed results and sample SD. Deletion realizations
are averaged within each training seed before across-seed descriptive effects;
they are not additional trained models. Feature associations are descriptive
and do not establish causality. Relative loss-benefit changes and accuracy gaps
remain distinct, and improvement in a relative gap does not imply better absolute
loss or accuracy. No bootstrap, regression fit, significance test, cross-dataset
loss pooling or architecture selection is added by this workflow.

The same validation data already participated in checkpoint selection. Full
coverage improves the scope of diagnostic evidence but is still exploratory
validation evidence, not independent confirmation or final-test evidence.
Keep negative/mixed pilot findings alongside these eventual results.

Project paths are repository-relative and resolve through `project_paths.py`,
independently of the working directory. Required local artifacts must exist;
Git does not transfer ignored checkpoints or datasets. Sharing the entire new
result root preserves its saved-output verification; repeating inference requires
the same recorded data and checkpoint files under the relative layout. Hardware
is provenance, not artifact identity. Source/config/environment guards still
apply when transferring to another checkout.

Read-only post-run review verified saved hashes, all anchors, full sample
coverage, shared feature identities, paired records and per-seed report consistency.
It did not rerun inference, features, statistical fits, resampling or tests.
Unit-test execution remains pending until explicitly evidenced. The
[single-command SSC suite](adaptation-ssc-suite-v1.md) now implements the remaining
SSC statistics and model-pair evaluation workflow and is preflighted for one PC.
Its user command includes the selected contract tests; execution/results are
still pending. ECG and gradients remain separate work. Stage 6 is still in progress.
