# SHD* numerical batching/precision follow-up

**Completed by the user and reviewed 16 November 2026:** all 568 registered
output hashes, 57 source hashes, 16 user tests and 546 jobs verified, with
unchanged model state. The [post-run review](../../results/experiments/adaptation_batch_precision_v1/POST_RUN_REVIEW.md)
records the numerical interpretation and limitations. Source-high original
anchors passed. Strict matmul precision removes prediction differences across
the tested pilot layouts, but changes three MT and five AMT predictions across
full validation; it cannot replace the original baseline. No gradients were
generated, and stage 7 remains incomplete. Recorded execute time was 277.547
seconds, excluding initial preflight. The next candidate is a separate
strict-precision gradient pilot with its own same-mode reference and stability
checks. That separate [gradient repeat pilot](adaptation-gradient-strict-v1.md)
is now implemented and preflight checked; its tests and scientific execution
have since been completed by the user and
[reviewed](../../results/experiments/adaptation_gradient_strict_v1/POST_RUN_REVIEW.md):
all 86 required repeat/padding checks passed. No original diagnostic outputs are
changed. Strict matched-deletion preparation is the next step.

The preparation protocol below was recorded 16 November 2026. The completed
[gradient pilot review](../../results/experiments/adaptation_gradient_pilot_v1/POST_RUN_REVIEW.md)
found an MT prediction change for recording **3896** when switching from
original batching to single-record diagnostics. Its CE increased by
0.3386995792. Original-batch anchors and same-mode autograd forward equivalence
passed. The cause has not been established.

For read-only verification of completed outputs:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_batch_precision.run --verify-only --output-root results/experiments/adaptation_batch_precision_v1
```

The original execution/resume command is retained for provenance. Run it from
the repository root, on the same local CUDA PC:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_batch_precision.run --device cuda:0 --output-root results/experiments/adaptation_batch_precision_v1
```

The command runs **16 new user contracts**, verifies/copies an authenticated
metadata subset of the pilot with a path index, then executes the forward-only
matrix and saves descriptive comparisons. No training, backward, new gradient
maps, deletions, input optimization, bootstrap, significance tests or official
test access occurs. The original pilot and its 544-output ledger remain frozen.
The small diagnostic stays local; results do not depend on a PC assignment.

## Fixed scope and comparisons

The [versioned specification](../../config/studies/adaptation_batch_precision_v1.yaml)
inherits the SHD section of `config/paper_baseline.yaml`. Both saved checkpoints
retain their training configs and parameter/buffer identities. Seed 42 and the
same 40 pilot recordings are fixed; **no outlier is removed**. The original
validation cohort/order, inputs, labels, burn-in blocks and padding masks remain
authenticated against the completed pilot.

| Factor | Conditions | What remains fixed |
|---|---|---|
| Precision | `source_high`; `highest_matmul`; `highest_matmul_cudnn_disabled` | Same CUDA device, checkpoint, data and batch layout |
| Pilot layout | Original batch; natural singleton; singleton retaining original batch padding length | Same physical input, target and supervised block mask |
| Recording 3896 controls | Original repeat; singleton repeat; original-size batch of repeated target copies; original batch reversed | All target input/mask values; target slot is tracked explicitly |

The original companion batch for 3896 has shape **256 × 295 × 140**, with the
target at slot **80** (zero based). Natural singleton input is reconstructed
independently in preflight to verify that slicing the original batch preserves
the pilot's exact input/target/mask contract.

Each precision mode evaluates all **1,631 validation records** with the original
seven batches for both models. Only the two `source_high` checks are mandatory
original-contract anchors: every class prediction must match and saved
aggregate accuracy/CE must pass the **unchanged** original tolerances. Other
precision modes are explicitly diagnostic, not newly selected baselines.

`highest_matmul` disables CUDA matmul TF32 while retaining the preflight cuDNN
TF32 setting. The third mode additionally disables cuDNN TF32. All actual flags
are saved with jobs. Historical evidence recorded `high` precision and disabled
cuDNN benchmarking, but did not exhaustively record every backend flag; the
follow-up records current settings rather than inventing historical ones.
Strict precision is not exact arithmetic and does not guarantee batch
invariance. A mode with better diagnostic agreement is not automatically a new
evaluation protocol or a uniquely proven explanation.

The fixed matrix contains **546 forward jobs**, **9,786 original-batch score
rows**, **720 pilot layout rows** and **24 focus control rows**. Full-batch and
pilot summaries overlap and are not independent observations. Focus controls
address one selected numerical failure, not population robustness.

## Outputs and resume

Results go to `results/experiments/adaptation_batch_precision_v1/`:

- `jobs/`: digest-guarded resumable scores, logits, backend flags and batch
  input/target/block hashes. Original-order companion IDs are in the manifest.
- `shared/pilot/` and `shared/PATH_INDEX.json`: authenticated source metadata
  copies; source manifests are preserved unchanged. They describe a selected
  subset, not a copy of all binary gradient arrays or checkpoints.
- `FULL_VALIDATION_CHECKS.json`: two mandatory source-high anchors and four
  diagnostic comparisons under stricter precision.
- `SCORES.json`/`.csv`, `COMPARISONS.csv`: scalar/logit contrasts across layouts,
  fixed-layout precision, singleton padding and focus controls. The original
  single-record pilot is also checked for reproduction in `SUMMARY.json`.
- `FROZEN_STATE_CHECKS.json`, `SUMMARY.json`/`.md`, `COMPLETE.json`: unchanged
  model-state evidence, descriptive findings and verifiable completion ledger.
- `PROGRESS.json`, `CONTRACT_TESTS.txt`/`.json`: per-run progress and user test log.

Resume with the **same command and output root**. An OS-owned lock prevents
concurrent writers. Changed source/config/environment/backend identities require
a new output root; never relax manifest guards or delete a lock to bypass an
active process. The user needs the exact frozen pilot, source code, checkpoints
and training H5 locally to execute; Git does not transfer ignored weights/data.
Persisted paths are repository relative and resolve against the checkout root.

Agent-safe preflight and saved-output verification:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_batch_precision.run --check-only --device cuda:0
.\.venv\Scripts\python.exe -m scripts.adaptation_batch_precision.run --verify-only --output-root results/experiments/adaptation_batch_precision_v1
```

The second command applies after user completion. Preflight authenticates the
executed pilot, checkpoints/configs, baseline, environment, data and focus input
layouts; it performs **no tests or model forward/backward**. The
[readiness record](../../results/experiments/adaptation_mechanism_study_v1/BATCH_PRECISION_PREFLIGHT.json)
authenticates 57 scientific source files and records no model execution. The
preflight was also checked from a different working directory, as with Hydra;
it is stored outside either frozen experiment root. To run the contracts
separately, the user may use:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_batch_precision
```

Review the saved numerical contrasts before choosing a separate future gradient
protocol or expanding coverage. Stage 7 remains incomplete after this diagnostic;
no mechanistic, causal or superiority claim is an automatic completion gate.
