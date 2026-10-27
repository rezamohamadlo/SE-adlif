# Review of extended-seed commit 6deeccd

Reviewed against the original attached implementation request and repository
`AGENTS.md`. The review covers the runner/Hydra entry point, validation/test
boundaries, checkpoint and CSV evidence, provenance, import aggregation and
the paired statistics. No existing experiment evidence was changed.

## Findings corrected

1. **High: continuation trusted the manifest without checking checkpoint identity.**
   A matching `FULL_RUN_CONFIG.json` and any `last.ckpt` made a run resumable,
   even with incompatible checkpoint hyperparameters, a missing selected best
   checkpoint, or missing source provenance. Preflight now compares checkpoint
   configs with the resolved experiment (ignoring only the resume routing field),
   verifies the saved validation selection and retained best checkpoint, and
   checks provenance before launching. Local completion checks also verify the
   selected epoch, full-budget last epoch and selection score against checkpoints.

2. **High: test-contaminated or overlapping CSV histories could be accepted.**
   Completion read only `val_acc` rows, accepting nonempty test columns and
   repeated validation epochs. New regression tests demonstrated both failures.
   Completion, resume and import checks now reject test artifacts/metrics,
   nonfinite/out-of-range scores, invalid epochs and duplicate/non-monotonic
   validation histories. Resume rejects validation history ahead of the retained
   checkpoint; it preserves evidence instead of silently truncating it.

3. **Medium: checkpoint callback paths were not portable.**
   The saved Lightning callback retained absolute checkout paths. Validation-only
   checkpoints now serialize callback paths through `project_paths.py` and resolve
   them against the destination checkout on load. A non-training test saves a real
   `ModelCheckpoint` state, moves it to a different checkout and verifies callback
   restoration. Validation child commands also use a relative Hydra output path.
   Ordinary full-test checkpoint behavior remains unchanged.

4. **Medium: generated resume instructions dropped study flags.**
   `TRAINING_SUMMARY.md` omitted `--validation-only` and `--seed-cohort` from its
   generated command. It now retains both; a regression test renders the summary
   and checks the command. The separate guide's commands remain valid.

5. **Medium: provenance and baseline checks were incomplete.**
   A Git diff hash missed changes to untracked source modules. Launch records now
   add a normalized source/config file fingerprint, and continuation preflight
   checks it. Old environment records are preserved when adding the new field.
   Baseline validation now checks the versioned baseline file rather than trusting
   a baseline embedded in the same config that it is validating. Aggregate JSON
   retains completed runs' configs, baseline identity, source directories and
   environment records; Markdown lists source commits.

The original `trainer.test` call is guarded by `run_test`, and runner-generated
validation-only commands explicitly set it false. The separate runner branch
does not call test-cost collection or write `completed.txt`. SHD* validation uses
the training-derived split with dataset split seed 42. No model, optimizer,
preprocessing, loss or training-budget settings were changed by this review.

## Statistical review

The original implementation correctly pairs only matching completed seeds and
rejects duplicate imported model/dataset/seed records. ECG differences are SA-MT
minus MT; SHD* differences are MT minus DTH. Differences are calculated before
rounding, in percentage points. The sample SD and two-sided 95% Student t interval
use the correct `n-1` degrees of freedom. Fewer than two pairs produce no SD or CI.
These calculations were retained. The documented normality, independence,
fixed-split and multiple-comparison caveats remain applicable. No test means are
constructed from validation-only scores.

## Verification and limits

Four regression tests first reproduced unsafe acceptance of test CSV columns,
repeated validation epochs, incompatible checkpoint configs and missing resume
provenance. Additional tests cover changed source, missing best checkpoints,
checkpoint/CSV disagreement, moved checkpoints, generated resume instructions,
baseline tampering, incomplete imports, aggregate provenance and untracked source.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_extended_seeds tests.test_full_runner tests.test_runner_costs tests.test_runner_ssc_mean tests.test_mt_consistency_runner tests.test_paper_config tests.test_project_paths
```

Result: all 61 tests passed.

Both actual PC dry-run commands resolve 14 jobs without creating the study output
directory. Implementation/documentation whitespace checks pass. Tests use temporary
metadata fixtures and synthetic tensors; no fitting, optimizer training steps or
official-test evaluation was performed. Real GPU training and end-to-end interrupted
training remain user-launched checks. Evidence-only aggregation cannot verify the
contents of omitted checkpoint files. Older absolute-path checkpoints still need
explicit path aliases when transferred to an unregistered checkout root.

Review fixes are recorded separately from commit `6deeccd`, without rewriting it.
Publication requires explicit user confirmation under `AGENTS.md`.
