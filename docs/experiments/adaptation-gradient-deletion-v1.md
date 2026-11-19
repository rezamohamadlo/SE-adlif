# Strict-gradient matched event-deletion pilot

Completed and reviewed 16 November 2026 under the user's explicit permission for
agent experiment execution through stage-7 completion. The
[saved review](../../results/experiments/adaptation_gradient_deletion_v1/POST_RUN_REVIEW.md)
verifies **557 outputs, 19 tests, 80 fresh clean reference checks, 160 shared
deletion inputs, 400 model evaluations and 200 paired rows**. No training or new
backward was performed; natural repeat-0 arrays came from the verified strict
gradient study. Model state stayed unchanged.

The [frozen specification](../../config/studies/adaptation_gradient_deletion_v1.yaml)
uses the same 40 SHD* records and MT/AMT seed-42 checkpoints at `highest_matmul`.
It applies a 1% exact event-deletion budget, MT absolute-gradient guidance on
existing event cells and three exact uniform random controls. Both models see
the same input for every condition. Targets, physical length and padding masks
are unchanged. All 80 clean checks pass before any deletions are generated.

Guidance with no active-event gradient is explicitly ineligible for both models;
clean/random controls remain. The same eligible cohort is used for guided/random
contrasts. Random scalar effects average within record, not as independent seeds.
Signed finite CE changes and each model's own surrogate linear predictions are
retained, without requiring their signs/magnitudes to agree. No regression,
confidence interval, significance test or superiority claim is produced.

Reproduce/resume only when no writer owns the output lock:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_deletion.run --device cuda:0 --output-root results/experiments/adaptation_gradient_deletion_v1
```

For ordinary review, use the read-only verifier:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_deletion.run --verify-only --output-root results/experiments/adaptation_gradient_deletion_v1
```

Results include bound raw deletion arrays, portable source snapshots/path index,
clean checks, paired/effect CSVs, a per-record figure and a completion ledger.
Recording 3896 is retained. Version-derived random draws differ from older
source-high controls, so their contrast is not a matched precision experiment.
The executed workflow took 163.859 seconds excluding initial preflight; this is
workflow time, not pure inference latency.

The [broader registered stage-7 suite](adaptation-stage7-suite-v1.md) now extends
comparable gradient and constrained-input analyses to all four pairs, three
datasets and three training seeds. This completed pilot alone does not close
stage 7 or alter original source-high baselines.
