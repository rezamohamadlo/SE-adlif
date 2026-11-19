# Fixed-precision SHD* input-gradient repeat pilot

**User execution completed and reviewed 16 November 2026.** The
[post-run review](../../results/experiments/adaptation_gradient_strict_v1/POST_RUN_REVIEW.md)
verifies 456 output hashes, 64 scientific source hashes, 17 passed user tests,
both same-precision full-validation references and 164 finite nonzero physical
gradients. All 86 required repeat/padding checks and all autograd/reference
score checks passed; model state stayed unchanged. Recorded execute time was
216.313 seconds, excluding initial preflight. The agent inspected saved evidence
only. The [matched strict-precision deletion controls](adaptation-gradient-deletion-v1.md)
have since been executed and verified. The
[registered three-dataset stage-7 suite](adaptation-stage7-suite-v1.md) has since
completed its registered exploratory scope and passed
[independent review](../../results/experiments/adaptation_stage7_suite_v1/POST_RUN_REVIEW.md)
on 17 November 2026. Stage 8 synthesis and cost/activity reporting remain next.

The preparation protocol below was recorded 16 November 2026. All experiment
execution, including tests and forward/backward, is user-launched.
This is a separate diagnostic protocol following the
[numerical review](../../results/experiments/adaptation_batch_precision_v1/POST_RUN_REVIEW.md).
The completed source-high gradient pilot and its baseline scores remain frozen.

For read-only verification of completed outputs:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_strict.run --verify-only --output-root results/experiments/adaptation_gradient_strict_v1
```

The original execution/resume command is retained for provenance. Run from the
repository root on the same local CUDA PC:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_strict.run --device cuda:0 --output-root results/experiments/adaptation_gradient_strict_v1
```

The command runs **17 user contracts**, including an analytical toy CE backward,
then two same-precision full-validation checks and **164 input backward passes**.
It saves stability findings and stops for review. There is no training,
fine-tuning, optimizer, official-test access, event deletion or input optimization.
This bounded repeat study stays on the same local PC; artifacts are PC independent.

## Why a separate reference

The [versioned specification](../../config/studies/adaptation_gradient_strict_v1.yaml)
inherits the SHD section of `config/paper_baseline.yaml`. Saved configs,
checkpoints, validation-selected epochs, training seeds and split remain
unchanged. Only the diagnostic numerical evaluation protocol changes:
`highest_matmul` uses `highest` float32 matmul precision with CUDA matmul TF32
disabled, keeping cuDNN settings identical to the authenticated numerical
reference. Actual flags are saved in manifests/caches and restored on exit.

The earlier numerical study found batch-layout prediction agreement at this
precision, but **three MT and five AMT full-validation predictions changed from
source-high**. These strict references are diagnostic evidence, never a
replacement for training-validation selection, the paper comparison, or original
source-high anchors. Precision is fixed for stability assessment, not selected
for higher accuracy.

Before any backward, both models must reproduce all **1,631 validation records**
in their original batch order under the saved `highest_matmul` reference. The
input/target/block hashes and every prediction must match. Aggregate accuracy
and CE differences must meet the registered 1e-6 fraction / .001 CE limits.
An anchor failure preserves chunks and blocks backward. No source-high
tolerance or historical result is changed.

## Repeat and padding scope

- Same MT/AMT checkpoints, seed 42 and 40 pilot records, including recording
  **3896**. There is no new outcome-based selection or exclusion.
- Two independent natural-singleton backward passes per record/model:
  **160 gradients**, forming **80 repeat pairs**.
- Two additional original-padded-singleton passes for recording 3896 per
  model: **four gradients**, **two padded repeat pairs** and **four natural
  versus padded comparisons**. The original padded length is 295 frames;
  input values, labels and supervised block masks are authenticated.
- **80 descriptive comparisons** of strict repeat 0 with the exact preserved
  source-high natural-singleton gradient arrays. These compare different
  precision protocols and are excluded from strict repeat pass/fail criteria.

Each backward creates a fresh floating input leaf through ordinary
`MLPSNN.forward`, differentiating only the saved per-record supervised CE.
Weights remain frozen in eval mode. Parameter/buffer hashes and empty parameter
gradients are checked after each pass; nonfinite or disconnected input gradients
are hard errors. Signed raw arrays retain all stored frames; physical input
includes burn-in, while synthetic-padding mass is summarized separately.

## Explicit numerical criteria

Every gradient records clean `no_grad` scores and a same-mode autograd forward.
Both prediction equality and the CE bound below are evaluated. The same check
compares clean scores with the saved strict-singleton reference and compares
repeat/padding scores. Unlike the earlier diagnostic's recorded-but-ungated
cross-run tolerance, these fields explicitly contribute to the saved stability
outcome.

| Check | Registered criterion |
|---|---|
| Score agreement | Same prediction; absolute CE difference ≤ `2e-5 + 1e-5 × abs(reference CE)` |
| Gradient repeat/padding size | Physical-support L2 difference / larger physical L2 norm ≤ `0.01` |
| Gradient repeat/padding direction | Physical-support cosine ≥ `0.999` when both norms are nonzero |
| Both physical maps zero | Explicit `both_zero`; cosine undefined; stable zero, not evidence of sensitivity |
| Only one physical map zero | Explicit `one_zero`; fails repeat stability; cosine undefined |

The new score bound is a diagnostic engineering criterion, informed by the
previous focus repeat/order drift of about 1.01e-5 CE. Gradient limits are
proposed tooling thresholds, not calibrated significance bounds or proof of
hard-spike derivatives. They do not modify original anchor tolerances. Physical
gradient comparisons exclude synthetic padding and include burn-in inputs;
normalization never replaces a zero map with fabricated values.

**Negative stability findings are saved rather than suppressed or converted
into success.** `COMPLETE.json` authenticates finished execution even if
`stability_criteria_passed` is false. Required checks include 80 natural repeat,
two padded repeat and four padding-layout checks, plus all 164 autograd/reference
score checks. Cross-precision differences are descriptive. No perturbation queue
launches after either outcome; manual evidence review comes next.

## Artifacts, resume and checks

Results are under `results/experiments/adaptation_gradient_strict_v1/`:

- `anchors/<model>/seed_42/`: seven resumable batches per model, followed by
  `FULL_VALIDATION_ANCHORS.json` for the same-mode checks.
- `gradients/<model>/seed_42/<layout>/`: 164 signed float32 NPZ arrays and bound
  JSON sidecars, loaded with `allow_pickle=False`; index and summaries at root.
- `STABILITY_CHECKS.json`/`.csv`: 86 required repeat/padding comparisons plus
  80 descriptive cross-precision comparisons, with explicit score bounds and
  zero-map status.
- `SUMMARY.json`/`.md`, `FROZEN_STATE_CHECKS.json`, `COMPLETE.json`: outcome,
  unchanged state evidence and frozen output ledger.
- `shared/`: authenticated selected strict reference jobs/metadata and all 80
  legacy gradient arrays, with an explicit portable path index. Historical
  manifests remain unchanged and describe their original complete packages.
- `PROGRESS.json`, `CONTRACT_TESTS.json`/`.txt`: run progress and user test log.

Resume with the **same command and output root**. Compatible caches resume
without redoing published backward passes; an orphan NPZ can restore its missing
sidecar from embedded metadata. A missing array with an existing sidecar fails
instead of generating replacement evidence. An OS-owned lock prevents concurrent
writers. Changed code/spec/environment requires a new output root.

Use repository-relative paths; loading resolves against the checkout root,
including from a different working directory. Executing on another checkout
requires the exact source packages, checkpoint files and training H5 locally;
Git alone does not transfer ignored datasets/weights. Shared result subsets
have their own verifiable hashes and path index.

Agent-safe commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_strict.run --check-only --device cuda:0
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_strict.run --verify-only --output-root results/experiments/adaptation_gradient_strict_v1
```

The second applies after user completion. Preflight authenticates the two
completed studies, scientific source hashes, saved checkpoints/configs,
environment/backend settings, training H5/split, strict references and 40 input
layouts. It executes no tests, model forward/backward or stability calculations.
Readiness is recorded separately in
[STRICT_GRADIENT_PREFLIGHT.json](../../results/experiments/adaptation_mechanism_study_v1/STRICT_GRADIENT_PREFLIGHT.json).
It authenticates 64 scientific source files. Preflight also passed from a
different working directory, as with Hydra.
The user may run contracts separately with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_gradient_strict
```

This evaluates surrogate-gradient reproducibility, not causal interpretation or
architectural superiority. Forty records and one seed are still a pilot;
full-validation **anchor coverage does not imply full-validation gradient
coverage**. Stage 7 remains incomplete. Review stability before preparing new
matched perturbations or expanding pairs/seeds/datasets.
