# Single-command SSC mechanism suite v1

**Completed by the user and verified on 2026-11-14.** All four SSC model-pair
evaluations and four statistical jobs finished. The user passed 25 original
contract tests and eight I/O contracts. Read-only review verified 230 core
outputs, I/O provenance, 108 unique prediction chunks and 718,632 paired rows.
See the [post-run review](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md)
for clean-accuracy tables, conditional uncertainty, timing scopes and limits.
The source specification remains v1 and the completed execution root is v2.
Earlier short-path, import and I/O failure histories remain preserved; the
compatibility launcher is the canonical saved-output verifier. No rerun is needed.
The completed [AMT/MT full-validation evaluation](../../results/experiments/adaptation_ssc_amt_full_validation_v1/POST_RUN_REVIEW.md)
is verified input evidence, not a job to rerun. This workflow addresses the four
SSC model-pair evaluations/statistics within roadmap stage 6. ECG and stage 7
gradients remain separate work.

## One user command

Run from the repository root in the existing project environment:

```powershell
# PC1: user launches the whole foreground SSC queue on the local GPU.
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --device cuda:0 --output-root results/experiments/adaptation_ssc_suite_v2
```

Launch with `-m` from the repository root to retain correct package imports.
The compatibility entrypoint records its own source/test hashes and I/O policy,
runs eight atomic I/O contracts, then starts the frozen queue with scoped atomic
JSON-write retries. It changes no scientific configuration or source manifest;
completed predictions/jobs remain reusable. Persistent write failures still stop
the queue without deleting destinations or bypassing guards.

This runs the selected SSC contract tests, copies the existing source evidence
into a portable snapshot, analyzes AMT/MT, then evaluates and analyzes LT-DTH/DTH,
SA-MT/MT and MT-TC/MT in order. It ends with a consolidated summary and package
verification. The queue never trains or opens official-test H5 data. Both
existing runner training workflows and completed experiment code are unchanged.

Budget **roughly 1–2 hours on a comparable idle RTX 4090**, based on the prior
36-minute two-model/six-condition evaluation and the additional required models.
This is an unmeasured queue estimate; statistical resampling, disk speed and
hardware affect the actual duration. The source snapshot is approximately
676 MB, and the completed package will require additional space for new
predictions and paired records. No timing benchmark or snapshot copy was run
by the agent. Execution remains visible in the user's terminal, with batch
progress during inference.

```powershell
# Agent-allowed preflight: disposable same-volume I/O probes and input/design checks.
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --check-only --output-root results/experiments/adaptation_ssc_suite_v2

# Verify the entire saved suite after completion, without repeating jobs.
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --verify-only --output-root results/experiments/adaptation_ssc_suite_v2

# Optional separate user test invocation; the launch command already runs these.
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ssc tests.test_adaptation_ssc_full tests.test_adaptation_ssc_suite
```

Compatibility preflight writes and removes only disposable metadata probes
under `io_failure_review/`, to check same-volume atomic replacement and identical
JSON serialization. It does not touch scientific files, run tests, copy the
source snapshot, forward a model, extract features, fit regressions or resample. It loads and
strictly checks trusted local checkpoint states, authenticates saved evidence,
checks the three-record loader contract and builds the complete-case design
for eligibility/rank checks only. The observed preflight cohort is 9,981 records,
9,979 regression-eligible records and 256 speaker groups. Outcome-dependent
checks for future model pairs are deferred until their evaluations finish;
they are not reported as already verified.

## Frozen plan and dependencies

The [suite definition](../../config/studies/adaptation_ssc_suite_v1.yaml) inherits
the SSC section of `config/paper_baseline.yaml`, freezes 18 checkpoint identities
and the source AMT/MT completion/manifest hashes, and records all four model
pairs and statistical settings. The [entrypoint](../../scripts/adaptation_ssc_suite/run.py)
and helpers live outside previous experiment source folders so their historical
source guards remain valid.

| Order | Work | Required gate |
|---|---|---|
| 1 | SSC input/integrity/queue contract tests | All selected tests pass; save log and status |
| 2 | Archive exact AMT/MT source evidence and index its relocated paths | All original registered hashes pass |
| 3 | AMT/MT statistics | Reused evaluation identities, sample pairing and design eligibility pass |
| 4 | LT-DTH/DTH full evaluation, then statistics | All six clean anchors pass before perturbation inference; completed evaluation verified before statistics |
| 5 | SA-MT/MT full evaluation, then statistics | Same gates; reuse existing MT predictions |
| 6 | MT-TC/MT full evaluation, then statistics | Same gates; preserve training-objective differences and exclude auxiliary JS from comparison loss |
| 7 | Consolidation and package verification | All four pairs complete; shared resampling weights/exclusions agree |

Each comparison uses training seeds 42/123/456, all 9,981 official validation
records in file order, and the same six conditions: clean, time scales .9/1.1,
and 10% event deletion with three deterministic realizations. There is no
subsampling or parameter search. The fixed SSC protocol remains batch 256,
four-ms bins, minimum 300 frames, ten-frame burn-in and mean-membrane loss.
Training metadata retains the SSC 40-epoch, LR .006 baseline. SHD holdout,
300-epoch training and sum-softmax settings are not imported.

The queue reuses six authenticated input-feature chunks for every pair, and
the three MT seeds' 18 condition chunks for both SA-MT and MT-TC. The complete
AMT/MT evaluation is indexed from its exact copied source package. This adds
no AMT/MT inference and avoids new MT inference in the other comparisons.
New inference covers DTH, LT-DTH, SA-MT and MT-TC (72 condition/model/seed chunks).
Every new or reused prediction is checked against the shared input metadata.
Reused MT outputs remain the same observations across comparisons.

For each new pair, the clean condition supplies its six compatibility anchors;
it is not evaluated twice. All anchors must reproduce saved selected-epoch
validation accuracy/loss within the frozen 1e-6 fraction/.001 loss tolerances
before perturbation conditions begin. Failures stop the queue. Preserve the
evidence and inspect framing, masks, aggregation and saved settings; do not
substitute checkpoints, relax guards or retrain to repair evaluation failures.

MT-TC retains `ssc_mean_loss_v1_mt_temporal_consistency_v1` and its recorded
weight .1/window 5 training deviation. Its comparison uses supervised
`mean_membrane_potentials` only. This suite's input-feature analysis does not
directly test stability of neighboring output windows; that diagnostic remains
separate. LT-DTH/DTH stores learned/fixed trace decays and e-folding times in
frames and milliseconds, at model/layer level only. These constants are not
treated as thousands of independent per-record observations.

## SSC statistical contract

The methods are **exploratory**, defined after seeing descriptive AMT results.
They do not establish preregistration, independent confirmation or causal effects.

- Joint clean-input regressors are fast/slow contrast and ACF crossing lag.
  Controls are 35-class fixed effects, speaker fixed effects, log1p(duration)
  and log1p(mean activity rate). Within-speaker demeaning implements speaker
  effects. Features and continuous controls use original complete-case population
  SDs, fixed during resampling. Fit the three seed outcomes separately with the
  same design; do not pool repeated predictions as independent trained models.
- Raw SSC speaker strings map to stable integer codes solely for the existing
  numerical helper. Manifests preserve the complete code-to-string map, and
  leave-one-speaker-out results include the original IDs. Full-cohort and
  complete-case speaker counts are recorded separately.
- Joint regression explicitly excludes invalid or censored feature records,
  with IDs/reasons saved. The source currently has two right-censored ACF
  descriptors (3552 and 7837); no horizon substitution or imputation occurs.
  Predictions and intervention changes retain all 9,981 recordings. This is
  complete-case association analysis, with the corresponding selection limit.
- Draw 2,000 speaker-pairs bootstrap resamples using the fixed seed and all
  256 original speaker groups. Sufficient statistics retain class/continuous
  slopes and within-speaker effects. The same draws preserve pairing across
  training seeds and model comparisons. Reject rank-deficient draws; require
  at least 95% valid draws or stop without reporting intervals. Use bounded BLAS
  threading for this small-matrix calculation. No additional regression is
  run in agent preflight.
- The primary family contains eight feature/pair coefficients (two features
  times four pairs); the secondary family contains twelve intervention/pair
  effects (three interventions times four pairs). Separate Bonferroni-adjusted
  approximate percentile intervals use alpha .05: nominal 99.375% and 99.5833%
  individual intervals, respectively. These do not give a combined-family
  guarantee or a formal significance claim. Two thousand draws give limited
  precision at these tail quantiles. Per-seed intervals are diagnostics only.
- Intervention outcomes are transformed loss benefit minus clean loss benefit.
  Deletion realizations are averaged within recording/seed before speaker
  resampling. Coefficients/effects retain per-seed values, mean and sample SD,
  leave-one-speaker-out sensitivity and eligibility diagnostics. Intervals
  condition on the three existing trained model pairs and do not represent
  variability over new training seeds.

There is no difficulty-control term constructed from the parent loss that also
defines the outcome, no test-set selection, cross-dataset loss pooling or automatic
winner selection. Negative and inconclusive results are retained. The shared
AMT pilot's positive subset gap must not replace the full-validation result.

## Artifacts, interruption and portability

The scientific specification remains v1. The **corrected execution root** is
`results/experiments/adaptation_ssc_suite_v2/`, independent of PC assignment.
Its explicit command-line override keeps the failed default `suite_v1` root
and source-hash guard intact:

```text
MANIFEST.json
PROGRESS.json
CONTRACT_TESTS.json
contract_tests.log
shared/amt_mt_full_validation/    # exact historical source outputs + COMPLETE
    PATH_INDEX.json              # old recorded paths -> copied files
pairs/amt_mt/evaluation/          # index of reused evidence, no artificial rerun
pairs/amt_mt/statistics_v1/
pairs/lt_dth/{evaluation,statistics_v1}/
pairs/sa_mt/{evaluation,statistics_v1}/
pairs/mt_tc/{evaluation,statistics_v1}/
SUMMARY.json
SUMMARY.md
COMPLETE.json
```

Snapshot manifests, completion records and cache identities are copied
byte-for-byte. The path index resolves their historical paths without rewriting
them. New pair manifests explicitly distinguish local predictions from reused
observations. Each statistical job registers the exact evaluation completion it
consumes. The compatibility entrypoint additionally retains per-attempt policy,
source/test hashes, retry events and eight-test status/log under `io_compatibility/`;
`IO_COMPATIBILITY_COMPLETE.json` binds successful I/O provenance to the final
scientific completion hash without rewriting it. All result dependencies live inside the complete suite package; the
pair directories alone are not sufficient transfer units.

The user repeats the **same launch command** after interruption. Completed
inference chunks and completed evaluation/statistics jobs are hash-verified and
reused. Selected contract tests rerun for an incomplete queue. An interrupted
inference chunk or unfinished statistical job restarts; a partially completed
bootstrap is not appended. Copying can resume through verified files and atomic
temporary copies. Once the source snapshot is complete, resume can use it
without the original external evaluation directory. Repeating a fully completed
queue verifies it instead of rerunning tests, inference or fitting.

OS-owned locks protect the suite and each job. Do not bypass active locks by
deleting their files or run another writer in the same root. A failed check
records the failing phase in `PROGRESS.json` and stops downstream jobs. Code,
configuration or environment changes trigger manifest guards; preserve the old
package and use a new explicit `--output-root results/experiments/<new-name>`
only as a distinct study execution. This does not bypass source compatibility
rules: reused software/GPU contracts must still match.

Paths resolve through `project_paths.py` against the checkout root, not the shell
or Hydra working directory. Git does not transfer ignored datasets or weights.
The required validation H5 and exact checkpoint files must exist for preflight
and new inference. Transferring the **entire suite root under its relative
layout** preserves read-only result verification without those datasets/weights
or the external source directory. Resume additionally requires the recorded
software, source code, validation data and checkpoint artifacts. The external
compatibility entrypoint is scoped to JSON destinations inside the selected
suite, retries only Windows replacement errors 5/32/33 with bounded delays, and
never deletes or overwrites destinations in place.

Agent verification covers genuine 18-checkpoint/data/config preflight, the
three-record loader contract, authenticated existing AMT/MT rows, 9,979-case
design eligibility, syntax and portable path/link checks. It does **not** execute
the corrected queue's contract tests, source snapshot copies, neural inference,
regression, bootstrap or plots. The v2 user run has documented passing evidence
for all 25 original contract tests and eight I/O contracts. All four pairs
completed; read-only post-run review checked saved hashes, coverage, pairing,
anchors, exclusions, estimate-array dimensions and shared speaker weights.
The agent did not rerun those tests or scientific jobs. ECG readiness is the
next main comparison step; this queue does not complete gradients, direct SSC
output-window stability or all of roadmap stage 6.
