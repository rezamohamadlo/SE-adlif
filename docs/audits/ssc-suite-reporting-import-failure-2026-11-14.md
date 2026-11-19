# SSC reporting import collision and preserved-work resume

Current outcome (2026-11-14): the user completed the v2 suite and all 33 tests.
[Post-run review](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md)
verifies all four pairs and the I/O provenance. The failure and repair sequence
below is retained as historical evidence.

Reviewed 2026-11-14. The corrected `suite_v2` user run **passed all 25 contract
tests**, archived the source evidence, completed AMT/MT statistics and produced
all 36 LT-DTH/DTH prediction chunks. It then stopped during report creation with
`AttributeError: module 'statistics' has no attribute 'mean'`. The suite and
LT-DTH evaluation have no completion record; existing predictions are verified
progress evidence, not a completed evaluation report.

## Cause and correction

Direct execution of `scripts/adaptation_ssc_suite/run.py` adds that script's
directory to Python's module search path. The local `statistics.py` can then
be imported as top-level `statistics`, shadowing Python's standard-library
module. The local module provides the SSC statistical job, not `mean` or
`stdev`. The contract tests use package imports through `-m unittest`, so their
launch environment did not expose this direct-script collision.

**Launch the queue as a package module from the repository root:**

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ssc_suite.run --device cuda:0 --output-root results/experiments/adaptation_ssc_suite_v2
```

This resolves the standard-library import correctly while retaining the
existing source bytes, configuration, manifests and cache identities. All
**68 frozen source hashes still match**. No manifest guard is bypassed, no
completed manifest is rewritten, and no new output root is needed. The guide,
project instructions and `command.txt` now specify module launch.

## Verified saved work

- The completed AMT/MT evaluation's **62 registered output hashes** and its
  statistical job's **eight output hashes** pass verification.
- All **36 LT-DTH/DTH cached prediction values** pass their content hashes,
  checkpoint identities, model/seed/condition checks and ordered 9,981-record
  coverage checks. Six trace-parameter records match their checkpoints.
- The paired CSV already contains **179,658 rows**. It was written before
  report creation failed; its file checksum is retained for comparison.
- [Failure review](../../results/experiments/adaptation_ssc_suite_v2/reporting_failure_review/REVIEW.json)
  records these checks and the original progress/test/manifest hashes. Exact
  copies of those records are retained in its `original_records/` directory
  before the user's resume can update live status or rerun contract tests.
- Import-only preflight confirms the reporter resolves the Python standard
  library and exposes `mean` and `stdev`. Actual package-module `--check-only`
  preflight also passes the 18-checkpoint/config, data and statistical-design
  checks against the **existing v2 manifests**.

The agent performed saved-file and preflight checks only. It did not rerun
tests, model forward passes, fits, resampling or report generation.

## What the user resume does

The same v2 directory retains completed AMT/MT jobs and all LT-DTH prediction
chunks. The queue reruns its small contract-test step, verifies completed jobs,
reads cached LT-DTH predictions and rebuilds its paired views/reports. It should
perform no new LT-DTH/DTH inference while those validated chunks remain intact.
LT-DTH statistics and the SA-MT/MT and MT-TC/MT jobs then proceed normally.

The failed session recorded 4,086.765 seconds before reporting stopped; its
progress record is archived. Elapsed times written after resume describe the
resumed invocation and may exclude earlier cached inference. They must not
be reported as total evaluation time or pure model latency. Retain both
session records when reviewing costs.

Report generation and the remaining jobs still await this user command.
Successful preflight does not establish completed reports or scientific
conclusions. ECG, stage 7 gradients and direct SSC output-window stability
remain separate work, as specified in the [suite guide](../experiments/adaptation-ssc-suite-v1.md).
