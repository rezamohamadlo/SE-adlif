# SSC queue contract-test failure and path correction

Current outcome (2026-11-14): the user completed the v2 suite and all 33 tests.
[Post-run review](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md)
verifies all four pairs and the I/O provenance. The failure and repair sequence
below is retained as historical evidence.

Later status: the v2 retry passed the contract tests and reached LT-DTH
reporting. See the [reporting import correction](ssc-suite-reporting-import-failure-2026-11-14.md)
for the current package-module resume command. The original failure/retry
sequence below is retained as historical evidence.

Reviewed 2026-11-14. The user's first SSC queue attempt stopped at
`contract_tests`: **25 tests ran, with two errors**. No source-evidence snapshot,
new neural inference, regression or bootstrap job started. There is no suite
`COMPLETE.json` or scientific summary. This is a failed prelaunch check, not a
completed experiment or evidence against a model.

Evidence is preserved in the original result directory:

- [Queue status](../../results/experiments/adaptation_ssc_suite_v1/PROGRESS.json).
- [Test status](../../results/experiments/adaptation_ssc_suite_v1/CONTRACT_TESTS.json)
  and [complete log](../../results/experiments/adaptation_ssc_suite_v1/contract_tests.log).
- [Failure review and evidence hashes](../../results/experiments/adaptation_ssc_suite_v1/prelaunch_failure_review/REVIEW.json).
- [Archived failing snapshot code](../../results/experiments/adaptation_ssc_suite_v1/prelaunch_failure_review/source_snapshot/scripts/adaptation_ssc_suite/common.py)
  and [archived test definitions](../../results/experiments/adaptation_ssc_suite_v1/prelaunch_failure_review/source_snapshot/tests/test_adaptation_ssc_suite.py).

## Cause and correction

Both errored tests exercise exact snapshot preservation. Windows supplies the
temporary directory using an 8.3 short-name path on this machine. The snapshot
writer stored that spelling in `PATH_INDEX.json`, while its validator resolved
the same destination to its long-name spelling. Strict string comparison then
rejected two references to the same physical artifact. A read-only path preflight
confirmed the spelling difference and identical resolved artifact identity.

The snapshot destination now resolves against the project path helper **before
copying or constructing the index**. Both writer and validator therefore use
the same physical spelling. The existing checksum and exact-index guards remain
strict; tests for changed snapshots are retained. The preservation test also
requires canonical destination/index paths. Main preflight now includes a
Windows temporary-directory path-only check without creating a probe directory
or copying evidence.

The original source files were archived with checksums before editing. Failed
logs, status, manifest and existing completed AMT/MT studies were not rewritten.
This changes path serialization, not the SSC data, model, checkpoint, split,
loss, statistical specification or training behavior. The scientific suite
specification stays at v1; the corrected execution uses a fresh result root:

```text
results/experiments/adaptation_ssc_suite_v2/
```

The old root records different source hashes, so its manifest must reject an
in-place launch with the corrected code. A fresh explicit `--output-root`
preserves that guard and the failed attempt's evidence.

## User retry

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_ssc_suite/run.py --device cuda:0 --output-root results/experiments/adaptation_ssc_suite_v2
```

The command reruns the 25 contract tests before any scientific job. If they
pass, it proceeds through the previously prepared SSC queue and reuses the
verified AMT/MT evidence. No model training is needed. After an interruption
within this corrected execution, repeat this exact command, retaining its
output-root argument.

The agent has run path/syntax and input/design preflight only. It has **not**
rerun the tests, copied the scientific source bundle, launched the queue or
claimed test success. Corrected test execution and all new scientific jobs
remain pending user launch. See the [suite guide](../experiments/adaptation-ssc-suite-v1.md)
for the unchanged evaluation/statistical contracts and scope limits.
