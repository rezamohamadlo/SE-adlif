# SSC resume: Windows denied atomic status-file replacement

Current outcome (2026-11-14): the user completed the v2 suite and all 33 tests.
[Post-run review](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md)
verifies all four pairs and the I/O provenance. The failure and repair sequence
below is retained as historical evidence.

Reviewed 2026-11-14. The package-mode user resume passed all 25 original contract
tests, verified the source snapshot and completed AMT/MT statistics, then stopped
in cached LT-DTH evaluation when `os.replace(PROGRESS.json.tmp, PROGRESS.json)`
raised **PermissionError / WinError 5**. The failed resumed session recorded
53.485 seconds. Neither the suite nor LT-DTH has a completion record.

The destination and old temporary file are not read-only and report write
access. Windows nevertheless denied replacement. A temporary sharing or scan
lock is plausible; no locking process was identified, so no specific editor or
scanner is asserted as the cause. Existing permissions were not changed and
no process was stopped.

## Preserved progress

[Failure review](../../results/experiments/adaptation_ssc_suite_v2/io_failure_review/REVIEW.json)
records the failure, old progress/test/manifest hashes and exact copies in
`original_records/`, including the uncommitted `PROGRESS.json.tmp`. The previous
reporting-failure archive remains intact. All 36 cached LT-DTH/DTH prediction
files and the 179,658-row paired CSV still match the earlier preserved file
checksums. Completed AMT/MT records are retained. No training or repeated
LT-DTH inference is needed to repair this I/O failure.

## Compatibility resume launcher

```powershell
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --device cuda:0 --output-root results/experiments/adaptation_ssc_suite_v2
```

The [launcher](../../scripts/resume_ssc_suite.py) and
[I/O helper](../../scripts/ssc_resume_io.py) live outside the frozen experiment
source folders. All **68 frozen source hashes remain unchanged**. The existing
queue, result manifests, checkpoint identities, statistical specifications and
completion guards retain their original identities.

For JSON destinations **inside the selected suite only**, the launcher supplies
an execution-time I/O compatibility writer before the queue/report modules are
imported. It uses the original JSON options, UTF-8 and native newline behavior,
but creates a unique closed temporary file beside the destination. Atomic
replacement retries only Windows error codes 5, 32 and 33, with 12 attempts and
bounded delays (roughly eight seconds total sleeping per blocked write).
It does not delete the destination, rewrite it in place, change ACLs, or silently
skip failed writes. Other errors fail immediately; persistent blocking stops
the queue and retains the uncommitted temporary payload.

The scope and compatibility source/test hashes are recorded under
`io_compatibility/<attempt>/PROVENANCE.json`. These are explicit execution
addenda, not changes to frozen scientific manifests. Eight additional atomic
I/O contract tests run from the user command before the original queue starts;
their log/status are retained in that attempt directory. The original 25 queue
tests still run in their own process. No compatibility tests were run by the
agent. After successful queue completion, `IO_COMPATIBILITY_COMPLETE.json`
binds the I/O provenance to the original suite completion hash. The scientific
completion file is not rewritten to add compatibility metadata.

## Checks and scope limits

```powershell
# Agent-allowed preflight: disposable I/O probes plus existing input/design checks.
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --check-only --output-root results/experiments/adaptation_ssc_suite_v2

# Verify core evidence and available compatibility completion after the user run.
.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite --verify-only --output-root results/experiments/adaptation_ssc_suite_v2
```

Preflight passed an actual same-volume atomic replacement probe and exact JSON
serialization comparison, touching only disposable metadata under the I/O
review directory. It also passed frozen-source integrity and the original
18-checkpoint/config, data and 9,979-complete-case/256-speaker design checks.
The probe does not prove that a transient or persistent lock on a live status
file has disappeared. Runtime retries remain bounded and persistent permission
problems are reported without bypassing evidence guards.

The agent did not rerun contract tests, predictions, reports, fits, bootstrap
or the scientific queue. The user resume will reuse completed AMT/MT jobs and
existing LT-DTH prediction chunks, regenerate pending views/reports, then
continue LT-DTH statistics and SA-MT/MT-TC evaluation/statistics. Partial and
resumed elapsed times are separate session evidence, not total model latency.
Stage 6 remains incomplete until all required evidence is produced and reviewed;
ECG, gradients and direct SSC output-window stability remain separate work.
