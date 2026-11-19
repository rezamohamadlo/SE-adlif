# One-command four-pair ECG validation queue

**Completed by the user and verified, 14 November 2026:** all four clean/deletion
pair evaluations passed. The [saved-evidence review](../../results/experiments/adaptation_ecg_pairs_v2/POST_RUN_REVIEW.md)
verified 96 output hashes, 56 source hashes, 18 passing clean anchors, 19 passing
user tests, 72 prediction chunks and 1440 paired rows. Recorded queue execution
was 415.641 seconds, excluding initial preflight. Features, time scaling and
inferential statistics remain pending; full ECG mechanism studies are incomplete.
The preparation notes below describe the launch and verification workflow.

Prepared 14 November 2026. Agent checks are limited to preflight. All tests,
checkpoint inference, paired reductions and report generation run only after
the user launches the command below. No training is involved.

The [v2 preflight record](../../results/experiments/adaptation_mechanism_study_v1/ECG_PAIRS_V2_PREFLIGHT.json)
passed for all 18 checkpoints, the fixed validation loader and the recorded
`high` numerical contract. It explicitly records `clean_anchors_verified: false`
and `tests_run: false`; numerical reproduction awaits user execution.

This queue covers all four requested **model pairs** across seeds 42/123/456:

| Pair | Parent | Variant |
|---|---|---|
| AMT/MT | MT_SE_adLIF | AMT_SE_adLIF |
| LT-DTH/DTH | DTH_SE_adLIF | LT_DTH_SE_adLIF |
| SA-MT/MT | MT_SE_adLIF | SA_MT_SE_adLIF |
| MT-TC/MT | MT_SE_adLIF | MT_TC_SE_adLIF |

It evaluates all 30 saved validation records with clean input and three shared
10% signed-event deletion realizations. It does **not** add ECG input features,
time scaling, regression or bootstrap; those are separate pending protocol work.
Finishing four pair evaluations does not finish four full mechanism studies or
stage 6. It does not access the official test set.

## Why there is a new output root

The [v1 failure review](../audits/ecg-clean-anchor-precision-2026-11-14.md)
records the failed MT seed-42 anchor. The original training and measurement
entrypoints set `torch.set_float32_matmul_precision('high')`; the pilot omitted
that setting. This queue explicitly restores and records it before model
construction. Preflight compares it with all 18 saved measurement environments.

The precision omission is a confirmed implementation mismatch. Whether it fully
explains the numerical failure must be verified by the user's clean-anchor run.
Accuracy and loss tolerances are **unchanged** (1e-6 fraction and .001 CE).
No automatic precision search, checkpoint substitution, label-mask change or
tolerance widening is used.

The failed v1 directory and all 48 frozen source files remain unchanged. The
new specification freezes hashes for its four evidence files, and execution
copies them byte-for-byte into `reviews/failed_alignment_pilot_v1/` with a
portable path index. Its failed predictions are never current prediction caches.
Preflight can use the copied review evidence on subsequent resumes, so the
new output package is self-contained with respect to this failure history.

## Command

From the repository root, on the local RTX 4090:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_suite.run --device cuda:0 --output-root results/experiments/adaptation_ecg_pairs_v2
```

The same command resumes compatible cached predictions. Do not launch two
writers against this root. An OS-owned lock prevents concurrent launchers.

The queue runs, in order:

1. Artifact/config/split/numerical-contract preflight.
2. The 11 original alignment contracts plus eight new queue contracts, launched
   in a child process by this user command.
3. Preservation of the failed pilot evidence and path index.
4. Clean validation of **all 18 unique model/seed checkpoints**. Each is compared
   with its saved validation-selected epoch. The review is saved after every
   anchor; individual metric mismatches are collected across the 18 anchors.
5. A strict gate: **all 18 anchors must pass before any deletion inference**.
   If the setting correction is insufficient, `CLEAN_ANCHOR_REVIEW.json`
   records every accuracy/loss difference and execution stops.
6. All four pair evaluations and descriptive reports, sharing parent predictions.
7. Hash registration and verification of the finished evaluation package.

Expected coverage is **72 unique prediction chunks** (18 model/seed combinations
× four conditions), 360 paired rows per pair, and 1440 paired rows overall.
MT is evaluated once per seed/condition and reused across its three pairs.
Labels, temporal blocks, polarity and the original all-zero-vector-to-class-0
argmax semantics remain unchanged. MT-TC is compared using supervised CE only,
without its training-time auxiliary JS penalty.

The observed first v1 clean chunk took about six seconds from the end of its
test log to cache publication, including model load and I/O. Extrapolating
72 chunks suggests roughly **8–15 minutes** locally, including startup/tests;
this is a scheduling estimate, not a benchmark or a measured queue duration.
No multi-PC split is needed for this bounded queue.

## Results and scientific limits

Outputs under `results/experiments/adaptation_ecg_pairs_v2/` are independent of
PC assignment:

- `MANIFEST.json`, `CONTRACT_TESTS.json`/`.txt`, `PROGRESS.json`;
- `CLEAN_ANCHOR_REVIEW.json` with the strict 18-anchor gate;
- `predictions/<model>/seed_<seed>/<condition>/rep_<rep>.json`;
- `PREDICTION_INDEX.json`, shared across all pairs;
- `pairs/<pair>/paired_records.csv`, `SUMMARY.json` and `SUMMARY.md`;
- overall `SUMMARY.json`/`.md` and `COMPLETE.json`.

Both record-mean and step-weighted metrics are recorded. Descriptive group
tables average deletion realizations inside each training seed, then report
mean and sample SD across the three training seeds. They do not count repeated
realizations or time steps as independent training seeds or patients. The MAT
does not provide verified patient IDs, so there is no patient-cluster bootstrap,
significance claim or independent-test conclusion.

Agent-safe preflight:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_suite.run --check-only --device cuda:0
```

Read-only verification after successful execution:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_suite.run --verify-only --output-root results/experiments/adaptation_ecg_pairs_v2
```

The next scientific work after reviewing these results remains ECG feature
definition, label-aligned time scaling and a defensible statistical protocol.
