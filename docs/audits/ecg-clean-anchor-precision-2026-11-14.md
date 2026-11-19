# ECG clean-anchor failure and precision-contract correction

**Post-run update:** the user completed the v2 queue. Under the explicitly matched
`high` contract, all 18 clean anchors passed the unchanged tolerances; all four
clean/deletion pair evaluations completed. The [read-only verification](../../results/experiments/adaptation_ecg_pairs_v2/POST_RUN_REVIEW.md)
confirmed 96 output hashes and 19 passing user tests. The failed v1 run remains
historical evidence. This resolves the clean-reproduction gate for the recorded
v2 run; features, time scaling and inferential analysis remain pending.

Review date: 14 November 2026. This review reads existing outputs; it performs
no model inference, loss recomputation from logits, tests or experiment runs.

The user-launched [v1 pilot](../experiments/adaptation-ecg-alignment-pilot-v1.md)
passed all 11 contract tests and stopped at its first clean anchor:

| MT_SE_adLIF, seed 42, selected epoch 398 | Saved validation | User-produced v1 cache |
|---|---:|---:|
| Accuracy (%) | 86.24647856 | 85.72124007 |
| CE | 0.37247619 | 0.39137601 |

The logged differences are −0.005252384858469794 accuracy fraction
(−0.52523849 percentage points) and +0.01889982238015009 CE. Deletion did not
start. The 18 checkpoint prerequisite messages confirm artifact/config checks;
they do not mean 18 clean validations succeeded.

Existing evidence is retained unchanged in
`results/experiments/adaptation_ecg_alignment_pilot_v1/`: `MANIFEST.json`,
`CONTRACT_TESTS.txt`, `PROGRESS.json` and the MT seed-42 clean cache.
The [new specification](../../config/studies/adaptation_ecg_pairs_v2.yaml)
records all four file hashes. None of those failed predictions is reused as a
valid clean anchor. The frozen v1 source hashes still match their 48 files.

Inspection found a concrete numerical-contract discrepancy: `run.py` and
`measure_run_costs.py` explicitly set float32 matrix precision to `high`, while
the pilot does not establish that setting. All measurement records checked by
the new preflight must record `high`. This can matter for recurrent thresholded
spiking dynamics. The existing v1 environment record does not record actual
matrix precision, so this review does not claim to have measured its value or
prove that precision alone caused every differing prediction.

The isolated [four-pair queue](../experiments/adaptation-ecg-four-pair-queue-v2.md)
sets and records the original `high` contract. It leaves the main runners,
shared models, original pilot code, checkpoints, labels and masks unchanged.
Its new root is `results/experiments/adaptation_ecg_pairs_v2/`.

**Verification boundary:** agent preflight establishes artifacts, configs,
backend settings and loader tensors only. The correction remains unverified
numerically until the user runs all 18 clean anchors. The queue records every
anchor and refuses all deletion work unless every unchanged tolerance passes.
If the mismatch remains, review `CLEAN_ANCHOR_REVIEW.json`; do not widen
tolerances, pick a different checkpoint or label the failed anchor complete.

Four pair evaluations are prepared; features, aligned time scaling, inferential
statistics and full ECG mechanism-study completion remain pending.

The [new preflight record](../../results/experiments/adaptation_mechanism_study_v1/ECG_PAIRS_V2_PREFLIGHT.json)
confirms all 18 saved measurement environments use `high` and matches the
checkpoint/data prerequisites. This is not a successful clean-anchor result.
