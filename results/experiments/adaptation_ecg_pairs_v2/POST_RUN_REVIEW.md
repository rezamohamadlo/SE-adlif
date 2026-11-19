# ECG four-pair saved-evidence review

Verified 14 November 2026 using existing files only.

- 96 registered output hashes and 56 current source hashes match.
- All 18 clean anchors passed unchanged tolerances under the recorded high matrix-precision contract.
- User log confirms all 19 contract tests passed; the agent did not rerun them.
- All 72 unique prediction chunks have intact content digests and the frozen 30-record coverage/order.
- Each of the four pair CSVs contains exactly the expected 360 unique record/seed/condition/realization combinations (1440 total).
- Conditions are clean and three realizations of 10% signed-event deletion. No ECG time scaling or feature/statistical study was performed.
- Recorded execution attempt: 415.641 seconds (about 6 minutes 56 seconds), excluding initial preflight; not pure inference latency.

All four bounded pair evaluations are verified. Full ECG mechanism studies remain incomplete: features, aligned time scaling and inferential statistics are pending. Stage 6 remains open. No new inference, tests, fitting, bootstrap or experiment execution was performed during this review.

See [overall summary](SUMMARY.md) and [machine-readable review](POST_RUN_REVIEW.json).
