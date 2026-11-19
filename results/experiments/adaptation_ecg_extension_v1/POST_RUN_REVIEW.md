# ECG extension saved-evidence review

Verified 16 November 2026 using existing files only.

- 163 registered output hashes and 64 current frozen source hashes match.
- The user log confirms 34 tests passed. The agent did not rerun tests or scientific execution.
- All 180 feature rows and 108 prediction chunks have intact content digests and exact coverage.
- Input, target, block, mask and time-map identities match across saved feature/prediction/paired files; all 2160 paired loss-benefit values agree with the underlying cached predictions.
- 72 completed chunks were preserved/reused, with only 36 new mapped chunks. All 18 reused clean anchors passed.
- Each pair has 540 records over all 30 validation rows, three training seeds and six conditions.
- All four descriptive designs use 30 complete cases, no exclusions, rank 9 and 21 residual degrees of freedom. Constant duration was dropped explicitly; all 30 leave-one-record-out designs remain estimable.
- No feature ACF values are undefined or censored in any of the six conditions.
- Original total events: 13545. At 0.9/1.1 resampling: 12201/14889. These are joint input/label resampling effects, not count-preserving duration interventions.
- Recorded execution attempt: 83.891 seconds (about 1 minute 24 seconds), excluding initial preflight; not pure inference latency.

## Interpretation boundary

All four ECG evaluation/feature/descriptive-analysis packages are executed and verified. This is not patient-population inference: no verified patient IDs, confidence intervals or p-values exist. Feature times use model steps. Results and coefficient signs vary across seeds; no causal or independent-test improvement claim follows.

Direct SSC window diagnostics and the outstanding study/code scope audit remain to be reviewed before closing stage 6. Stage 7 has not started.

See [overall summary](SUMMARY.md) and [machine-readable verification](POST_RUN_REVIEW.json).
