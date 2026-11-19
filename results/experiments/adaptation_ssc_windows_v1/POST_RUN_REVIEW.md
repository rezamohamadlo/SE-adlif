# SSC direct output-window saved-evidence review

Verified 16 November 2026 using existing files only.

- All 256 registered output hashes and 79 current source hashes match.
- User log confirms all 12 tests passed; the agent did not rerun tests.
- Six clean anchors cover all 9981 records each. All 59886 ordinary predictions match their frozen reference classes exactly; accuracy/loss deltas remain within unchanged tolerances.
- All 234 chunk content digests and exact batch membership/order are valid.
- 179658 model/seed/record/width descriptor rows and 89829 paired CSV rows match cached scalar descriptors; retained window arrays reproduce their stored means/accuracies.
- Diagnostic support excludes ten burn-in frames and minimum/batch padding; windows are nonoverlapping 5/10/20 frames (20/40/80 ms), with explicit tail/undefined accounting.
- Recorded execute() attempt: 380.953 seconds (about 6 minutes 21 seconds), excluding initial preflight. This is not pure inference latency.

## Interpretation

At every width, seeds 42 and 123 give lower MT-TC adjacent JS and window CE and higher mean-membrane window accuracy; seed 456 reverses these directions. The across-seed average therefore does not establish consistent superiority. Confidence, class alignment and CE must be interpreted jointly.

Windows/records are not independent training seeds. This diagnostic has no p-values, bootstrap, confidence-benefit regression, TTA, causal or independent-test claim. Original SSC/SHD/ECG evidence remains unchanged.

The direct SSC window diagnostic is executed and verified. The remaining step is the read-only study/code scope audit before deciding stage-6 closure; stage 7 has not started.

See [summary](SUMMARY.md) and [machine-readable review](POST_RUN_REVIEW.json).

Subsequent closeout: the [read-only study/code scope audit](../../../docs/audits/supervisor-stage6-closeout-2026-11-16.md) is complete. It records scoped stage-6 completion and readiness to prepare stage 7, without rewriting frozen experiment records or claiming upstream figure parity.
