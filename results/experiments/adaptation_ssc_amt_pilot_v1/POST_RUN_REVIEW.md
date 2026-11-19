# SSC AMT/MT pilot: saved-evidence review

Reviewed 2026-11-12. The user completed the pilot on an RTX 4090 in
**412.344 seconds (6 minutes 52 seconds)**. This is the launcher-recorded
execution time, not model latency or the duration of standalone startup preflight.
It supersedes the earlier unmeasured 10–20-minute planning estimate for this run.

The review read saved evidence and checked its integrity and numerical
consistency. It did not rerun inference, transformations, feature extraction,
regression, bootstrap or tests. See [machine-readable checks](POST_RUN_REVIEW.json),
[completion record](COMPLETE.json), [manifest](MANIFEST.json) and
[anchor records](VALIDATION_ANCHORS.json).

## Verification

- All **63 recorded output hashes**, completion/manifest identity, 47 current
  source hashes and 48 cached-value hashes passed.
- All six local checkpoint hashes and original validation-metric file hashes
  match their frozen records. No checkpoint was replaced or reselected.
- Each full clean anchor has all **9,981 official validation records in order**.
  Recorded accuracy/loss agree with the saved selected-epoch metrics; maximum
  absolute discrepancies are 2.63e-8 (accuracy fraction) and 8.47e-8 (loss),
  well within the frozen 1e-6/.001 tolerances. Anchor summaries also agree with
  their saved per-record predictions.
- Each pilot feature/prediction chunk contains the same 256 fixed sample IDs.
  Targets, speaker IDs, input hashes and shared feature metadata match across
  models and seeds. Saved count totals are conserved under time scaling and
  nonincreasing under deletion; physical support remains separate from padding.
  Input arrays were not regenerated during this review.
- All **4,608 unique paired CSV records** match the saved prediction pairs.
  Per-seed summary loss/accuracy/benefit values agree with those records.
  All three reports identify the pilot subset and complete three-seed cohort.
- No test-run evidence was supplied. Unit-test execution remains unverified;
  successful experiment completion does not establish that tests passed.

## Full clean validation anchors

These are compatibility anchors on the official validation split, distinct
from the subset diagnostics below. Values are percentages per training seed.

| Training seed | MT validation accuracy | AMT validation accuracy |
|---|---:|---:|
| 42 | 81.9958% | 82.2463% |
| 123 | 81.9958% | 81.9457% |
| 456 | 82.4667% | 81.9457% |

AMT is higher in one training seed and lower in two. These anchors do not
establish a consistent accuracy improvement. They reuse validation data already
involved in checkpoint selection and are not independent final-test evidence.

## Pilot findings: 256 fixed validation records

The following values are copied from the saved summaries. Means and sample SDs
use the three matching training seeds; accuracy SDs and differences are in
percentage points. There is no numeric paper-baseline comparison here.

| Clean-input pilot metric | Saved mean ± sample SD |
|---|---:|
| MT accuracy | 75.2604% ± 0.9831 pp |
| AMT accuracy | 76.0417% ± 0.9021 pp |
| AMT − MT accuracy | +0.7813 ± 1.7901 pp |
| MT supervised loss | 0.867422 ± 0.069641 |
| AMT supervised loss | 0.932238 ± 0.062160 |
| Loss benefit (MT − AMT) | −0.064817 ± 0.097210 |

The subset's slightly higher average AMT accuracy accompanies **higher average
AMT loss**. Its accuracy and loss findings therefore do not support an
unqualified improvement claim. The class-balanced subset also differs from the
full-validation population; it must not replace the full clean anchors.

| Pilot condition | Loss-benefit change from clean | AMT − MT accuracy |
|---|---:|---:|
| Time scale .9 | +0.058594 ± 0.045239 | −0.1302 ± 2.9319 pp |
| Time scale 1.1 | +0.024310 ± 0.048572 | −1.0417 ± 0.2255 pp |
| Delete 10% (three realizations averaged within seed) | +0.033952 ± 0.005283 | −0.0434 ± 2.5260 pp |

Positive loss-benefit change means AMT's *relative loss gap* improves against
clean input; it does not mean its absolute loss or accuracy improves. Time
compression increases both models' mean loss in the saved pilot summaries.
The .9 change is positive in all three seeds, the 1.1 change has mixed signs,
and deletion's replicate-averaged change is positive in all three. These are
descriptive subset observations; SD is not a confidence interval and no
significance or causal claim follows. Perturbation realizations are not extra
independent training seeds. Inspect the [clean](analyses/clean/SUMMARY.md),
[time-scaling](analyses/time_scaling/SUMMARY.md) and
[deletion](analyses/spike_deletion/SUMMARY.md) reports for per-seed values and
descriptive feature associations.

## Next step and remaining scope

The bounded SSC pilot is executed and reviewed. **Roadmap stage 6 remains in
progress**: prepare a separate versioned full-validation SSC AMT/MT diagnostic
study covering all 9,981 validation records under every condition, with its own
output root and runtime/scheduling assessment. Preserve this pilot unchanged.
Such expansion remains exploratory validation work, not independent confirmation.
Other SSC model pairs, statistical uncertainty analysis, ECG, stage 7 gradients
and the final synthesis remain pending. No further experiment was launched.
