# SSC AMT/MT full validation: saved-evidence review

Reviewed 2026-11-12. The user completed evaluation on **one RTX 4090** in
**2,175.671 seconds (36 minutes 16 seconds)**, within the earlier 30–45-minute
planning estimate. This is the recorded launcher execution after initial
preflight, including preprocessing, inference and reporting; it is not model
latency. No results were imported from the completed pilot.

This review inspected existing files and verified their identities and numerical
consistency. It did not regenerate inputs/features, forward a model, train,
fit a statistical model, resample or run tests. See the
[machine-readable review](POST_RUN_REVIEW.json), [completion](COMPLETE.json),
[manifest](MANIFEST.json) and [clean anchors](VALIDATION_ANCHORS.json).

## Verified scope

- All **57 recorded output hashes**, completion/manifest identity, **52 source
  hashes**, **42 cached-value hashes**, six checkpoint hashes and six original
  validation-metric file hashes passed verification.
- All six condition/realization combinations contain **all 9,981 official
  validation IDs in file order**, for both models and all three matching training
  seeds. No subset sampling or loss of sample coverage occurred.
- All six fresh clean anchors match their selected-epoch training validation
  metrics within the frozen tolerances. Maximum absolute discrepancies are
  **2.63e-8 accuracy fraction** and **8.47e-8 loss**. Saved anchor summaries agree
  with clean per-record predictions. No checkpoint was replaced or reselected.
- The **179,658 unique paired CSV records** agree field by field with the saved
  model predictions and shared feature rows. Targets, speaker IDs, input hashes,
  conditions and seeds match. Saved time-scaling totals conserve framed counts;
  deletion totals never exceed clean totals. Physical support and minimum padded
  length remain distinct. These checks did not reconstruct the actual inputs.
- Per-seed report losses, accuracies, benefits, sample counts and scope metadata
  agree with the saved records. Report row counts are 29,943 clean, 89,829
  time-scaling including clean, and 119,772 deletion including clean; these
  overlapping report views must not be added as independent evidence.
- Two clean ACF descriptors, sample IDs **3552 and 7837**, are right-censored
  because no crossing occurs within the defined horizon. They remain in the
  full diagnostic predictions and accuracy/loss calculations. Saved ACF
  correlations use 9,979 finite pairs, while contrast correlations use 9,981.
  Future analysis must record its missingness/censoring rules explicitly.
- No evidence of unit-test execution was supplied. Tests remain unverified;
  successful evaluation does not prove that the prepared tests passed.

## Clean full-validation results

These values come from the saved clean report. Means and sample SD use the
three matching training seeds, not recordings or perturbation realizations.
Accuracy means are percentages; accuracy SD and differences are in percentage
points (pp). This is not a numeric paper-baseline comparison.

| Metric | Mean ± sample SD, n=3 training seeds |
|---|---:|
| MT accuracy | 82.1528% ± 0.2719 pp |
| AMT accuracy | 82.0459% ± 0.1735 pp |
| AMT − MT accuracy | −0.1069 ± 0.3889 pp |
| MT supervised loss | 0.621635 ± 0.016267 |
| AMT supervised loss | 0.625387 ± 0.020949 |
| Loss benefit (MT − AMT) | −0.003752 ± 0.027020 |

| Training seed | MT accuracy | AMT accuracy | AMT − MT |
|---|---:|---:|---:|
| 42 | 81.9958% | 82.2463% | +0.2505 pp |
| 123 | 81.9958% | 81.9457% | −0.0501 pp |
| 456 | 82.4667% | 81.9457% | −0.5210 pp |

AMT is more accurate in one seed and less accurate in two. Its mean clean loss
is slightly higher. These results **do not establish a consistent AMT
improvement**. The pilot's positive average accuracy gap (+0.7813 pp on 256
class-balanced records) does not carry over to the full validation population.
Preserve both scopes and do not substitute the pilot estimate for full coverage.

## Full-validation interventions

Values below are copied from the saved replicate-averaged summaries. Deletion's
three realizations are averaged within each training seed first. Positive
loss-benefit change means AMT's relative loss gap improves against clean; it
does not imply better absolute loss or accuracy. SD is not a confidence interval.

| Condition | Loss-benefit change from clean, mean ± SD | AMT − MT accuracy, mean ± SD |
|---|---:|---:|
| Time scale .9 | +0.007364 ± 0.011155 | +0.1369 ± 0.2005 pp |
| Time scale 1.1 | −0.005873 ± 0.021222 | −0.1336 ± 0.6401 pp |
| Delete 10%, realizations averaged within seed | +0.004651 ± 0.008227 | +0.0390 ± 0.2459 pp |

Each intervention's loss-benefit change has mixed signs across training seeds:
.9 is negative in seed 42, while 1.1 and deletion are negative in seed 123;
1.1 is also negative in seed 456. The pilot's across-seed positive deletion
change therefore does not generalize uniformly to full validation. No formal
significance, causal explanation or consistent robustness advantage is claimed.
Read the [time-scaling](analyses/time_scaling/SUMMARY.md) and
[deletion](analyses/spike_deletion/SUMMARY.md) reports for the per-seed values.

Saved [clean feature associations](analyses/clean/SUMMARY.md) are small marginal
Pearson correlations: fast/slow contrast ranges from about −.0027 to +.0227,
and ACF crossing lag from −.0238 to −.0139 across seeds. These are descriptive,
unadjusted associations, not evidence of a controlled mechanistic effect or
proof of no association. The current workflow has no regression, bootstrap,
formal significance test or confounder-adjusted conclusion.

## Remaining work

The SSC AMT/MT full-validation evaluation is **executed and verified**.
Roadmap stage 6 as a whole remains in progress. Next, prepare a separate
SSC-specific analysis of these stored results with explicit controls,
speaker-aware uncertainty and censoring rules. Do not apply SHD's split or
ten-speaker assumptions; SSC uses official validation, 35 classes and 256
recorded speaker groups. Any method defined after viewing these descriptive
results remains exploratory. The user launches statistical analysis and tests.

Other SSC model pairs, ECG, stage 7 gradients and final synthesis remain
pending. These checkpoints were selected using this validation split; full
coverage is still development evidence, not independent confirmation or an
official-test improvement. No further experiment was launched in this review.
