# SSC paired full validation: MT_TC_SE_adLIF versus MT_SE_adLIF / clean

**Full official SSC validation: 9981 records per condition.** Clean results also supply the saved compatibility anchors. These are exploratory validation diagnostics, with the original class frequencies. They are not independent confirmation or final-test results.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 9981; paired records across all groups: 29943.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 9981 | 0.640417 | 0.607480 | +0.032937 | 82.00% | 82.23% | +0.23 pp |
| 42 | clean | 0 | 9981 | 0.612064 | 0.603137 | +0.008927 | 82.00% | 82.21% | +0.21 pp |
| 456 | clean | 0 | 9981 | 0.612423 | 0.635688 | -0.023264 | 82.47% | 81.91% | -0.56 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.621635 ± 0.016267 | 0.615435 ± 0.017673 | +0.006200 ± 0.028200 | 82.15 ± 0.27 | 82.11 ± 0.18 | -0.04 ± 0.45 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 9981 | -0.0065 | - |
| 123 | `fast_slow_contrast` | 9981 | +0.0156 | - |
| 123 | `autocorr_tau_ms` | 9979 | -0.0155 | - |
| 123 | `active_channel_fraction` | 9981 | -0.0293 | - |
| 123 | `duration_ms` | 9981 | -0.0143 | - |
| 42 | `mean_rate_hz` | 9981 | +0.0083 | - |
| 42 | `fast_slow_contrast` | 9981 | -0.0089 | - |
| 42 | `autocorr_tau_ms` | 9979 | -0.0156 | - |
| 42 | `active_channel_fraction` | 9981 | -0.0045 | - |
| 42 | `duration_ms` | 9981 | -0.0315 | - |
| 456 | `mean_rate_hz` | 9981 | -0.0216 | - |
| 456 | `fast_slow_contrast` | 9981 | +0.0159 | - |
| 456 | `autocorr_tau_ms` | 9979 | +0.0054 | - |
| 456 | `active_channel_fraction` | 9981 | +0.0161 | - |
| 456 | `duration_ms` | 9981 | +0.0275 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors MT_TC_SE_adLIF relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
