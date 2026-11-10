# SHD* paired validation pilot

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 12; paired records across all groups: 12.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | clean | 0 | 12 | 0.000569 | 0.001468 | -0.000899 | 100.00% | 100.00% | +0.00 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 1 | complete | 0.000569 ± undefined | 0.001468 ± undefined | -0.000899 ± undefined | 100.00 ± undefined | 100.00 ± undefined | +0.00 ± undefined |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 42 | `mean_rate_hz` | 12 | -0.3850 | - |
| 42 | `fast_slow_contrast` | 12 | +0.1901 | - |
| 42 | `autocorr_tau_ms` | 12 | -0.1199 | - |
| 42 | `active_channel_fraction` | 12 | -0.0212 | - |
| 42 | `duration_ms` | 12 | +0.0078 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.
