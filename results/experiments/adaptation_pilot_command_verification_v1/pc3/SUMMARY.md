# SHD* paired validation pilot

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 12; paired records across all groups: 48.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 456 | clean | 0 | 12 | 0.002487 | 0.001702 | +0.000785 | 100.00% | 100.00% | +0.00 pp |
| 456 | delete_0.1 | 0 | 12 | 0.014791 | 0.001590 | +0.013201 | 100.00% | 100.00% | +0.00 pp |
| 456 | delete_0.1 | 1 | 12 | 0.027258 | 0.020441 | +0.006817 | 100.00% | 100.00% | +0.00 pp |
| 456 | delete_0.1 | 2 | 12 | 0.189081 | 0.014212 | +0.174869 | 91.67% | 100.00% | +8.33 pp |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 456 | `mean_rate_hz` | 12 | +0.3875 | - |
| 456 | `fast_slow_contrast` | 12 | -0.1893 | - |
| 456 | `autocorr_tau_ms` | 12 | +0.1213 | - |
| 456 | `active_channel_fraction` | 12 | +0.0198 | - |
| 456 | `duration_ms` | 12 | -0.0087 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in `SUMMARY.json` metadata.
