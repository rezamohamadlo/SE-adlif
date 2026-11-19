# MT/MT-TC clean window evidence diagnostics

Nonoverlapping 5-frame windows (20 ms), after burn-in and excluding synthetic padding. Each sample has equal weight within a model/seed; windows are not pooled as independent observations. Negative MT-TC minus MT JS means smoother adjacent probability evidence, not necessarily correct evidence.

| Seed | Metric | Finite pairs | MT-TC minus MT |
|---|---|---:|---:|
| 42 | adjacent_js_nats | 1631 | -0.00030694 |
| 42 | true_class_concentration | 1631 | -0.00171555 |
| 42 | true_class_mean_probability | 1631 | +0.00045718 |
| 42 | window_accuracy | 1631 | +0.01866179 |
| 42 | top_class_switch_fraction | 1631 | -0.01652424 |
| 123 | adjacent_js_nats | 1631 | +0.00012586 |
| 123 | true_class_concentration | 1631 | +0.00115058 |
| 123 | true_class_mean_probability | 1631 | +0.01389321 |
| 123 | window_accuracy | 1631 | -0.00248919 |
| 123 | top_class_switch_fraction | 1631 | +0.00049945 |
| 456 | adjacent_js_nats | 1631 | +0.00091244 |
| 456 | true_class_concentration | 1631 | +0.00345795 |
| 456 | true_class_mean_probability | 1631 | +0.01955343 |
| 456 | window_accuracy | 1631 | -0.02452343 |
| 456 | top_class_switch_fraction | 1631 | +0.01471116 |

## Across training seeds

Mean and sample SD of per-seed paired differences; no window/sample pooling. JS is in nats; other metrics are dimensionless fractions (accuracy difference .01 = 1 pp). SD is not a confidence interval.

| Metric | Training seeds | Mean MT-TC minus MT | Sample SD |
|---|---:|---:|---:|
| adjacent_js_nats | 3 | +0.00024379 | 0.00061819 |
| true_class_concentration | 3 | +0.00096433 | 0.00259178 |
| true_class_mean_probability | 3 | +0.01130127 | 0.00980843 |
| window_accuracy | 3 | -0.00278361 | 0.02159411 |
| top_class_switch_fraction | 3 | -0.00043788 | 0.01563878 |

Limitations:
- Descriptive only: windows and samples are not independent training seeds.
- Stable low-confidence or consistently incorrect evidence is not stable useful class evidence; inspect true-class mean and window accuracy jointly.
- JS is an output diagnostic in nats, never a prediction-loss penalty here.
- No diagnostic-benefit regression: parent-derived confidence and prediction loss can be mathematically coupled.
- Exploratory after descriptive results; not causal or independent final-test confirmation.
