# SSC MT/MT-TC clean output-window diagnostics

Full official validation: 9981 records, training seeds 42/123/456, six saved checkpoints.
Nonoverlapping windows of 5/10/20 frames (20/40/80 ms), after ten burn-in frames and before synthetic padding.
Positive differences mean MT-TC minus MT. JS and CE use nats; other metrics use fractions (.01 accuracy = 1 pp).
Probability-window accuracy and mean-membrane window accuracy are separate metrics.

| Window (frames) | Metric | Seeds | Mean MT-TC minus MT | Sample SD |
|---:|---|---:|---:|---:|
| 5 | adjacent_js_nats | 3 | -0.00214872 | 0.00568084 |
| 5 | true_class_concentration | 3 | -0.00125619 | 0.00329795 |
| 5 | true_class_mean_probability | 3 | +0.00172436 | 0.00815477 |
| 5 | window_accuracy | 3 | +0.00231066 | 0.00949547 |
| 5 | top_class_switch_fraction | 3 | -0.00159417 | 0.02023620 |
| 5 | window_mean_supervised_ce | 3 | -0.14131081 | 0.26994195 |
| 5 | window_supervised_accuracy | 3 | +0.00233380 | 0.00959338 |
| 10 | adjacent_js_nats | 3 | -0.00340185 | 0.00941889 |
| 10 | true_class_concentration | 3 | -0.00207517 | 0.00549368 |
| 10 | true_class_mean_probability | 3 | +0.00163275 | 0.00792158 |
| 10 | window_accuracy | 3 | +0.00217470 | 0.00934392 |
| 10 | top_class_switch_fraction | 3 | -0.00370242 | 0.03003507 |
| 10 | window_mean_supervised_ce | 3 | -0.13749484 | 0.25865799 |
| 10 | window_supervised_accuracy | 3 | +0.00203946 | 0.00949246 |
| 20 | adjacent_js_nats | 3 | -0.00458390 | 0.01106598 |
| 20 | true_class_concentration | 3 | -0.00313692 | 0.00706038 |
| 20 | true_class_mean_probability | 3 | +0.00147857 | 0.00749128 |
| 20 | window_accuracy | 3 | +0.00124505 | 0.00813720 |
| 20 | top_class_switch_fraction | 3 | -0.00559845 | 0.03956768 |
| 20 | window_mean_supervised_ce | 3 | -0.12928647 | 0.22986532 |
| 20 | window_supervised_accuracy | 3 | +0.00111091 | 0.00838641 |

## Per-seed paired differences

| Seed | Frames | Metric | Finite paired records | MT-TC minus MT |
|---:|---:|---|---:|---:|
| 42 | 5 | adjacent_js_nats | 9981 | -0.00435284 |
| 42 | 5 | true_class_concentration | 9981 | -0.00210377 |
| 42 | 5 | true_class_mean_probability | 9981 | +0.00331717 |
| 42 | 5 | window_accuracy | 9981 | +0.00420624 |
| 42 | 5 | top_class_switch_fraction | 9981 | -0.00635573 |
| 42 | 5 | window_mean_supervised_ce | 9981 | -0.16209060 |
| 42 | 5 | window_supervised_accuracy | 9981 | +0.00415204 |
| 42 | 10 | adjacent_js_nats | 9981 | -0.00757359 |
| 42 | 10 | true_class_concentration | 9981 | -0.00355520 |
| 42 | 10 | true_class_mean_probability | 9981 | +0.00338911 |
| 42 | 10 | window_accuracy | 9981 | +0.00386266 |
| 42 | 10 | top_class_switch_fraction | 9981 | -0.00912033 |
| 42 | 10 | window_mean_supervised_ce | 9981 | -0.15532753 |
| 42 | 10 | window_supervised_accuracy | 9981 | +0.00368609 |
| 42 | 20 | adjacent_js_nats | 9981 | -0.01011557 |
| 42 | 20 | true_class_concentration | 9981 | -0.00457326 |
| 42 | 20 | true_class_mean_probability | 9981 | +0.00351365 |
| 42 | 20 | window_accuracy | 9981 | +0.00348869 |
| 42 | 20 | top_class_switch_fraction | 9981 | -0.00526234 |
| 42 | 20 | window_mean_supervised_ce | 9981 | -0.13776707 |
| 42 | 20 | window_supervised_accuracy | 9981 | +0.00259447 |
| 123 | 5 | adjacent_js_nats | 9981 | -0.00639719 |
| 123 | 5 | true_class_concentration | 9981 | -0.00404762 |
| 123 | 5 | true_class_mean_probability | 9981 | +0.00896521 |
| 123 | 5 | window_accuracy | 9981 | +0.01071536 |
| 123 | 5 | top_class_switch_fraction | 9981 | -0.01902498 |
| 123 | 5 | window_mean_supervised_ce | 9981 | -0.40026235 |
| 123 | 5 | window_supervised_accuracy | 9981 | +0.01088795 |
| 123 | 10 | adjacent_js_nats | 9981 | -0.01001444 |
| 123 | 10 | true_class_concentration | 9981 | -0.00667722 |
| 123 | 10 | true_class_mean_probability | 9981 | +0.00852874 |
| 123 | 10 | window_accuracy | 9981 | +0.01055959 |
| 123 | 10 | top_class_switch_fraction | 9981 | -0.03065977 |
| 123 | 10 | window_mean_supervised_ce | 9981 | -0.38677503 |
| 123 | 10 | window_supervised_accuracy | 9981 | +0.01060088 |
| 123 | 20 | adjacent_js_nats | 9981 | -0.01179336 |
| 123 | 20 | true_class_concentration | 9981 | -0.00936868 |
| 123 | 20 | true_class_mean_probability | 9981 | +0.00774205 |
| 123 | 20 | window_accuracy | 9981 | +0.00802504 |
| 123 | 20 | top_class_switch_fraction | 9981 | -0.04533312 |
| 123 | 20 | window_mean_supervised_ce | 9981 | -0.35479414 |
| 123 | 20 | window_supervised_accuracy | 9981 | +0.00865655 |
| 456 | 5 | adjacent_js_nats | 9981 | +0.00430388 |
| 456 | 5 | true_class_concentration | 9981 | +0.00238283 |
| 456 | 5 | true_class_mean_probability | 9981 | -0.00710930 |
| 456 | 5 | window_accuracy | 9981 | -0.00798961 |
| 456 | 5 | top_class_switch_fraction | 9981 | +0.02059821 |
| 456 | 5 | window_mean_supervised_ce | 9981 | +0.13842052 |
| 456 | 5 | window_supervised_accuracy | 9981 | -0.00803859 |
| 456 | 10 | adjacent_js_nats | 9981 | +0.00738246 |
| 456 | 10 | true_class_concentration | 9981 | +0.00400691 |
| 456 | 10 | true_class_mean_probability | 9981 | -0.00701961 |
| 456 | 10 | window_accuracy | 9981 | -0.00789814 |
| 456 | 10 | top_class_switch_fraction | 9981 | +0.02867285 |
| 456 | 10 | window_mean_supervised_ce | 9981 | +0.12961804 |
| 456 | 10 | window_supervised_accuracy | 9981 | -0.00816859 |
| 456 | 20 | adjacent_js_nats | 9981 | +0.00815722 |
| 456 | 20 | true_class_concentration | 9981 | +0.00453119 |
| 456 | 20 | true_class_mean_probability | 9981 | -0.00681998 |
| 456 | 20 | window_accuracy | 9981 | -0.00777858 |
| 456 | 20 | top_class_switch_fraction | 9981 | +0.03380009 |
| 456 | 20 | window_mean_supervised_ce | 9981 | +0.10470179 |
| 456 | 20 | window_supervised_accuracy | 9981 | -0.00791828 |

## Limits

- Descriptive validation only; windows, speakers and records are not extra independent training seeds.
- Lower adjacent JS or fewer class switches may describe consistently wrong or low-confidence outputs.
- Inspect true-class probability, window accuracy and window CE together; smoother is not automatically better.
- Window CE uses mean raw membrane potentials; probability metrics use mean frame-softmax probabilities.
- Diagnostic windows exclude burn-in/minimum/batch padding, whereas ordinary anchor loss retains its original mask.
- Window widths and training JS overlap/stride are different contracts; JS is never added to this prediction loss.
- No p-values, bootstrap, automatic confidence/benefit regression, TTA or independent-test/causal claim.

All ordinary clean anchors must pass before these summaries are published. Raw per-batch window arrays and scalar CSVs retain explicit undefined reasons; no record is silently discarded.
