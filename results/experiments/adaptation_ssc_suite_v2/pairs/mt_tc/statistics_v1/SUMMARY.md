# SSC exploratory statistics: MT_TC_SE_adLIF versus MT_SE_adLIF

Records: 9981; joint-regression complete cases: 9979; speakers: 256; fixed training seeds: 3.

Outcome = loss(MT_SE_adLIF) - loss(MT_TC_SE_adLIF); positive favors the variant. Both standardized clean-input features enter jointly, controlling class, speaker, log1p(duration) and log1p(rate). Speaker effects use within-speaker demeaning. Scales remain fixed during resampling.

Hypothesis: `exploratory_input_features_only_no_direct_window_stability_claim`. Inspect per-seed signs alongside every interval. Negative and inconclusive outcomes are retained. These exploratory intervals are conditional on the existing trained models; they do not establish formal significance, causal effects or robustness across new training seeds.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 99.375% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | +0.003780 | -0.000463 | +0.041994 | +0.015104 +/- 0.023384 | [-0.023361, +0.050295] | inconclusive |
| autocorr_tau_ms | +0.004503 | -0.005827 | -0.013329 | -0.004884 +/- 0.008953 | [-0.029624, +0.021098] | inconclusive |

Primary intervals use the eight-coefficient family across four pairs. Per-seed intervals in JSON are diagnostic only. SD is variation across the three training seeds, not a confidence interval.

## Changes from clean input

Deletion realizations are averaged within recording and seed first. All 9,981 records enter these effects, including the two censored ACF records. Approximate 99.5833% individual intervals use the separate twelve-effect family.

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | +0.005669 | -0.012159 | +0.023896 | +0.005802 +/- 0.018028 | [-0.006674, +0.017803] |
| time_scale_1.1 | +0.000444 | -0.005895 | -0.014992 | -0.006814 +/- 0.007759 | [-0.016839, +0.004795] |
| delete_0.1 | +0.012045 | -0.006948 | -0.009251 | -0.001385 +/- 0.011688 | [-0.009116, +0.005744] |

Positive change means the variant improves its relative loss gap against clean input; absolute loss/accuracy need not improve.

## Limits

- Exploratory SSC validation analysis defined after observing AMT descriptive results; not preregistration or independent confirmation.
- Intervals condition on three observed trained model pairs; bootstrap speakers and repeated predictions are not new training seeds.
- Speaker-pairs percentile intervals are approximate; 2000 draws provide limited precision for multiplicity-adjusted tail quantiles.
- Eight feature/pair coefficients and twelve intervention/pair effects form separate Bonferroni families; no combined guarantee.
- Per-seed intervals and leave-one-speaker-out estimates are diagnostics, not an additional simultaneous testing family.
- Class and speaker fixed effects plus linear log-duration/activity adjustment do not remove all difficulty confounding.
- Joint regression excludes explicitly recorded invalid/censored features without imputation; intervention changes retain all recordings.
- Shared MT predictions across comparisons are identical observations and must not be counted as independent replication.
- MT-TC uses a distinct training objective; evaluation benefit excludes its auxiliary JS. Input features do not directly measure output-window stability.
- LT-DTH trace descriptors are model/layer constants, not independent per-record measurements; no trace-parameter mediation regression.
- No official-test access, new neural inference, training or causal confirmation in this statistical job.

Original speaker strings and code mappings are retained in the manifest. Exclusions describe regression eligibility only; no diagnostic predictions were silently removed.
