# SSC exploratory statistics: SA_MT_SE_adLIF versus MT_SE_adLIF

Records: 9981; joint-regression complete cases: 9979; speakers: 256; fixed training seeds: 3.

Outcome = loss(MT_SE_adLIF) - loss(SA_MT_SE_adLIF); positive favors the variant. Both standardized clean-input features enter jointly, controlling class, speaker, log1p(duration) and log1p(rate). Speaker effects use within-speaker demeaning. Scales remain fixed during resampling.

Hypothesis: `positive_adjusted_fast_slow_association`. Inspect per-seed signs alongside every interval. Negative and inconclusive outcomes are retained. These exploratory intervals are conditional on the existing trained models; they do not establish formal significance, causal effects or robustness across new training seeds.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 99.375% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | -0.003337 | -0.011134 | +0.016431 | +0.000653 +/- 0.014209 | [-0.032253, +0.034704] | inconclusive |
| autocorr_tau_ms | +0.021111 | +0.045577 | +0.008130 | +0.024939 +/- 0.019015 | [-0.002811, +0.052444] | inconclusive |

Primary intervals use the eight-coefficient family across four pairs. Per-seed intervals in JSON are diagnostic only. SD is variation across the three training seeds, not a confidence interval.

## Changes from clean input

Deletion realizations are averaged within recording and seed first. All 9,981 records enter these effects, including the two censored ACF records. Approximate 99.5833% individual intervals use the separate twelve-effect family.

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | +0.012075 | -0.015411 | -0.003727 | -0.002354 +/- 0.013794 | [-0.013502, +0.009156] |
| time_scale_1.1 | -0.014687 | -0.004666 | -0.007782 | -0.009045 +/- 0.005128 | [-0.020826, +0.002160] |
| delete_0.1 | +0.004728 | +0.000732 | -0.016002 | -0.003514 +/- 0.010998 | [-0.011806, +0.004787] |

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
