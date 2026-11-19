# SSC exploratory statistics: LT_DTH_SE_adLIF versus DTH_SE_adLIF

Records: 9981; joint-regression complete cases: 9979; speakers: 256; fixed training seeds: 3.

Outcome = loss(DTH_SE_adLIF) - loss(LT_DTH_SE_adLIF); positive favors the variant. Both standardized clean-input features enter jointly, controlling class, speaker, log1p(duration) and log1p(rate). Speaker effects use within-speaker demeaning. Scales remain fixed during resampling.

Hypothesis: `exploratory_two_sided_input_timescale_associations`. Inspect per-seed signs alongside every interval. Negative and inconclusive outcomes are retained. These exploratory intervals are conditional on the existing trained models; they do not establish formal significance, causal effects or robustness across new training seeds.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 99.375% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | -0.010212 | -0.008612 | -0.008817 | -0.009214 +/- 0.000870 | [-0.048210, +0.030794] | inconclusive |
| autocorr_tau_ms | +0.026195 | -0.017752 | -0.027009 | -0.006189 +/- 0.028425 | [-0.032503, +0.016609] | inconclusive |

Primary intervals use the eight-coefficient family across four pairs. Per-seed intervals in JSON are diagnostic only. SD is variation across the three training seeds, not a confidence interval.

## Changes from clean input

Deletion realizations are averaged within recording and seed first. All 9,981 records enter these effects, including the two censored ACF records. Approximate 99.5833% individual intervals use the separate twelve-effect family.

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | -0.032313 | -0.000234 | +0.006093 | -0.008818 +/- 0.020592 | [-0.023859, +0.005816] |
| time_scale_1.1 | +0.005554 | +0.003595 | +0.033119 | +0.014089 +/- 0.016510 | [+0.002457, +0.025773] |
| delete_0.1 | +0.007688 | -0.008955 | -0.005822 | -0.002363 +/- 0.008844 | [-0.011064, +0.005775] |

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
