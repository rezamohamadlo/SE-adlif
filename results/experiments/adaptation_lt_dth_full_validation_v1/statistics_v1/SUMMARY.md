# Stage 5: exploratory adjusted associations

Samples: 1631; complete cases: 1631; speakers: 10; training seeds: 3.

Outcome = loss(DTH) - loss(LT-DTH), positive favors LT-DTH. Both features enter jointly. Controls: class, speaker, log1p(duration), log1p(activity). Feature coefficients use one original-cohort SD; bootstrap scales are fixed.

Intervals are approximate speaker-cluster percentile sensitivity intervals conditional on three trained model pairs. There are only ten speakers; coverage may be poor. No p-values, formal significance claim, causal confirmation or independent test claim is made.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 97.5% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | -0.023895 | -0.008971 | +0.002772 | -0.010032 +/- 0.013365 | [-0.061780, +0.024167] | inconclusive |
| autocorr_tau_ms | -0.066485 | +0.031369 | -0.006982 | -0.014033 +/- 0.049306 | [-0.053943, +0.008900] | inconclusive |

Two-feature Bonferroni family, nominal 95% family level subject to unreliable small-cluster bootstrap coverage. See JSON for seed-specific intervals and leave-one-speaker-out sensitivity.

## Changes from clean input

Deletion realizations are averaged within sample and seed first. Three interventions form a separate Bonferroni family (nominal 98.3333% individual intervals).

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | +0.009914 | -0.023285 | -0.020724 | -0.011365 +/- 0.018473 | [-0.037136, +0.015769] |
| time_scale_1.1 | +0.013781 | -0.021470 | +0.001337 | -0.002117 +/- 0.017878 | [-0.015867, +0.012959] |
| delete_0.1 | +0.002873 | -0.010478 | -0.009727 | -0.005778 +/- 0.007501 | [-0.013320, -0.000443] |

## Limits and interpretation

- Only ten speaker clusters: percentile intervals may have poor coverage; no formal significance or causal confirmation.
- Intervals condition on three observed trained model pairs and do not generalize over training seeds.
- Three-seed sample SD is not a confidence interval; samples/replicates are not independent training runs.
- Primary two-feature and secondary three-intervention Bonferroni families are separate; no combined familywise guarantee.
- Seed-specific intervals are diagnostic, not a simultaneous six-coefficient testing family.
- Exploratory specification written after descriptive validation results were observed; not preregistration.
- Speaker/class/linear duration/activity adjustment does not remove all sample-difficulty confounding.
- No official-test access, new neural-model inference or training.

The LT-DTH hypothesis concerns dependence on input temporal scale, without assuming a monotonic positive direction. Autocorrelation crossing lag is the input-timescale proxy; fast/slow contrast is an exploratory joint covariate. Both coefficient intervals form the two-feature family. A conditional association does not prove causality, learned-parameter alignment or robustness over training seeds. Negative/inconclusive outcomes are retained.


## Layer-level trace descriptors

These are model/layer properties, not independent sample-level parameters. Milliseconds use the saved input bin width, not neuron dt. No parameter/performance correlation is fitted.

| Model | Layer | Trace | Models (seeds) | E-folding ms, mean +/- sample SD |
|---|---|---|---:|---:|
| DTH_SE_adLIF | l1 | fast | 3 | 37.965 +/- 0.000 |
| DTH_SE_adLIF | l1 | slow | 3 | 397.997 +/- 0.000 |
| DTH_SE_adLIF | l2 | fast | 3 | 37.965 +/- 0.000 |
| DTH_SE_adLIF | l2 | slow | 3 | 397.997 +/- 0.000 |
| LT_DTH_SE_adLIF | l1 | fast | 3 | 52.716 +/- 1.088 |
| LT_DTH_SE_adLIF | l1 | slow | 3 | 300.929 +/- 7.320 |
| LT_DTH_SE_adLIF | l2 | fast | 3 | 147.800 +/- 17.532 |
| LT_DTH_SE_adLIF | l2 | slow | 3 | 507.751 +/- 108.981 |
