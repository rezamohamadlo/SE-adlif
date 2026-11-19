# Stage 5: exploratory adjusted associations

Samples: 1631; complete cases: 1631; speakers: 10; training seeds: 3.

Outcome = loss(MT) - loss(AMT), positive favors AMT. Both features enter jointly. Controls: class, speaker, log1p(duration), log1p(activity). Feature coefficients use one original-cohort SD; bootstrap scales are fixed.

Intervals are approximate speaker-cluster percentile sensitivity intervals conditional on three trained model pairs. There are only ten speakers; coverage may be poor. No p-values, formal significance claim, causal confirmation or independent test claim is made.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 97.5% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | -0.007492 | -0.046105 | +0.067889 | +0.004764 +/- 0.057977 | [-0.026625, +0.054401] | inconclusive |
| autocorr_tau_ms | -0.023785 | +0.092327 | +0.011765 | +0.026769 +/- 0.059492 | [+0.003042, +0.063770] | positive_within_fixed_cohort |

Two-feature Bonferroni family, nominal 95% family level subject to unreliable small-cluster bootstrap coverage. See JSON for seed-specific intervals and leave-one-speaker-out sensitivity.

## Changes from clean input

Deletion realizations are averaged within sample and seed first. Three interventions form a separate Bonferroni family (nominal 98.3333% individual intervals).

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | +0.003497 | -0.019201 | -0.010480 | -0.008728 +/- 0.011450 | [-0.032091, +0.007219] |
| time_scale_1.1 | +0.002333 | -0.026362 | +0.030372 | +0.002114 +/- 0.028368 | [-0.021878, +0.026025] |
| delete_0.1 | +0.005339 | -0.022455 | +0.021545 | +0.001476 +/- 0.022253 | [-0.006908, +0.011398] |

## Limits and interpretation

- Only ten speaker clusters: percentile intervals may have poor coverage; no formal significance or causal confirmation.
- Intervals condition on three observed trained model pairs and do not generalize over training seeds.
- Three-seed sample SD is not a confidence interval; samples/replicates are not independent training runs.
- Primary two-feature and secondary three-intervention Bonferroni families are separate; no combined familywise guarantee.
- Seed-specific intervals are diagnostic, not a simultaneous six-coefficient testing family.
- Exploratory specification written after descriptive validation results were observed; not preregistration.
- Speaker/class/linear duration/activity adjustment does not remove all sample-difficulty confounding.
- No official-test access, new neural-model inference or training.

The fast/slow hypothesis predicts a positive adjusted coefficient. Inspect both seed consistency and leave-one-speaker-out ranges; a conditional interval direction is not evidence of population-wide training-seed robustness. Autocorrelation is exploratory, with no directional AMT mechanism specified. Negative or inconclusive outcomes are retained.
