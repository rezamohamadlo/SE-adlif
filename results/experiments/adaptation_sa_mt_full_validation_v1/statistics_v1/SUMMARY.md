# Stage 5: exploratory adjusted associations

Samples: 1631; complete cases: 1631; speakers: 10; training seeds: 3.

Outcome = loss(MT) - loss(SA-MT), positive favors SA-MT. Both features enter jointly. Controls: class, speaker, log1p(duration), log1p(activity). Feature coefficients use one original-cohort SD; bootstrap scales are fixed.

Intervals are approximate speaker-cluster percentile sensitivity intervals conditional on three trained model pairs. There are only ten speakers; coverage may be poor. No p-values, formal significance claim, causal confirmation or independent test claim is made.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 97.5% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | -0.035902 | -0.033081 | +0.009992 | -0.019664 +/- 0.025721 | [-0.047620, +0.001216] | inconclusive |
| autocorr_tau_ms | +0.096856 | +0.039093 | -0.038937 | +0.032337 +/- 0.068148 | [+0.005389, +0.078847] | positive_within_fixed_cohort |

Two-feature Bonferroni family, nominal 95% family level subject to unreliable small-cluster bootstrap coverage. See JSON for seed-specific intervals and leave-one-speaker-out sensitivity.

## Changes from clean input

Deletion realizations are averaged within sample and seed first. Three interventions form a separate Bonferroni family (nominal 98.3333% individual intervals).

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | -0.015221 | -0.022127 | +0.001559 | -0.011930 +/- 0.012181 | [-0.036516, +0.004683] |
| time_scale_1.1 | -0.015137 | +0.002278 | +0.014361 | +0.000501 +/- 0.014829 | [-0.021833, +0.018629] |
| delete_0.1 | -0.001786 | -0.000995 | +0.009794 | +0.002337 +/- 0.006469 | [-0.009266, +0.011061] |

## Limits and interpretation

- Only ten speaker clusters: percentile intervals may have poor coverage; no formal significance or causal confirmation.
- Intervals condition on three observed trained model pairs and do not generalize over training seeds.
- Three-seed sample SD is not a confidence interval; samples/replicates are not independent training runs.
- Primary two-feature and secondary three-intervention Bonferroni families are separate; no combined familywise guarantee.
- Seed-specific intervals are diagnostic, not a simultaneous six-coefficient testing family.
- Exploratory specification written after descriptive validation results were observed; not preregistration.
- Speaker/class/linear duration/activity adjustment does not remove all sample-difficulty confounding.
- No official-test access, new neural-model inference or training.
- Fast/slow input contrast is a proxy, not a direct measurement of branch-specific adaptation strengths or mediation.
- Reused MT predictions are the original parent observations, not new independent replications.

The SA-MT hypothesis predicts a positive adjusted relationship between the frozen transient/sustained input-activity contrast proxy and relative loss benefit. Autocorrelation is a two-sided exploratory covariate in the same regression. Inspect training-seed and leave-one-speaker-out consistency. A conditional direction does not prove a causal mechanism, actual branch-strength alignment or robustness over training seeds. Negative/inconclusive findings are retained. Reused MT predictions are not new independent observations.
