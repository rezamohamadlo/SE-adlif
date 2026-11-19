# Stage 5: exploratory adjusted associations

Samples: 1631; complete cases: 1631; speakers: 10; training seeds: 3.

Outcome = loss(MT) - loss(MT-TC), positive favors MT-TC. Both features enter jointly. Controls: class, speaker, log1p(duration), log1p(activity). Feature coefficients use one original-cohort SD; bootstrap scales are fixed.

Intervals are approximate speaker-cluster percentile sensitivity intervals conditional on three trained model pairs. There are only ten speakers; coverage may be poor. No p-values, formal significance claim, causal confirmation or independent test claim is made.

## Adjusted clean-input associations

| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 97.5% interval | Conditional direction |
|---|---:|---:|---:|---:|---:|---|
| fast_slow_contrast | +0.022038 | +0.010494 | +0.004770 | +0.012434 +/- 0.008796 | [-0.056932, +0.078558] | inconclusive |
| autocorr_tau_ms | +0.055703 | +0.082148 | -0.046594 | +0.030419 +/- 0.067993 | [-0.008861, +0.111900] | inconclusive |

Two-feature Bonferroni family, nominal 95% family level subject to unreliable small-cluster bootstrap coverage. See JSON for seed-specific intervals and leave-one-speaker-out sensitivity.

## Changes from clean input

Deletion realizations are averaged within sample and seed first. Three interventions form a separate Bonferroni family (nominal 98.3333% individual intervals).

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |
|---|---:|---:|---:|---:|---:|
| time_scale_0.9 | -0.013566 | -0.049496 | -0.006014 | -0.023025 +/- 0.023233 | [-0.046182, -0.002106] |
| time_scale_1.1 | +0.014602 | -0.001854 | +0.014438 | +0.009062 +/- 0.009454 | [-0.007546, +0.028551] |
| delete_0.1 | +0.009476 | -0.000870 | +0.016062 | +0.008223 +/- 0.008535 | [-0.003511, +0.021013] |

## Limits and interpretation

- Only ten speaker clusters: percentile intervals may have poor coverage; no formal significance or causal confirmation.
- Intervals condition on three observed trained model pairs and do not generalize over training seeds.
- Three-seed sample SD is not a confidence interval; samples/replicates are not independent training runs.
- Primary two-feature and secondary three-intervention Bonferroni families are separate; no combined familywise guarantee.
- Seed-specific intervals are diagnostic, not a simultaneous six-coefficient testing family.
- Exploratory specification written after descriptive validation results were observed; not preregistration.
- Speaker/class/linear duration/activity adjustment does not remove all sample-difficulty confounding.
- No official-test access, new neural-model inference or training.
- Input contrast and ACF do not directly measure temporal stability of model class evidence; direct window diagnostics are separate.
- Reused MT predictions are the original parent observations, not new independent replications.

Both input-feature associations are exploratory and two-sided for MT-TC. Neither input contrast nor ACF directly measures stability of adjacent output-window class evidence. The window diagnostic is a separate frozen-model evaluation. Inspect seed and speaker sensitivity; conditional directions do not establish causality or robustness across training seeds. The auxiliary JS penalty is excluded from every loss-benefit calculation. Reused MT records are not independent replications.
