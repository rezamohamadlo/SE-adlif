# Stage 5: adjusted feature–benefit analysis

Specification date: 12 November 2026. This is an exploratory analysis defined
after descriptive validation results were observed, not a preregistered or
independent confirmation. The versioned definition is
[adaptation_statistics_v1.yaml](../../config/studies/adaptation_statistics_v1.yaml).
It pins the exact completed full-validation manifest and completion hashes.
No new checkpoints, training, inference or official-test data are used.

Execution status: completed locally on 12 November 2026. All 1,631 samples
were complete cases, 2,000 bootstrap draws were valid, and all eight recorded
output hashes verified. See the [summary](../../results/experiments/adaptation_mechanism_full_validation_v1/statistics_v1/SUMMARY.md)
and [interpretation](../../results/experiments/adaptation_mechanism_full_validation_v1/statistics_v1/INTERPRETATION.md).
The 70 targeted non-training tests passed. The original 57-output neural
evaluation still verifies unchanged. No rerun is required.

## Analysis defined before the first adjusted fit

The primary outcome for each sample and training seed is supervised
`loss(MT) - loss(AMT)`. Both clean-input temporal features enter the same linear
regression: fast/slow activity contrast and autocorrelation crossing lag.
The fast/slow hypothesis is a positive adjusted association with AMT benefit.
The autocorrelation association is exploratory and two-sided; a directional
LT-DTH hypothesis is not reassigned to AMT.

Controls are class fixed effects, speaker fixed effects, log1p(valid frame
duration in ms), and log1p(overall frame-count rate in Hz). Speaker effects are
removed by demeaning the design and outcome within speaker. Class indicators
and both continuous controls are demeaned too. The two temporal features and
two transformed controls are standardized using population SD on complete
clean samples, with scaling held fixed during resampling. Each coefficient
therefore describes loss-benefit change per one full-cohort feature SD,
conditional on the other feature and controls. No outcome clipping, outlier
removal or feature selection is performed. Save conditioning diagnostics and
leave-one-speaker-out coefficients to expose sensitivity.

Missing, invalid, or censored temporal features are not imputed. Complete-case
exclusions, IDs and reasons must be saved, and shared across seeds. Secondary
perturbation effects use all valid saved outcomes, not only primary complete
cases. No data-dependent threshold or subgroup is selected. MT loss is not a
classification-difficulty control: it is already part of the response, so that
adjustment risks mathematical coupling. Residual sample difficulty remains an
explicit limitation.

There are only three training seeds and ten validation speakers. Compute one
regression per training seed using the same design, and report each coefficient
plus the mean and sample SD across seeds. Bootstrap whole speakers with
replacement 2,000 times, with seed 20261112. The same speaker draw is applied
to every training seed and intervention, retaining within-sample pairing.
The aggregate coefficient averages the three seed coefficients within each
bootstrap draw. This estimates sensitivity to speaker-cluster resampling
conditional on these three trained model pairs; it does not establish
uncertainty over a population of training seeds. The sample SD across three
seeds is not a confidence interval. Independent/exchangeable speaker clusters
are an assumption, not a property established by this dataset.

Use approximate percentile intervals. For the two primary feature coefficients,
Bonferroni-adjust each interval to nominal 97.5% coverage (alpha .05, family 2).
For the three secondary interventions, use a separate nominal 98.3333% family
of three intervals. These are separate analysis families, not joint 95% coverage
across both. With ten clusters, percentile coverage can be poor: these intervals
are exploratory sensitivity summaries, not reliable formal significance tests.
No p-values or claims of confirmed/rejected causal hypotheses are produced.
Record positive/negative/inconclusive conditional directions and seed/leave-one-
speaker consistency. Contradictory and negative results remain in the report.

Secondary outcomes are within-sample changes in loss benefit from clean for
time scales .9 and 1.1 and deletion .1. Average the three deletion realizations
within sample and training seed before aggregation or resampling. The inference
unit remains speaker clusters and three fixed training seeds; neither repeated
samples nor perturbation realizations create training runs. No transformed-input
feature regression or mediation analysis is implied.

The implementation solves weighted within-speaker least squares through
per-speaker sufficient statistics. It checks rank/conditioning on the full
design and every bootstrap draw. Reject non-estimable draws without replacement
draws, record their number, and refuse intervals if fewer than 95% of the
specified draws are valid. Require at least eight speaker clusters. No
speaker intercept columns for absent resampled speakers are retained.

The clustering and few-cluster caveats follow the methodological issues
discussed by [Cameron and Miller (2015)](https://facultysites.dss.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf).
That guide does not establish that this particular ten-speaker dataset has
accurate bootstrap coverage; the choices above are this study's exploratory
specification.

## Local execution and sharing

This is short CPU analysis of saved result files; no PC assignment is needed.
Use the repository's `.venv` from the checkout root:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_statistics/analyze.py --check-only
.\.venv\Scripts\python.exe scripts/adaptation_statistics/analyze.py
.\.venv\Scripts\python.exe scripts/adaptation_statistics/analyze.py --verify-only
```

Outputs are under the source result directory's `statistics_v1/`: resolved
specification/manifest, summary, adjusted-effect figures, complete-case audit,
bootstrap estimates and completion hashes. Source evaluation records remain
unchanged. Detailed sample tables and bootstrap arrays stay in
`diagnostic_rows/`, which Git ignores; share that directory with the summaries.
Changed source identity, configuration or implementation requires a new
statistics output root. Verification checks the saved statistics package
without dataset access or a GPU. Existing historical evaluation verification
remains valid even as this separate statistical implementation evolves.

After this exploratory analysis, review robustness and limitations before
adding other model pairs or datasets. Speaker adjustment and linear controls
cannot establish causality or remove all difficulty confounding.
