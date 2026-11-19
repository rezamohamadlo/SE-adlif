# MT-TC statistics and window evidence: saved-result review

Reviewed 12 November 2026 after user execution. Read-only checks matched the
completion identities and all eight statistical and ten diagnostic output
hashes. The statistical summary reports 1,631 complete cases, ten speakers,
three training seeds and 2,000 valid bootstrap draws with no rejections.
All six diagnostic chunks have matching identities and value digests, cover
the same 1,631 samples, and have at least two valid windows with no undefined
descriptor flags. Six saved clean compatibility anchors are present; ordinary
predictions passed the runtime's exact-match gate. Saved mean loss drift is
on the order of 1e-8 and maximum per-sample drift on the order of 1e-5.
No experiment, regression, bootstrap or tests were rerun by the agent.

## Adjusted input-feature associations

- Fast/slow contrast: adjusted mean +0.012434 loss-benefit units per feature
  SD, approximate corrected interval [-0.056932, +0.078558]. All three seed
  coefficients are positive, but speaker-cluster uncertainty includes zero.
  The association is inconclusive.
- Input autocorrelation timescale: adjusted mean +0.030419, interval
  [-0.008861, +0.111900]. Signs differ across seeds and the interval includes
  zero. This association is also inconclusive.
- Time compression .9: change in relative loss benefit -0.023025, interval
  [-0.046182, -0.002106]; all three seed estimates are negative. This is an
  exploratory negative direction conditional on the observed model cohort,
  not reliable formal significance, population-wide seed robustness or proof
  of the temporal-consistency mechanism.
- Time scale 1.1 and 10% event deletion have intervals containing zero.
  Their effects remain inconclusive. The auxiliary training JS penalty is
  excluded from every supervised loss-benefit calculation.

These regressions control class, speaker, valid duration and total activity.
With only ten speaker clusters, percentile interval coverage may be poor.
Primary two-feature and secondary three-intervention families are separate.
Specification followed descriptive validation inspection, not preregistration.

## Direct clean output-window diagnostic

The completed diagnostic uses nonoverlapping 20-ms windows after burn-in,
excluding minimum and batch padding. It describes model output probabilities,
not input ACF, and does not reconstruct temporal evidence from aggregate loss.

MT-TC minus MT adjacent-window JS is negative in seed 42, but positive in
123 and 456. The across-seed mean is +0.00024379 nats with sample SD 0.00061819.
Thus the training regularizer did not produce consistently smoother clean
window distributions under this distinct nonoverlapping-window diagnostic.
This is not a claim about whether the original overlapping training JS
objective decreased during training.

Mean true-class probability is higher for MT-TC in all three seeds. Window
classification accuracy, however, improves only in seed 42 and falls in 123
and 456. Across-seed mean window-accuracy difference is -0.00278361 in fraction
units (approximately -0.2784 pp), with substantial seed variation. Higher
true-class probability alone does not establish better decisions or a useful
temporal-consistency mechanism. Evidence concentration and switching measures
also vary by seed. Windows and samples are not independent training runs.

The diagnostic is descriptive: no confidence/stability-to-benefit regression,
mediation analysis or causal claim was fitted. Parent-derived confidence can
be mathematically coupled to a benefit outcome that already includes parent
loss. This output diagnostic does not independently validate the hypothesized
conditions under which MT-TC helps or harms. Raw window records remain available
for a separately justified analysis rather than post-hoc feature selection.

## Scope and next step

All four planned SHD* model-pair evaluations now exist, together with their
exploratory adjusted input-feature statistics; MT-TC additionally has the direct
clean-window diagnostic. This does not finish the supervisor roadmap: SSC/ECG
analyses, comprehensive feature distributions and paper-code mapping, gradient/
diagnostic-input work, and the final mechanism/cost report remain incomplete.

Next: consolidate the four SHD* findings with source links, explicitly retain
mixed/negative results and shared-parent dependence, and inspect SSC data and
checkpoint readiness before defining its next user-launched experiment. Do
not claim cross-dataset generality from SHD* or compare raw loss units across
tasks without justification. Agent work remains limited to preparation,
preflight and saved-record inspection; execution/tests are user-launched.
