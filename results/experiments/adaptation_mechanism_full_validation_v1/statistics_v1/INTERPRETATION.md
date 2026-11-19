# Interpretation: SHD* AMT/MT Stage 5

Reviewed 12 November 2026. The adjusted analysis is completed and its eight
recorded output hashes verify. All 1,631 samples are complete cases; all 2,000
speaker bootstrap draws were estimable. This is exploratory validation analysis
of ten speakers and three fixed trained model pairs, not independent testing.

- **Fast/slow contrast:** mean adjusted coefficient +0.004764 loss units per
  feature SD; approximate corrected interval [-0.026625, +0.054401]. Two seeds
  are negative and one positive. Leave-one-speaker-out mean coefficients range
  from -0.012771 to +0.011501. The proposed positive AMT association is
  **inconclusive** in this cohort, not established.
- **Autocorrelation timescale:** mean adjusted coefficient +0.026769; approximate
  corrected interval [+0.003042, +0.063770]. The fixed-cohort speaker-resampling
  summary is positive and leave-one-speaker-out mean coefficients remain
  positive, but seed 42 is negative while seeds 123 and 456 are positive.
  This is an **exploratory positive association conditional on this cohort**,
  not evidence of a training-seed-robust AMT mechanism or causal benefit.
- **Interventions:** approximate corrected intervals for change in benefit
  from clean all include zero: time scale .9, time scale 1.1 and 10% deletion.
  Effects differ across seeds. These analyses do not establish that manipulating
  the temporal statistics causes a consistent AMT advantage.

Controls are class, speaker, valid input duration and total activity; the other
temporal feature enters jointly. Overall within-speaker regression R-squared is
approximately 0.016, 0.025 and 0.016 across seeds, including all controls. This
is not a variance contribution attributable solely to the temporal features.
No sample-difficulty control derived from the same MT loss was included.
Residual difficulty/confounding, ten-cluster coverage limitations and the
previous observation of descriptive validation/test histories remain limits.

In `adjusted_effects.png` and `.svg`, blue diamonds denote the equal-weight
mean across training seeds, blue horizontal lines denote the approximate
speaker-cluster intervals, and grey crosses denote the three seed estimates.
The two panels use separate multiplicity families (two primary features and
three secondary interventions); there is no joint familywise coverage claim
across both panels. No formal p-values are reported.

Source neural evaluation records were preserved. No training, new model
inference, dataset download or official-test access was needed. The analysis
specification was written before the adjusted fit but after descriptive
validation results were observed; it is not preregistration.

Next: retain these negative/mixed findings, review the adjusted analysis and
its sensitivity, then prepare the next planned model pair (LT-DTH/DTH on SHD*)
with the same evidence discipline. Cross-model and cross-dataset generality
remain untested. Do not select new features or checkpoints to make this result
positive, or use official-test performance to choose an explanatory story.
