# LT-DTH/DTH statistical results: review of saved outputs

Reviewed 12 November 2026 after user execution. Read-only inspection confirmed
the completion identity and all nine recorded output hashes. The saved summary
reports 1,631 complete cases, ten speakers, three training seeds and 2,000
valid speaker bootstrap draws with no rejected draws. The agent did not rerun
regression, resampling, tests or neural inference.

- **Fast/slow contrast:** adjusted mean coefficient -0.010032 loss-benefit
  units per feature SD, approximate corrected interval [-0.061780, +0.024167].
  Signs differ across seeds; the leave-one-speaker-out mean range also crosses
  zero. The association is inconclusive.
- **Input autocorrelation timescale:** adjusted mean coefficient -0.014033,
  interval [-0.053943, +0.008900]. Although leave-one-speaker-out mean estimates
  remain negative, seed 123 is positive and the cluster interval includes zero.
  The proposed relationship with input timescale is not established by this
  analysis; neither causality nor learned-timescale alignment is demonstrated.
- **Time scaling:** both .9 and 1.1 changes in relative loss benefit have
  intervals containing zero. There is insufficient evidence for a consistent
  effect across the three trained model pairs.
- **10% event deletion:** mean change in relative loss benefit -0.005778,
  approximate corrected interval [-0.013320, -0.000443]. This is an exploratory
  negative direction conditional on the observed cohort: deletion reduces
  LT-DTH's relative loss advantage on average. Seed 42 is positive while seeds
  123 and 456 are negative. This is not a claim of seed-robust deterioration,
  reduced absolute accuracy or a reliable formal significance result.

Learned fast/slow trace descriptors differ from the fixed DTH traces and across
layers. Their table is descriptive at model/layer level: each group has three
trained model observations, not 1,631 independent constants. No regression or
correlation of these parameters against sample-level benefit was performed.

Intervals use separate two-feature and three-intervention Bonferroni families.
With only ten speaker clusters, bootstrap coverage can be poor, and inference
is conditional on three fixed training seeds. The specification was defined
after descriptive validation results were observed. These are exploratory
development results, not preregistered confirmation, independent final-test
evidence, causal explanations or proof of a useful learned timescale.

Next: preserve these mixed/inconclusive findings and prepare the next planned
pair, SA-MT versus MT on SHD*, with a separate frozen study/output identity.
Existing MT predictions may be reused only after verifying input, checkpoint,
split, condition and numerical-protocol identities. Statistical analysis of
that pair requires its own hypotheses and source identities. All execution
remains user-launched; agents may only inspect files and perform preflight.
