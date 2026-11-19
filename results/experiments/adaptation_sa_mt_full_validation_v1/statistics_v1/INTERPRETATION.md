# SA-MT/MT adjusted statistics: saved-result review

Reviewed 12 November 2026 after user execution. Read-only inspection confirmed
the completion identity and all eight recorded output hashes. The saved summary
reports 1,631 complete cases, ten speakers, three training seeds and 2,000 valid
speaker bootstrap draws with no rejected draws. The agent did not rerun the
analysis, tests, bootstrap or neural inference.

- **Transient/sustained contrast proxy:** mean adjusted coefficient -0.019664
  loss-benefit units per feature SD; approximate corrected interval
  [-0.047620, +0.001216]. Seeds 42 and 123 are negative, seed 456 positive.
  Leave-one-speaker-out mean estimates remain negative. The hypothesized
  positive association is not supported; the interval includes zero, so the
  analysis remains inconclusive rather than proving the opposite hypothesis.
- **Autocorrelation timescale:** mean adjusted coefficient +0.032337,
  interval [+0.005389, +0.078847]. Leave-one-speaker-out mean coefficients
  remain positive, but seed 456 is negative. This is an exploratory positive
  association conditional on the observed cohort, not a causal explanation
  or a relationship robust over training seeds.
- **Controlled interventions:** all three approximate corrected intervals
  for change in relative loss benefit include zero (time scales .9/1.1 and
  10% event deletion). Effects vary across seeds. Consistent intervention
  benefit or evidence of the proposed mechanism has not been established.

The two features enter jointly, controlling class, speaker, valid duration
and total activity. Input contrast is a proxy rather than a direct measurement
of branch adaptation strengths; no parameter alignment or mediation analysis
was performed. Reused MT predictions remain the original parent observations,
not additional independent replications.

All intervals are approximate speaker-cluster percentile summaries with only
ten clusters, conditional on three fixed trained model pairs. Coverage may be
poor. The primary two-feature and secondary three-intervention families are
separate. These are exploratory validation-development findings defined after
descriptive results were observed, not preregistered confirmation, independent
official-test evidence or reliable formal significance tests. Negative and
inconclusive findings must be retained.

Next: prepare the last planned SHD* model pair, MT-TC versus MT, with a distinct
frozen study/output identity. Its temporal-consistency hypothesis requires a
separate statistical specification; the existing features are not automatically
direct measures of stable class evidence in adjacent windows. Use supervised
prediction loss only, excluding the auxiliary JS training penalty. Preserve
checkpoint identity and validate any reuse of MT predictions. Execution and
tests remain user-launched; agents may only inspect records and run preflight.
