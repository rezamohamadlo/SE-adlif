# LT-DTH/DTH: adjusted validation statistics

User execution completed 12 November 2026. Read-only inspection confirmed the
completion identity and all nine output hashes. The saved results report
1,631 complete cases and 2,000 valid bootstrap draws. See the
[summary](../../results/experiments/adaptation_lt_dth_full_validation_v1/statistics_v1/SUMMARY.md)
and [interpretation](../../results/experiments/adaptation_lt_dth_full_validation_v1/statistics_v1/INTERPRETATION.md).
No rerun is needed. The agent ran only preflight before launch and inspected
saved artifacts after completion; tests remain unrun by the agent.

The [versioned specification](../../config/studies/adaptation_lt_dth_statistics_v1.yaml)
pins the completed LT-DTH/DTH evaluation's manifest and completion hashes.
It inherits the numerical settings of the AMT/MT statistical specification,
but supplies the LT-DTH pair, source identity and hypotheses explicitly.
Existing evaluation and AMT/MT statistical records are preserved unchanged.

The LT-DTH hypothesis is that benefit may depend on the input's effective
temporal scale; no monotonic direction is assumed. The clean-input ACF crossing
lag is its proxy, not a direct measurement of neuronal memory. Fast/slow
contrast remains a second, two-sided exploratory association. Both enter the
same regression, controlling class, speaker, log1p(valid input duration) and
log1p(total input activity). No MT/DTH baseline loss is used as a difficulty
control because it is already part of the response. Residual difficulty
confounding remains a limitation.

The outcome is `loss(DTH) - loss(LT-DTH)`, positive favoring LT-DTH. Per-seed
coefficients, sample SD over three seeds, leave-one-speaker-out sensitivity
and 2,000 shared speaker-cluster bootstrap draws use the established
[statistical method](adaptation-statistics-v1.md). Scaling is frozen on complete
clean inputs. Missing/invalid/censored features have explicit exclusion reasons.
The primary two-feature and secondary three-intervention multiplicity families
remain separate. Interventions compare within-sample benefit changes from clean;
the three deletion realizations are averaged within sample and seed first.

Only ten speakers and three trained model pairs are observed. Percentile
intervals are approximate exploratory sensitivity summaries conditional on
these seeds; no reliable formal significance, causality or population-wide
training-seed robustness is claimed. The specification is written after the
descriptive evaluation was observed, so it is not preregistration. Nothing is
selected using official-test accuracy, adjusted coefficient signs or a desired
positive result. Negative and inconclusive findings must be retained.

The saved fast/slow trace descriptors are summarized separately for each model,
layer and trace, with each seed as one model observation. `TRACE_SUMMARY.json`
contains eight groups (two models x two layers x two traces), each with three
seed-specific values. E-folding milliseconds use the saved physical input bin.
No trace constant is treated as 1,631 independent observations, and no
parameter–performance correlation or learned-timescale matching test is fitted
with these six models/twelve layer records.

Run from the repository root:

```powershell
# Preflight: verifies saved data, identities, trace records and complete cases;
# no adjusted fits, resampling, output writes, inference or training.
.\.venv\Scripts\python.exe scripts/adaptation_lt_statistics/analyze.py --check-only

# USER LAUNCH: computes the adjusted statistics and figures.
.\.venv\Scripts\python.exe scripts/adaptation_lt_statistics/analyze.py

# USER LAUNCH after completion: verify all saved output hashes.
.\.venv\Scripts\python.exe scripts/adaptation_lt_statistics/analyze.py --verify-only
```

Prepared tests (not run by the agent):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_lt_statistics tests.test_adaptation_statistics
```

Outputs are independent of PC assignments:
`results/experiments/adaptation_lt_dth_full_validation_v1/statistics_v1/`.
The manifest saves the resolved specification, source identities, complete-case
IDs, scaling and checkpoint-identified trace records. Summaries and PNG/SVG
figures accompany `TRACE_SUMMARY.json`, exclusions and detailed sample/bootstrap
artifacts under `diagnostic_rows/`. Share the entire directory, including ignored
rows and arrays; Git alone does not transfer them.

An identical completed invocation verifies and skips. Changes to settings or
implementation require a different `--output-root`; never edit a completed
manifest to bypass its guard. Code/artifact paths resolve against the checkout
root and support a changed current directory. This is short CPU analysis; no
training, new neural inference, dataset download or official-test access occurs.
