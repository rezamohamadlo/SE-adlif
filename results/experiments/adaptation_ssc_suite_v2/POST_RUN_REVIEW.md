# SSC four-pair suite: completed saved-evidence review

Reviewed 2026-11-14. **All four SSC model-pair evaluations and their four
statistical jobs are executed and verified.** The user ran the queue and tests.
This review read saved artifacts and checked their hashes, identities,
coverage, pairing and numerical consistency. It did not rerun tests, neural
inference, feature extraction, regression, bootstrap or plots.

See the [machine-readable review](POST_RUN_REVIEW.json),
[scientific completion](COMPLETE.json), [I/O completion](IO_COMPATIBILITY_COMPLETE.json),
[consolidated result](SUMMARY.md) and [manifest](MANIFEST.json).

## Verified evidence

- **230 core output hashes**, three I/O provenance output hashes, completion
  identities, **68 frozen scientific source hashes** and three compatibility
  source/test hashes pass verification. Eighteen checkpoint and original
  metric-file hashes match the frozen registry.
- The user-run logs record **25 original contract tests plus eight atomic I/O
  tests passed**. Tests were not rerun by the agent. The successful compatibility
  attempt records no actual runtime retries; denial messages in its test log
  come from deliberately simulated errors. Do not attribute the original lock
  to a particular process or claim that an actual retry occurred this session.
- Each pair covers **9,981 official validation records, three training seeds
  and six conditions/realizations**, preserving original sample order. The
  review verified **108 unique prediction chunks** (1,077,948 model predictions)
  and six shared feature chunks. Reused MT observations were checked wherever
  consumed and are not extra independent predictions or training seeds.
- All **718,632 paired CSV rows** (179,658 per comparison) match their saved
  prediction pairs field by field. Labels, speakers, input hashes and shared
  feature metadata agree. Saved time-scaling spike totals are conserved and
  deletion totals do not increase. Actual count arrays were not regenerated.
- The four pair views contain six clean anchors each. These are **18 unique
  model/seed anchors**, with MT reused in multiple comparisons. Every anchor
  agrees with its saved selected-epoch validation metrics within the frozen
  accuracy/loss tolerances and with the saved clean predictions.
- Per-seed losses, accuracies, benefits, report sample counts and scope metadata
  agree with the stored rows. The statistical jobs each use **9,979 complete
  cases and 256 speaker groups**, explicitly excluding censored ACF IDs **3552
  and 7837** from joint regression. All 9,981 recordings remain in evaluation
  and intervention effects.
- Each pair stores **2,000 valid bootstrap draws and zero rejected draws**.
  Saved estimate arrays have the expected shapes and finite entries; the
  2,000-by-256 speaker-weight matrices and exclusion records are identical
  across comparisons. Speaker-code mappings preserve the original strings.
  No resamples, coefficients or confidence intervals were regenerated here.
- Six LT-DTH/DTH parameter records match checkpoint identities, learned/fixed
  status and physical conversion: e-folding frames = −1/log(rho), milliseconds
  = four ms times frames. These remain model/layer descriptors, not independent
  per-record parameter observations.

## Clean SSC validation accuracy

Values below are copied from the saved clean reports: mean ± sample SD across
the three matching training seeds. Accuracy means are percentages; accuracy SDs
and paired differences are in percentage points (pp). There is no numeric paper
baseline comparison in this table.

| Variant / parent | Parent accuracy | Variant accuracy | Variant − parent |
|---|---:|---:|---:|
| AMT / MT | 82.1528% ± 0.2719 pp | 82.0459% ± 0.1735 pp | −0.1069 ± 0.3889 pp |
| LT-DTH / DTH | 81.5783% ± 0.3465 pp | 81.5917% ± 0.0504 pp | +0.0134 ± 0.3357 pp |
| SA-MT / MT | 82.1528% ± 0.2719 pp | 81.8822% ± 0.0652 pp | −0.2705 ± 0.2768 pp |
| MT-TC / MT | 82.1528% ± 0.2719 pp | 82.1127% ± 0.1796 pp | −0.0401 ± 0.4513 pp |

AMT and LT-DTH are higher than their parents in one seed and lower in two.
SA-MT is lower in all three seeds. MT-TC is higher in two and lower in one.
**None establishes a consistent clean-accuracy improvement across seeds.**
LT-DTH's small positive average gap is not sufficient evidence of superiority.
The repeated MT column denotes the same parent observations, not independent
replication. Original validation checkpoint selection and the different MT-TC
training objective remain part of the comparison's provenance.

## Statistical interpretation

Both adjusted clean-input feature intervals include zero in **every pair**:
all eight fast/slow-contrast or ACF feature/pair results are inconclusive under
the specified exploratory interval method. This is not proof of no association.
LT-DTH's contrast slopes are negative in all seeds and SA-MT's ACF slopes
positive in all seeds, but their adjusted intervals still include zero.

Among the twelve intervention/pair effects, one saved adjusted interval excludes
zero: **LT-DTH/DTH under time scale 1.1** has relative loss-benefit change
**+0.014089 ± 0.016510** across seeds, with approximate interval
**[+0.002457, +0.025773]**; all three per-seed changes are positive. This is a
conditional exploratory relative-loss observation. It does not establish
better absolute loss/accuracy, a causal learned-trace mechanism, population
robustness over new training seeds or an independent test improvement.
The other eleven intervention intervals include zero.

The primary eight-coefficient and secondary twelve-effect families are separate
Bonferroni-adjusted approximate percentile families, conditional on three existing
trained model pairs and 256 sampled speaker groups. Two thousand draws provide
limited tail precision. The method was specified after viewing AMT descriptive
results, and class/speaker/duration/activity controls do not remove all difficulty
confounding. No formal significance or joint-family coverage claim is added.

Read the complete per-seed results and limitations:

- [AMT/MT statistics](pairs/amt_mt/statistics_v1/SUMMARY.md).
- [LT-DTH/DTH statistics](pairs/lt_dth/statistics_v1/SUMMARY.md).
- [SA-MT/MT statistics](pairs/sa_mt/statistics_v1/SUMMARY.md).
- [MT-TC/MT statistics](pairs/mt_tc/statistics_v1/SUMMARY.md).

The MT-TC benefit excludes auxiliary JS while preserving its distinct training
objective. Input contrast/ACF do not directly test adjacent output-window
stability. Validation data used for checkpoint selection remain development
evidence; full coverage does not make them independent confirmation.

## Timing and remaining scope

The **last successful resumed core invocation** records **3,426.25 seconds
(57 minutes 6 seconds)**. The enclosing compatibility attempt records about
58 minutes 5 seconds, including its extra checks and startup. Earlier failed
sessions (4,086.765 seconds and 53.485 seconds) and the original AMT evaluation
are preserved separately. Do not present the latest elapsed value, LT-DTH's
cached reporting time or a simple mixture of these scopes as total model
evaluation time or inference latency.

For the main four-pair-by-three-dataset comparison matrix, **SHD* and SSC now
account for 8 of 12 completed pair studies (about 67% by count)**. This is not a
runtime/effort estimate or proof that every stage-6 deliverable is complete.
ECG's four comparisons remain pending; direct SSC output-window stability,
additional diagnostics, stage 7 gradients and final synthesis remain separate.

**Next:** perform an ECG artifact/config and validation-protocol readiness audit,
then prepare a small validation-only pilot with label/mask compatibility checks.
No ECG experiment, new training or gradient analysis was started in this review.
Transfer the entire SSC suite root, including its shared snapshot, path index,
raw rows, statistical arrays and I/O provenance, for self-contained verification.
