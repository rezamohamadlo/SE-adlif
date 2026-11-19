# MT-TC/MT validation output review

Reviewed 12 November 2026 after user execution. Read-only inspection matched
the completion identity and all 58 recorded output hashes. All 24 reused
value hashes remain unchanged; the 36 prediction chunks have matching
checkpoint/condition identities and all 1,631 sample IDs. Six saved clean
anchors are within the recorded tolerances. The agent did not rerun inference,
statistics, bootstrap or tests.

The source retains the distinct MT-TC training objective and JS weight .1 /
window 5. All three report metadata records identify the comparison loss as
supervised cross-entropy with the auxiliary training JS excluded. The plain
frozen forward and direct classification cross-entropy are used for evaluation;
training loss totals are not substituted. The shared baseline does not make
the intentionally different training objectives identical.

Completion runtime: 154.219 seconds, approximately 2 minutes 34 seconds. MT
predictions were reused, so this is not a pure model-latency comparison with
previous evaluations.

Saved clean-validation results, mean +/- sample SD across training seeds
42, 123 and 456:

| Metric | MT | MT-TC |
|---|---:|---:|
| Accuracy (%) | 99.2643 +/- 0.0613 | 99.1825 +/- 0.0937 |
| Supervised loss | 0.06651386 +/- 0.02405521 | 0.06858920 +/- 0.01114392 |

Accuracy SD uses percentage points. The paired MT-TC minus MT accuracy gap is
-0.0817 +/- 0.0937 pp. Seed 123 ties MT; seeds 42 and 456 are lower. MT-TC has
lower supervised loss in seed 42, but higher loss in 123 and 456. Accuracy and
loss need not rank models identically. No consistent clean-input improvement
is established. These are validation-development results, not final-test scores.

The descriptive perturbation tables report negative mean change in relative
loss benefit under time scale .9 and positive mean changes under time scale
1.1 and 10% event deletion. These values do not establish adjusted statistical
uncertainty, temporal-evidence consistency, causality or seed-population
robustness. Reused predictions and deletion realizations are not new training
seeds.

Next: specify MT-TC-specific statistical analysis. Existing fast/slow contrast
and input ACF can support exploratory input-feature associations, but neither
directly measures whether class evidence persists across adjacent model-output
windows or appears in brief events. Such a window-evidence diagnostic requires
an explicit definition and, if saved temporal outputs are insufficient, a
separate user-launched frozen-model evaluation. Do not treat the auxiliary JS
training objective or per-sample true-class probability as automatically being
that diagnostic, and do not invent temporal outputs from saved aggregate losses.
No such diagnostic or statistical analysis was run during this review.
