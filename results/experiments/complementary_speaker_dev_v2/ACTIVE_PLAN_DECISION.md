# Active research-plan decision: generalization first

Effective 2026-11-12. This decision confirms and clarifies the subsequent-stage
order in [the v2 guide](../../../docs/experiments/complementary-mechanisms-study-v2.md).
It supersedes random-split validation as the sole SHD idea-screening criterion
and the earlier request to evaluate exploratory official tests now. It does
not change frozen run configurations, completed evidence or canonical SHD*.

The central problem is weak generalization, especially to unseen SHD speakers.
High random-split validation accuracy has not reliably predicted official-test
performance. Development must therefore measure clean held-out-speaker accuracy
directly. A smaller training–validation gap alone is not a successful result.

## Frozen development protocols

Use only `data/SHDWrapper/shd_train.h5`; never load the official test file for
idea screening. Keep the already frozen speaker IDs and original sample indices.

| Role | Training speakers | Validation speakers | Train / validation rows |
|---|---|---|---:|
| Primary development | 1, 2, 6, 7, 8, 9, 10, 11 | 0, 3 | 6,521 / 1,635 |
| Predefined shortlist sensitivity | 0, 1, 2, 3, 7, 8, 9, 11 | 6, 10 | 6,920 / 1,236 |

The manifests in `primary_0_3/` and `sensitivity_6_10/` retain the source hash,
exact indices and class counts. Recheck disjoint rows and speakers, exhaustive
coverage of the official training file, and all 20 classes in both subsets
before candidate evaluation. Do not regenerate a split after seeing scores.
Speakers 0/3 have extensive tuning exposure: their results are development.
The fixed 6/10 pair is a sensitivity diagnostic, has also accumulated exposure,
and must not be described as untouched confirmation. Do not tune on it, try
additional pairs, or select a favorable pair.

Start comparisons at training seed 42. Train Baseline, A and A+B fresh under
the same split, preprocessing, batch size, optimizer, LR schedule, dropout,
screening budget and checkpoint-selection rule. Record intended mechanism
differences explicitly. Include B-only before claiming complementarity. Reuse
a historical control only after matching every relevant setting, source/model
behavior, exact indices, budget and training exposure. A checkpoint trained on
any held-out speaker is ineligible, even if its filename or seed matches.

Select candidates using improved clean held-out-speaker accuracy; review hard-label
validation CE, learning curves and individual-speaker results as supporting
evidence. Keep dropout/augmentation off in validation. Retain negative outcomes.
The existing +0.50 percentage-point practical screen is a development decision
rule, not statistical significance or a guaranteed test improvement. Compare
matched-budget controls and retain small or mixed effects honestly.

## Revised stage order

1. **Audit and preserve.** Identify actual protocols from live commands, resolved
   configs, manifests and metrics. Preserve active runs and all prior evidence.
   Verify the frozen training-only speaker manifests.
2. **Brief primary speaker screen.** Fresh Baseline / A / A+B, seed 42, fixed
   speakers 0/3 and matched 30-epoch ceiling. Review B-only to separate generic
   regularization from a complementary effect. Extend the matched comparison
   to the predefined 75-epoch ceiling only with a recorded justification.
3. **Pair sensitivity for the shortlist.** Freeze candidate settings first;
   train fresh controls and shortlisted candidates on the fixed 6/10 split.
   Primary weights have seen 6/10 and cannot initialize this assessment. No
   pair search, secondary tuning or score-driven extension.
4. **SSC validation transfer.** Check promising mechanisms with the verified
   official train/validation protocol and matched controls, initially seed 42,
   10 epochs; consider the predefined 40-epoch stage only after reviewing the
   bounded screen. Official tests remain disabled.
5. **ECG validation transfer.** Use the verified existing row split, labels and
   masks, with matched controls and bounded 50-to-100-epoch review where
   justified. No patient-independence claim without patient IDs. Do not fully
   train every idea on every dataset.
6. **Freeze the final study.** Freeze A, B, four arms, configurations, selection
   rules, seeds, budgets and final endpoints using development evidence only.
   If two complementary ideas are not supported, return to development rather
   than declaring a successful combination.
7. **Fresh canonical comparisons.** Run Baseline / A / B / A+B under canonical
   SHD*: random 20% validation from official training, split seed 42, full
   300-epoch budget. Train each arm on that canonical split; speaker-holdout
   weights are ineligible. Complete frozen SSC 40 / ECG 400 comparisons and
   matching training seeds 42, 123 and 456. Preserve dataset-specific settings.
8. **Frozen final evaluation and reporting.** Only after validation-based
   selection, evaluate ordinary official-test inference under the frozen plan.
   Report all arms, datasets and seeds, mean/sample SD, paired contrasts,
   train/validation/test definitions, uncertainty, compatible paper deltas and
   costs. Acknowledge historical test exposure; do not use test scores to
   choose ideas, pairs, hyperparameters or TTA.

Speaker-development and canonical SHD* outputs and tables remain separate.
Held-out-speaker validation is not an official-test score and is not directly
comparable with the paper's canonical SHD* test mean. Historical random-split
results remain evidence under their original protocol, not clean speaker controls.

## Current bounded-stage disposition

Read-only inspection found launcher PID 20700, training process PID 9232 and
its data-loader children active. The actual study is
`config/studies/complementary_C005_dropout_ssc_v1.yaml`: fresh Baseline and
R-Drop A-only, SSC official validation, seed 42, mean-membrane CE, training batch
512 / validation batch 1024, LR 0.006, 10-epoch ceiling within a 40-epoch schedule.
`run_test=false`, `validation_only=true`, and training/validation-only data are
explicit in the live command. This is stage 4 above, not random-split SHD*,
speaker-validation SHD or official-test evaluation. Baseline had completed
5/10 epochs at inspection; this is progress, not a completed comparison.

**Decision: leave the existing bounded queue intact.** It fits the corrected
transfer stage. Review both completed arms before deciding on any continuation;
there is no automatic 40-epoch extension. Do not stop, restart, duplicate or
change its settings. This plan update starts no training or test evaluation.

The main guide and code/config source are currently hash-frozen for this queue.
This decision is stored beside study evidence to preserve that identity; it is
linked from `command.txt`. Reconcile the guide's historical status paragraphs
after the bounded stage and any same-source continuation have closed. Current
progress must be established from processes and run evidence, not old guide text.
