# DTH complementary-mechanism study v1

Effective 2026-11-12. Objective: improve DTH with one complementary mechanism B,
starting on SSC, then verify ECG transfer and preservation of canonical SHD*.
This replaces the earlier open-ended two-idea discovery objective for subsequent
work. Preserve all previous studies, their negative findings and test exposure.
The user explicitly authorizes launching, resuming and safely stopping this
study's own training jobs, overriding the default user-launch-only rule. Do not
stop unrelated jobs, commit or push. Use `runner.py` and `run.py`; no new runner.

Suggested short Goal: **Execute `docs/experiments/dth-complement-study-v1.md`:
improve DTH with B on SSC, check ECG and canonical SHD*, then freeze ablations.**
Keep the existing native Goal and its usage accounting. The current Goal was
observed active with no native token budget; available Goal tools cannot edit
an active objective or budget without creating another Goal. Do not recreate it
or change global settings. The user requested 50,000 only if native configuration
is supported. Respect the actual native limit, which is currently unset; do not
invent a retroactive hard cap or reset accounting. Keep bounded stage gates and
efficient monitoring. A running bounded job must always be disclosed.

## Evidence and controls

Authoritative ordinary-test records and original `metrics.csv` histories under
`results/valid/DTH_SE_adLIF/` give these matching training-seed results:

| Dataset | Paper mean | DTH seed 42 (Δ paper) | DTH seed 123 (Δ paper) | DTH seed 456 (Δ paper) | DTH mean ± sample SD (Δ paper) |
|---|---:|---:|---:|---:|---:|
| SSC | 80.44% | 80.463153 (+0.023153 pp) | 79.840052 (-0.599948 pp) | 79.560399 (-0.879601 pp) | 79.954535 ± 0.462137% (-0.485465 pp) |
| ECG | 88.18% | 87.811887 (-0.368113 pp) | 87.809706 (-0.370294 pp) | 87.267840 (-0.912160 pp) | 87.629811 ± 0.313478% (-0.550189 pp) |

Δ paper is local minus the published mean, in percentage points. Individual
seed deltas compare a single run with the paper mean, not paired paper seeds.
These are three-run summaries, not required individual-seed outcomes. Paper
source: `reference_paper.pdf`, dataset-specific values retained in
`reports/paper-results.md` and `config/paper_baseline.yaml`. No TTA is included.
Save detailed verification, source/config compatibility and checkpoint evidence
in the new study results directory before selecting a reusable control.

Existing SSC DTH uses official validation, batch 256, LR 0.006, full 40 epochs,
32-bit execution, unchanged mean-membrane CE, no early stopping and the inherited
plateau schedule. Seed 42 retains only selected epoch-index 19 and last weights;
it has no validation-selected checkpoint restricted to the first ten epochs.
It cannot supply the matched ten-epoch checkpoint control. If calibration
selects another batch or preprocessing execution path, start a new matched
DTH control rather than comparing architecture effects across those changes.
Preserve the old control and its test result as historical full-budget evidence.

## B001: dropout-output consistency

Start with established [R-Drop](https://arxiv.org/abs/2106.14448), not a new neuron
or a novelty claim. DTH adapts internal thresholds to causal spike activity;
that does not explicitly require output agreement across dropout realizations.
B penalizes stochastic output disagreement on identical inputs. Whether this
helps DTH generalize across tasks is the hypothesis, not an assumed result.

For the unchanged target-aligned, temporally aggregated classification logits,
use two independent dropout forwards during training:

`L = (CE(view1) + CE(view2))/2 + 1 * (KL(p1||p2) + KL(p2||p1))/2`.

Fix consistency weight 1, temperature 1, smoothing epsilon 0 and ordinary
dropout 0.15 before screening. Sum KL over classes and average over valid target
blocks. Both views receive gradients. Reset normal recurrent state per forward;
do not share DTH activity traces between views or samples. Respect original
burn-in/target masks, integer labels and dataset-specific temporal aggregation.
No input augmentation, FIR, time warp, sequence truncation, precision change,
new inference pass or extra trainable parameter is introduced.

The previous C005 SE-adLIF SSC screen reached 77.862820% ordinary test accuracy
for its matched baseline and 79.109018% for R-Drop at ten epochs (+1.246198 pp).
That pair used batch 512 and is not a DTH control. Conversely, C005 SHD speaker
development gave R-Drop 87.367491% test versus 90.150177% baseline (-2.782686 pp),
despite better validation. Smoothing combinations also failed to establish
complementarity. Retain these negatives; do not reuse their weights or claim
uniform robustness. This B choice is influenced by already exposed tests and
the entire adaptive study is therefore exploratory. Test exposure cannot be
erased by new training seeds. No prior DTH + R-Drop run was found in the targeted
configuration/failure audit. Do not repeat DTH + smoothing as a new hypothesis.

## Calibration, correctness and frozen setup

Use dataset-specific inheritance from `config/paper_baseline.yaml`. Retain DTH
cell settings and the ordinary reference behavior. Extend R-Drop only through
an explicit DTH-backbone opt-in. Check two-pass loss/gradients/masks on CPU and
CUDA, ordinary inference parity, recurrent-state reset, equal initial weights
and RNG, and invalid/mixed-backbone rejection. Calibration serves as a bounded
real-batch smoke check; none of its states may initialize main training.

Before main SSC training, measure disposable seed-117 DTH and DTH+B states on
real full-length fitting samples, including long sequences. Consider batches
256, 512 and 1024, stopping expansion when measured memory makes it unhelpful.
Use warm-up, synchronized optimizer-step timing, allocated/reserved VRAM,
updates/epoch and epoch estimates. Require at least approximately 15% headroom.
Select the fastest stable shared configuration; keep LR 0.006 unscaled.
Benchmark validation batch separately and verify predictions/CE consistency.
Reuse prior deterministic SSC data-loading calibration only after verifying
unchanged transform sources/tensors and adequate producer throughput; state
clearly which evidence is reused. Save calibration and source/config hashes.

Freeze the selected batch/validation batch/workers/precision before the main
comparison. One GPU training job at a time. Save fresh initial tensor/RNG
evidence for both arms. Use new outputs under
`results/experiments/dth_complement_v1/b001_rdrop_ssc_v1/` and distinct study
metadata; never overwrite canonical DTH or C005 results.

## Stage order and decisions

1. **SSC discovery, seed 42.** Audit old DTH settings and checkpoint coverage.
   Prepare one matched fresh DTH / DTH+B comparison with a hard ten-completed-
   epoch ceiling in the inherited 40-epoch schedule. Primary screening metric:
   clean official-validation accuracy; ordinary hard-label CE and curves support
   interpretation. Practical threshold: +0.50 pp versus matched DTH. Lower loss
   or a smaller training gap alone is insufficient. Freeze the rule and measured
   compute estimate before training. Do not tune consistency weight during a run.
2. **Bounded extension decision.** After both ten-epoch arms and their selected
   validation checkpoints are verified, extend both from their own `last.ckpt`
   to total 40 only if promising or genuinely inconclusive and still improving.
   Keep optimizer/scheduler/epoch state; do not restart or use weights-only
   continuation. Reject a flat or clearly harmful candidate with an explicit
   evidence-based decision. Never extend past the declared full budget.
3. **SSC exploratory test check.** Freeze candidate/settings and exact
   validation-selected checkpoint hashes before opening test scores. Evaluate
   the matched pair once at the chosen completed stage, ordinary single-pass,
   no TTA. Reuse an old control test only if all relevant settings and checkpoint
   identity match. Report actual epochs, paired seed-42 difference and clearly
   labelled per-seed deltas against the published mean. No test search over
   checkpoint epochs, seeds or inference variants. Record any test influence
   on a later B proposal explicitly as exploratory.
4. **ECG transfer, seed 42.** Only shortlist B after the SSC review. Use the
   same fixed mechanism with inherited ECG settings; verify signed input,
   labels/masks, exhaustive row separation and class coverage. Patient IDs are
   absent, so do not claim patient independence. Calibrate DTH pair batches
   64/128/256 before expensive work. Fresh matched controls are required if
   settings differ. Screen 50, extend to 100 when justified. Underlearned early
   controls cannot establish a large final gain. Full-budget comparisons need
   400 epochs. Validation-select, freeze and perform the exploratory test check;
   retain transfer regressions.
5. **Preserve canonical SHD*.** Before new seeds, use its own random-recording
   20% validation split, seed 42. Speaker-holdout weights and diagnostics remain
   separate. Screen fresh matched DTH / DTH+B at 30, optionally 75 when justified,
   with 300-epoch schedule retained. A validation drop beyond -0.30 pp is a
   practical preservation concern, not a statistical noninferiority test.
   Full-budget claims require 300 epochs and matching seeds.
6. **Freeze and finish ablations.** Only after cross-dataset evidence is
   promising, freeze B/settings and compare SE-adLIF / DTH / SE-adLIF+B / DTH+B.
   Reuse only demonstrably compatible controls. Expand to seeds 123/456 after
   freezing; report paired contrasts and interaction without assuming synergy.
   Complete dataset budgets before full-budget paper comparisons. More seeds
   may be needed for small effects. Measure costs with existing post-test tools;
   never retrain merely to repair a cost record.

If B001 fails, preserve it and investigate before proposing another B. Register
one justified next hypothesis at a time within remaining resources; no unlimited
queue or silent budget increases. A positive test does not authorize ignoring
failed validation gates, mismatched controls or another dataset's regression.

## Evidence, monitoring and stopping

Use flushed stage starts and one compact epoch line with train/validation
metrics, best epoch, LR, batch, updates, epoch time, peak VRAM and measured ETA.
Distinguish zero-based indices from completed epochs. Log decisions as ADVANCE /
EXTEND / REVISE / REJECT / BLOCKED with evidence and next action. Keep resolved
configs, baseline identity/deviations, best/last optimizer-bearing checkpoints,
metrics, hashes, `STATUS.json`, a compact registry and cumulative report beside
the study results. Monitor roughly every ten minutes or at errors/boundaries;
do not repeatedly dump logs or review the entire repository while waiting.

Report validation and test separately, all attempted candidates including
failures, per-seed values, mean/sample SD only when multiple independent seeds
exist, paired changes, practical effect, protocol deviations and measured costs.
Use signed local-minus-paper deltas only for compatible metrics and identify
single-seed comparisons to a published mean. Never promise significance,
superiority, complementary contributions or publication. An honest negative
result is acceptable. Preserve all active bounded jobs before stopping work.

## Resource observation before main launch

The first native accounting check after implementation reported 56,364 tokens
used, with no native hard limit. An agent-imposed operational hold and extra
approval question were introduced, then corrected: the original 50,000 request
was conditional on supported native configuration, not an existing hard cap.
The user already authorized this bounded study. The extra question is superseded;
no answer or elapsed time is treated as new approval. Proceed under the original
authorization and actual unbounded native Goal, without replacing/resetting it.
Preserve the previous hold record in the study evidence.
