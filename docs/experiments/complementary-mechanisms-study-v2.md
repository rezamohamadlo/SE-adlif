# Complementary mechanisms study v2: speaker-disjoint discovery

Effective user correction: 2026-11-11. Supersedes the discovery order in
[v1](complementary-mechanisms-study-v1.md); preserves its evidence. The full
objective remains two complementary ideas, Baseline -> A -> A+B, with a final
four-arm ablation. Existing `runner.py`/`run.py` remain the entry points.
The original study request explicitly authorizes the agent to launch, resume
and safely stop its own study jobs, overriding the repository's default
user-launch-only rule for this study. Unrelated processes remain protected.
The plan-update turn launched no training. No commit/push is authorized.

## Active-plan confirmation (2026-11-12)

The central problem is weak generalization, especially to unseen SHD speakers.
High random-split validation accuracy previously failed to predict official-test
performance. The stage order below governs subsequent experiments; canonical
SHD* validation is no longer the sole idea-screening criterion.

**Earlier plan-review process audit (historical):** no local Python training process was found in that
review. Both B002 baseline and label-smoothing ECG runs have completed their
bounded 50-epoch validation stage, under `label_smoothing_transfer_v1_ECG`
(with the explicit objective suffix for label smoothing). This is established
ECG row-split validation, not SHD speaker validation, canonical SHD*, or an
official-test evaluation. Stage receipts record tests disabled. Preserve their
50-epoch checkpoints, metrics, resolved settings and source evidence. No process
was launched, stopped or resumed during this plan update.

The ECG50 comparison recommends the already predefined matched 100-epoch review
because both controls are still underlearned; it does not establish a gain.
That is a possible next bounded stage, not a running job or an automatic
extension. Do not launch full 400-epoch training or broad replication merely
because a development stage has completed. Historical process observations and
mechanism proposals below remain historical, not current launch instructions.

**Frozen SHD manifests reverified against the official training file only:**
primary validation speakers 0/3, 6,521 training / 1,635 validation recordings;
sensitivity validation speakers 6/10, 6,920 / 1,236. Both retain identical
fingerprints, exhaustive original row indices, eight fitting speakers, no row
or speaker overlap, and all 20 classes on each side. No official test file was
read for this review. Never regenerate indices or search for a favorable pair.
The 0/3 pair has extensive tuning exposure and remains development; 6/10 is a
predefined shortlist-only sensitivity check, not another tuning target.

Compare fresh Baseline, A and A+B initially at training seed42, with identical
split, preprocessing, batch size, optimizer, learning-rate schedule, dropout,
selection rules and screening budget; record intended mechanism/objective
changes explicitly. B-only separates standalone regularization from synergy.
A checkpoint previously trained on a validation speaker is ineligible for clean
unseen-speaker assessment. Historical controls require every relevant setting,
source/model behavior, exact split and training exposure to match; a similar
score or model name is insufficient.

Advance on improved **clean held-out-speaker accuracy**, supported by ordinary
validation loss and learning curves. A smaller gap alone cannot advance an
idea. Shortlist first, then check the single predefined sensitivity pair and
verified SSC/ECG validation protocols. Do not fully train each idea under both
speaker-development and canonical protocols.

Official tests remain outside this loop. Speaker-development and canonical
SHD* runs require separate outputs, tables and training on their own splits;
never reuse speaker-holdout weights as canonical weights. Freeze mechanisms
before the fresh canonical four-arm comparison. Preserve active runs elsewhere
and assess their actual protocol and bounded stage before intervening.

## Primary endpoint and protocol separation

### C001 DTH with B002 smoothing: completed bounded development

B002's SSC40 and ECG100 transfer screens are closed. Smoothing improved SHD
speaker development, but SSC missed its practical threshold and ECG regressed;
two complementary mechanisms are not established. Preserve the completed
evidence in `B002_TRANSFER_STAGE_COMPLETION.json` beside the study results.
The subsequent distinct hypothesis is a fresh matched Baseline / DTH /
DTH+LS / LS primary0/3 screen, seed42. Follow the
[C001 preregistered plan](complementary-C001-dth-ls-v1.md), not the historical
launch/status observations above or below. Existing DTH conditions internal
thresholds; it is not the rejected input-gain mechanism. No novelty or
cross-dataset success is presumed. C001 must pass the same speaker-development,
shortlist-only sensitivity and transfer gates before canonical comparisons.

The root source-identity guard now includes the DTH implementation, and the new
study explicitly freezes its hidden-cell settings. Historical source snapshots
and completed result identities are retained unchanged; do not bypass their
guards to resume old studies under this source revision.

### C002 static versus dynamic thresholds: completed mechanism control

C001 primary30 and predefined6/10 sensitivity30 are now complete. Its
combination-over-LS accuracy differences are+1.53pp and+0.40pp respectively;
the latter misses the unchanged+0.5pp practical threshold. DTH alone failed
the primary comparison. No secondary tuning/extension or cross-dataset/final
advancement follows from these results. See
`C001_DISCOVERY_STAGE_COMPLETION.json` beside the study results.

The next [C002 primary-only mechanism control](complementary-C002-static-threshold-control-v1.md)
compares fresh LS, bounded-static-threshold+LS and DTH+LS. It freezes both
DTH activity gains at zero for the static arm, retaining trainable bounded
per-neuron thresholds and the same scan structure. This separates a possible
static-threshold benefit from activity-dependent effects, without claiming
inference savings or novelty. It uses the existing frozen0/3 training-only
split, seed42 and matched30epoch budget. Do not revisit6/10 for tuning.
All final four-arm, transfer, canonical own-split, replication and test gates
remain outstanding. Current live status belongs in the experiment registry,
per-run STATUS and process inspection, not historical observations in this guide.

### C003 static-threshold four-arm comparison: rejected bounded screen

C002 closed at30epochs: LS86.73%, static+LS87.77%, dynamic+LS87.03% on primary0/3.
Static improved over LS by1.04pp and over dynamic by0.73pp in that cohort;
dynamic-specific benefit was not supported. Keep the single-seed/development
caveats and C001's below-threshold secondary result. Official tests remain out.

The next [C003 full static factorial screen](complementary-C003-static-factorial-v1.md)
uses fresh Baseline/static-only/static+LS/LS, completing the missing standalone
static contrast and interaction before transfer. Shared settings and initial
30epoch ceiling remain fixed. A new opt-in initial-state evidence hook records
tensor/RNG hashes without changing weights, RNG, precision or deterministic
execution. Repeated seed42controls are not independent seeds; small synthetic
CUDA numerical differences do not establish the cause of the observed score
spread. Freeze source after non-training checks and disposable calibration.

### Latest discovery branch: C004 residual current normalization (closed)

C004's fresh matched primary30 screen is complete: Baseline83.12%, A83.00%,
A+B86.91%, smoothing-only B88.01%. A-baseline is-0.122327pp and AB-B is-1.100916pp;
the combination also has worse selected ordinary CE and late10mean than B.
Actual common initial tensors/RNG/settings, all30epochs, source269files,
best/last checkpoint identities and ordinary clean validation reproduction
were verified. Curves were inspected before rejecting this branch under its
declared screen. No75extension, conditional gain-control training, secondary
pair evaluation, SSC/ECG transfer or official testing follows. Preserve all
evidence; this bounded negative result does not rule out all normalization
methods. The experiment registry and C004_FACTORIAL_STAGE_COMPLETION.json
record closure. The setup description below is historical.

C003 closed with Baseline81.65%, static78.90%, static+LS86.61%, LS87.40% at30epochs.
Actual initial common tensors and CPU/CUDA RNG hashes matched across all four;
static alone and its increment over LS failed the declared screen. Preserve
C002's earlier positive result and this negative factorial; do not cherry-pick.
No extension or transfer of the static branch follows.

The distinct [C004 current-normalization hypothesis](complementary-C004-current-normalization-v1.md)
adds a bounded zero-initialized non-affine normalization residual to each
sample/frame's bias-free feed-forward current. It does not pool across time or
samples or alter the SE update. Zero input preserves baseline bias current;
no time warp, truncation or new mask is introduced. Normalization and zero-gated
residuals are prior art, and performance/generalization are hypotheses.
Parameter-matched current-gain controls are registered upfront; they must be
reviewed before a normalization-specific interpretation of any positive result.
Start the same fresh four-arm primary30 screen and preserve all usual gates.

Weak generalization, especially to unseen SHD speakers, is the central problem.
Random-split SHD* validation is no longer the sole idea-selection criterion.
The primary **development** endpoint is clean held-out-speaker accuracy, with
validation loss and learning curves as supporting evidence. A smaller
train/validation gap alone is not success. Final success still requires matched
ordinary-inference official-test accuracy; no test result selects ideas.

The earlier request for exploratory official-test evaluation is superseded by
the latest instruction to keep official tests outside this loop. No official
test evaluation was performed before this change. Speaker-development
checkpoints are explicitly ineligible as canonical SHD* test evidence.

## Frozen speakers and sample indices

Only `data/SHDWrapper/shd_train.h5` is read to create and verify these splits.
Both manifests were frozen before any v2 candidate evaluation; they include
original HDF5 row indices, source SHA-256, per-class counts and fingerprints.
There is no performance-dependent pair selection or fallback.

| Role | Training speakers | Validation speakers | Training / validation recordings |
|---|---|---|---:|
| Reused development pair | 1,2,6,7,8,9,10,11 | 0,3 | 6,521 / 1,635 |
| Predefined pair sensitivity | 0,1,2,3,7,8,9,11 | 6,10 | 6,920 / 1,236 |

For both pairs, speaker and recording intersections are empty, train+validation
exhaust the 8,156 official training rows, and each side covers all 20 classes.
Primary 0/3 has extensive historical tuning exposure and is labelled
development, not independent confirmation. Secondary 6/10 was fixed without
looking at candidate performance; it is a pair-sensitivity diagnostic, not
patient/speaker-population proof. No search for a better pair is permitted.

Only shortlisted methods plus the matched baseline may enter the secondary
pair, enforced by `SHORTLIST.json`. They **start fresh**: weights learned on
the primary pair include speakers 6/10 and must not be reused there. Likewise,
the earlier canonical random-split checkpoints saw many rows from each held-out
pair. Their use as unseen-speaker validation weights is prohibited. An exposure
audit is beside the new manifests.

## Initial mechanisms and matched conditions

- **Baseline:** unchanged two-layer SE-adLIF.
- **A001:** channel-specific causal residual FIR before projection, lags
  1,2,4,8 frames, zero-initialized identity. The v1 diagnostic found learned
  constant-input gain around 2.45, so this must not be called pure delay
  learning; held-out speaker performance, not the old random-split peak,
  determines whether it advances.
- **B001:** training-only independent 10% event deletion before the filter,
  hypothesized to reduce rate/gain reliance. A+B is compared with A, and the
  later B-only arm identifies generic augmentation benefits. This is not a
  novelty claim: [EventDrop, IJCAI 2021](https://ijcai.org/proceedings/2021/97)
  is prior art for event removal. Here deletion is binomial per event in count
  bins, rather than entire occupied-bin deletion; its combination with this
  FIR and the matched SE control is a new hypothesis relative to previous
  DTH/LG-DTH augmentation combinations. There is no assurance it will help.

All compared arms use the same frozen pair and indices, training seed42,
training/validation batches1024, LR0.01, Adam, dropout0.15, 32-bit/high matmul
precision, preprocessing, full300epoch schedule metadata and screening ceiling.
Paper baseline inheritance remains authoritative; split/batch/augmentation
deviations are explicit. Historical controls do not match this entire factorial
setup and are not reused. No pretrained or canonical weights initialize v2.

The initial shared ceiling is30 completed epochs. Extend technically sound,
improving or inconclusive comparisons to75 **for all compared arms**; do not
fully train every proposal under both protocols. Initial practical screen
threshold is +0.50 pp clean held-out-speaker accuracy (A-Baseline and A+B-A),
supported by loss/stability. This is a decision rule, not significance evidence.
If A fails but A+B improves, do not assume complementarity; inspect B-only and
revise the hypothesis. Negative pair sensitivity or dataset transfer is retained.

The old frame-warp augmentation operated on minimum-padded sequences, so it
could change artificial quiet duration. It is not used here. B001 never warps
time or changes tensor length/masks; padding zeros remain zeros, ECG negative
end markers remain unchanged, and validation/inference are clean. Augmentation
is drawn afresh during each training forward, outside deterministic caching.
No physical duration is inferred from the last nonzero frame.

## Revised stage order

0. Audit actual active jobs and preserve v1 canonical evidence. Freeze both
   training-only speaker manifests and verify overlap/class coverage.
1. Fresh primary 0/3 development: Baseline, A and A+B, seed42, matched30epoch
   screen; extend the same comparison to75 only when justified.
2. Fresh secondary 6/10 pair-sensitivity check: baseline and **only shortlisted**
   candidates, one predefined pair and matching budgets. No weights cross pairs.
3. Check promising mechanisms on SSC official validation: calibrated matched
   control/candidates, seed42, 10->40 epochs. Investigate regressions before B
   investment or broad replication. Preserve all unsuccessful outcomes.
4. Check ECG with verified established row separation/labels/masks: matched
   seed42, 50->100 when justified, full400 only for frozen comparisons. Patient
   IDs are absent, so patient independence is not claimed.
5. Freeze A, B, settings, practical test thresholds and final evaluation plan.
   If necessary, revise mechanisms using development evidence, never tests.
6. Run the frozen four-arm **canonical SHD*** comparison on its own random
   20% validation split, split seed42, full300epoch budget. Train those weights
   on that canonical split; do not relabel/reuse speaker-development weights.
   Complete matching four-arm SSC40/ECG400 evidence. B-only is mandatory.
7. Replicate frozen comparisons with training seeds123,456, preserving split42
   where applicable. No test-driven idea selection; additional seeds if needed.
8. Evaluate official tests once under the frozen ordinary-inference plan;
   report all datasets, train/validation/test metrics, paired differences,
   interaction, mean/sample SD, uncertainty, paper deltas and measured costs.

Speaker-development and canonical results remain in separate roots and tables.
Never subtract a speaker-validation score from the paper's SHD* test mean.
All earlier generalization, uncertainty, staging, calibration, cost and
checkpoint-safety obligations in v1 remain in force where not superseded here.

## Existing processes and evidence at correction

### Historical bounded-stage audit (2026-11-11)

The live launcher uses
`config/studies/complementary_transfer_A002_gain_v1.yaml`: fresh Baseline and
gain-only A002 on SSC, training seed42, official SSC validation, batch1024,
LR0.006, a10epoch ceiling within the40epoch schedule, and tests disabled.
The baseline's first completed epoch is recorded in the launcher log. This is
stage3 cross-dataset development, not random-split SHD*, speaker-holdout SHD,
or official-test evidence. Preserve the process and let this bounded stage
finish; this audit does not launch, resume, stop or extend training.

The active plan remains the stage order above. Baseline/A/A+B comparisons for
subsequent mechanisms must use fresh, matched speaker-development runs; the
gain-only diagnostic does not replace the required A+B and final B-only arms.
Historical controls require an exact split and configuration match, including
preprocessing, optimization, budget and checkpoint-selection rules. Previously
exposed speaker weights cannot serve as clean held-out-speaker controls.

Audit evidence: live launcher and training-child command lines, and
`results/experiments/complementary_speaker_dev_v2/a002_ssc_stage010_stdout.log`.
Progress is a timestamped observation, not a claim that the queue has completed.

### Conditional mechanism check after transfer

If the full SSC40epoch screen gives no practical improvement, do not extend it
past its declared budget or invest heavily in B. First run the preregistered
gain-only control on the **same frozen primary0/3development pair**:
`config/studies/complementary_gain_control_A001_v1.yaml`. It compares fresh
Baseline/FIR/gain-only at seed42 and30epochs, sharing all optimization and
preprocessing settings. The gain-only mode retains four coefficients per
channel and identity initialization, but mixes no temporal history. It tests
an interpretation of A rather than establishing a new timing-specific result.
The secondary pair is not a tuning target, and no favorable pair is sought.

SSC findings, including regressions, remain in the cumulative evidence. ECG
remains explicitly pending; this diagnostic does not replace a dataset in the
final SHD*/SSC/ECG assessment. No two-dataset improvement is claimed from SHD
development alone. Source changes for this control occur only after the active
transfer queue closes and its source, checkpoints and metrics are preserved.

The canonical v1 seed42 baseline/A001 each completed75 epochs. They are
historical random-split development evidence, not unseen-speaker controls.
The attempted150epoch launcher exited **before training** with
`Source revision changed; refusing to resume this study`. No model training was
active at the audit, so no training was stopped or discarded. Source guards
were not bypassed. The ongoing SSC training/validation-only download is
preserved; it is not a GPU training job. No unrelated job is altered.

## Artifacts and execution

The gain control can advance as **A002 input conditioning** only through the
same development gates; do not rebrand it as a timing mechanism. Its follow-up
uses `config/studies/complementary_transfer_A002_gain_v1.yaml`, with fresh
matched controls under `transfer_A002_gain_v1/`. SSC uses10->40epochs,
official validation, batch1024/eight workers/LR0.006; ECG uses50->100staging
before any frozen400epoch comparison, its established row validation,
batch256/LR0.01 and separate calibration. Practical screening thresholds and
all qualifications are preregistered in the results root before training.
Keep the failed SSC FIR transfer and all earlier source versions as evidence.

## B002: opt-in train-only label smoothing after A002 transfer

A002's completed discovery transfer is recorded in
`results/experiments/complementary_speaker_dev_v2/A002_TRANSFER_STAGE_COMPLETION.json`.
SSC40has only a nominal+0.10pp validation difference. ECG100has a preliminary
+1.08pp difference with an extra baseline77epoch interruption; it is not a
full400epoch or final-test result. Both speaker-pair gains remain development
evidence. No two-dataset final-test improvement is established.

The next hypothesis is B002uniform label smoothing, epsilon0.1, training only.
It targets the primary-speaker confidence/CE concern while A002conditions inputs.
These are established components; neither novelty nor synergy is assumed.
The opt-in objective is `classification_label_smoothing_v1`. Integer labels,
ignored targets, temporal aggregation, neuron dynamics and ordinary evaluation
predictions remain unchanged. `train_loss` is the smoothed objective for B002;
`train_hard_ce` is reported separately. Validation uses ordinary hard-label CE.
The coefficient is distinct from the neuron's surrogate-gradient `alpha`.

`config/studies/complementary_B002_primary_v1.yaml` defines fresh matched
Baseline/A/A+B (SE/Gain/Gain+LS), primary speakers0/3, seed42, batch1024and
30epochs within the300epoch schedule. Keep all shared optimization/dropout/
preprocessing settings matched. Only the intended training objective differs;
record that deviation and its distinct protocol metadata. B-only LS is configured
for the required factorial ablation. All experiment configs inherit the paper
baseline through their existing base configs and the runner comparison group.

Advance by clean A+B-minus-A accuracy (+0.5pp practical threshold), supported by
ordinary validation CE/curves. A smaller gap alone is insufficient. If A no longer
improves over the fresh baseline, inspect B-only and revise the interpretation
rather than claiming complementarity. Extend all compared arms30->75only when
justified. Only shortlisted candidates and required matched controls enter the
existing fixed6/10sensitivity diagnostic with fresh weights; no pair search or
tuning there. Transfer configs are prepared separately, but SSC/ECGobjective
calibration confirmations and primary selection precede any transfer launch.

Preregistration: `results/experiments/complementary_speaker_dev_v2/B002_SCREEN_PREREGISTRATION.json`.
Complete correctness, default-parity and runner preflights before freezing source
and launching the real bounded queue. No synthetic training smoke run is needed.
Official tests remain outside development; canonical SHD* four-arm training and
matching-seed replication still follow a later freeze.

Study root: `results/experiments/complementary_speaker_dev_v2/`.
Configs: `config/studies/complementary_speaker_dev_v2.yaml` and
`config/studies/complementary_speaker_sensitivity_v2.yaml`.
Cross-dataset A001 transfer uses the separate
`config/studies/complementary_transfer_A001_v2.yaml`, with outputs under
`results/experiments/complementary_speaker_dev_v2/transfer_A001/`.
Its SSC protocol uses official training/validation, calibrated train/validation
batch1024, LR0.006 unchanged, eight persistent workers, and opt-in spawn-safe
preprocessing verified to preserve tensors, labels and masks exactly. It does
not apply an SHD speaker split to SSC. Main transfer weights start fresh;
historical SSC weights were used only for numerical batch calibration.
Calibration evidence and the pre-launch advancement rule are beside the study
root. Do not use the primary/sensitivity SHD configs to launch transfer jobs.
The completed SHD source snapshot is retained; changes for the separate transfer
stage never bypass historical resume guards. Dataset defaults remain unchanged.
Each pair has its own frozen manifest, cache identity and seed outputs.
Keep STATUS, registry, epoch CSV/JSONL, best/last and stage snapshots, resolved
configs and cumulative reports. Source/provenance guards stay enabled; freeze
implementation before launching a queue and diagnose incompatible revisions
rather than modifying historical records. Every direct resume also verifies
the checkpoint belongs to the same run, split and configuration.

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF FIR_SE_adLIF FIR_THIN_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_speaker_dev_v2.yaml --stage-ceiling 30
```

This is the command for the authorized bounded discovery screen, not the final
paper comparison or test evaluation. The local checkout has `.ven`; use
`.venv` instead on a machine with that environment directory. Append `--dry-run`
for the non-training preflight that has been verified locally.

## Subsequent discovery: C005 dropout consistency

C004 closed negatively: A-baseline -0.122327pp, AB-B -1.100916pp, with worse combined CE/late10mean. No extension or transfer. The next distinct opt-in hypothesis is [C005](complementary-C005-dropout-consistency-v1.md): average two independent dropout supervised losses plus fixed alpha1 symmetric KL. B remains epsilon0.1 smoothing. It preserves ordinary single-pass inference and SE dynamics. Fitting-only susceptibility was modest, not causal evidence. Registered alpha0 two-pass controls are required for a positive KL-specific interpretation. Complete non-training checks and real two-graph calibration before freezing source and launching the fresh matched primary30 screen. All speaker/sensitivity/transfer/canonical/test gates above remain in force.

## C005 primary closure and fixed-pair sensitivity

Primary30 verified Baseline82.14%, RDrop83.85%, RDrop+LS86.30%, LS88.44%; combination rejected. Predeclared alpha0two-pass control81.04%; RDrop-control+2.813456pp, ordinaryCE-0.106498, late10mean+1.602446pp, bothspeaker contrastspositive. RDrop still worsens speaker3 versus the ordinary baseline; no uniform robustness or testclaim. All5actualinitialweights/RNG/settings and stage30/checkpoint/config/source/validation evidence verified before closure.

Only A005alpha1,epsilon0 is shortlisted for fresh Baseline/A sensitivity on predefined6/10, original frozenindices, seed42, batch1024/LR.01/dropout.15/full300schedule/30onlyceiling. Primaryweights saw6/10 and must never initialize this check. No failedcombination/LS/alpha0pair sweep, secondary tuning or extension. Require+0.5pp with ordinaryCE/curves support; preserve either outcome and accumulatedpair-exposure caveat. VerifiedSSC/ECGvalidation remains after this gate; finalcanonicalSHD*trainingmuststartfreshonitsownsplit. Officialtestsremainoutside development. See C005_SHORTLIST.json and C005_SENSITIVITY_STAGE030_PREREGISTRATION.json beside results.

## C005 fixed-pair closure and SSC transfer preparation

Fresh6/10 sensitivity30 verified Baseline83.82%, RDrop85.52%, +1.699030pp; ordinaryCE-0.008327, late10mean+1.116505pp. Speaker6-1.342282pp and10+3.422053pp: weightedgain,notuniformspeakerrobustness. Allsource278/protocol/initialstate/config/checkpoint/epoch/ordinaryvalidationchecks passed andcurvesinspected. No secondarytuning/extension.

Aonly nowqualifiesforbounded freshSSCseed42 Baseline/RDrop10,then40onlyifjustified. Useofficialvalidation anddataset-specificmean_membrane_potentials loss;alpha1/epsilon0fixed. Actualone/two-graph256/512calibration andindependentvalidationbatchbenchmark mustprecede frozencommonbatch;LR0.006unchanged. IsolatedSSCstudyfile iscomplementary_C005_dropout_ssc_v1.yaml. ExistingcanonicalSSCcontrolremainsintact;batchchangesrequirenewmatchedcontrol. NoB/combinationadvancement,ECGlaunch,canonicalSHD*trainingorofficialtestaccess yet. SeeC005_SENSITIVITY_STAGE030_DECISION.json andc005_dropout_ssc_v1/SSC_TRANSFER_PREREGISTRATION.jsonbesideresults.
