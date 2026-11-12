# Speaker-disjoint development v2: cumulative evidence

Clean held-out-speaker validation, not canonical SHD* or official test. Primary holdout0/3 is development due historical exposure. Seed42; all arms fresh; shared batch1024/LR0.01/dropout0.15/full300epoch scheduler, stopped at30. No official tests were opened.

| Arm | Selected completed epoch | Training accuracy at selection | Clean held-out accuracy | Validation loss at selection | Final training / held-out accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 23 | 98.60% | 81.59% | 0.961734 | 99.80% / 77.25% |
| A | 26 | 99.63% | 82.94% | 0.785733 | 99.84% / 82.32% |
| A+B | 22 | 97.84% | 81.22% | 1.112529 | 99.33% / 79.27% |

A minus Baseline: +1.34557 pp. A+B minus A: -1.71254 pp. Differences computed from unrounded metrics.

**Decision:** Advance A only, with a fresh matched baseline, to the already frozen6/10 holdout at the same30epoch budget. B001 event deletion is unpromising under this screen; preserve its evidence. No75epoch extension is justified merely to rescue the combination. No claim that A and B are complementary.

Support: A selected loss0.785733 versus baseline0.961734; final10epoch means and sample SD across epochs are saved in PRIMARY_STAGE030_COMPARISON.json. Across-epoch variation is not across-seed uncertainty. Training metrics include dropout and augmentation; they are not clean training-set inference results. Smaller gaps alone do not determine advancement.

Evidence: each stages/030/RESULT.json and checkpoint hashes, checkpoint epoch/hyperparameters,30 distinct validation CSV epochs, EPOCH_PROGRESS.jsonl, matched saved configs and frozen split manifest. Source provenance verified unchanged. No numerical paper delta applies to incompatible speaker-validation metrics.

Canonical random-split75epoch evidence remains separate under results/experiments/complementary_mechanisms_v1/ and cannot serve as an unseen-speaker control. The predefined secondary6/10 check uses fresh weights and no pair search. Later SSC/ECG transfer, frozen canonical four-arm training, matching seeds and ordinary official tests remain pending.

[Primary learning curves](PRIMARY_STAGE030_CURVES.png). Secondary pair check completed; see below.

## Predefined6/10 pair sensitivity

Fresh matched controls; seed42;30epochs. No pair search and no reuse of primary weights.

| Arm | Selected completed epoch | Training accuracy at selection | Clean held-out accuracy | Validation loss at selection |
|---|---:|---:|---:|---:|
| Baseline | 28 | 99.24% | 83.41% | 0.695776 |
| A | 26 | 99.24% | 89.40% | 0.347613 |

A minus matched baseline: +5.98705pp. The gain has the same sign on both predefined pairs. Do not average these into an across-seed estimate: they share one seed and overlapping training samples.

**Decision:** Advance A to a fresh, matched SSC official-validation10epoch screen after throughput and validation-batch calibration. B001 remains unpromising; neither synergy nor official-test generalization is established. Source/config/checkpoint/CSV audits passed for both secondary arms.

## Mechanism and transfer readiness

A001_FILTER_DIAGNOSTICS.json shows mean constant-input gain around2.25 on both speaker pairs; the improvement cannot yet be attributed specifically to temporal filtering. A gain-only/normalized-filter control is needed before a timing-specific claim.

SSC transfer preregistration: SSC_STAGE010_PREREGISTRATION.json. Batch1024 and8spawn-safe workers were separately measured; exact preprocessing parity verified. HistoricalSSC weights were used only as a numerical validation-batch probe, never as matched control or initialization. Disk caching was tested and rejected because it was slower here. Main SSC runs start fresh. No test data were opened.

SSC10epoch queue completed; matched comparison is below. Live log: ssc_stage010_stdout.log. Main baseline config independently verified fresh initialization, batch1024,8workers,LR0.006,full40epoch schedule and no official tests.

[Learned FIR response](A001_FILTER_RESPONSE.png): descriptive linear-filter response, not a spiking-network transfer function or timing-specific causal evidence.

SSC_VALIDATION_SPEAKER_AUDIT.json verifies zero overlap between1864training and256official-validation stored speaker IDs; all35classes occur on both sides. SSC official validation therefore provides another unseen-speaker development check. Official test files were not opened.

## SSC10epoch official-validation screen

Stored speaker IDs are disjoint;1864training/256validation speakers. Fresh matched batch1024,8workers,LR0.006,dropout0.15,full40epoch schedule.

| Arm | Selected completed epoch | Training accuracy at selection | Clean validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| Baseline | 10 | 79.02% | 78.05% | 0.722434 |
| A | 10 | 80.76% | 78.28% | 0.706704 |

A minus Baseline: +0.23044pp, below the declared+0.50pp practical threshold. **Inconclusive:** both selected the final10th epoch and continued improving. Lower loss is supportive preliminary evidence, not proof of generalization or an official-test improvement.

**Decision:** One planned matched full-resume extension to40epochs, restoring each own optimizer/scheduler/epoch. No hyperparameters, preprocessing or split changes. No further SSC extensions beyond40 to rescue a method. Preserve10epoch snapshots; final test remains outside development.

The matched SSC40epoch extension is now active: Baseline restored all own checkpoint states; A is queued for the same full-resume procedure. Live log: ssc_stage040_stdout.log. Both pause at10; restored optimizer/scheduler/epoch does not guarantee the same dropout/data-order randomness as uninterrupted training. Original10epoch snapshots and logs remain preserved.

[SSC10epoch learning curves](SSC_STAGE010_CURVES.png). These curves are frozen to the completed10epoch stage while the40epoch extension proceeds.

A001_GAIN_FACTORIZATION_AUDIT.json verifies first-projection reparameterization on32real training records (relative L2 error5.25e-8). It does not compare fresh models or prove spiking-trajectory equality. A001_GAIN_CONTROL_PREREGISTRATION.json specifies the pending parameter-matched gain-only mechanism control; no new training has been launched for it.

The active aggregate STATUS plan link is hard-coded to v1 by the frozen callback. Use the saved study_file and evaluation_protocol for the actual v2 protocol; this display issue is recorded for a later source revision.

## SSC40epoch extension: baseline complete, A pending

Baseline completed40/40training epochs. Selected completed epoch24: train86.40%, clean validation81.26%, validation loss0.618796. Checkpoint hashes,40distinct CSV epochs and2920restored optimizer updates verified. A is still running, so no matched40epoch model difference is available. The legacy summary sentence calling this a short screen is a known display bug; the authoritative receipt correctly marks full budget complete. No official test evaluated.

## Prepared follow-up implementation (not active)

The isolated cost draft passed26CPUtests and its patch passes git-apply checks; it includes strict registered-model loading, executed FIR scalar work, analog first-projection accounting and incompatible-cache refusal. The isolated parameter-matched gain-only control and reporting fixes pass60combinedCPUchecks with that cost draft. Shared classifier initialization and RNG consumption match FIR. Patches/manifests are under cost_draft/ and gain_control_draft/. Neither patch is applied to the active checkout, no gain-control training has started, and real-manifest preflight remains required after the source-frozen SSC queue closes. Generated checkout copies are not intended for Git.

## Completed SSC40epoch validation comparison

Both arms completed40/40epochs, with full-state resume after10. Source, resolved shared settings, checkpoint hashes and40distinct CSV epochs verified.

| Arm | Selected completed epoch | Training accuracy at selection | Clean validation accuracy | Validation loss at selection |
|---|---:|---:|---:|---:|
| Baseline | 24 | 86.40% | 81.26% | 0.618796 |
| A | 33 | 87.60% | 80.92% | 0.665969 |

A minus Baseline: -0.34065pp. No SSC improvement under this protocol/seed; selected validation loss is also worse. This is validation-only evidence, not an official-test result or across-seed significance finding.

**Decision:** Do not extend SSC past40 or claim two improved datasets. Preserve positive SHD development and negative SSC transfer evidence. Run the preregistered primary0/3gain-only mechanism control before heavier B investment; ECG remains explicitly pending. No favorable-pair search or official tests.

The exact completed transfer source/environment is archived in SSC_TRANSFER_SOURCE_SNAPSHOT.zip/.json before applying the next opt-in source revision. Historic summary wording is a known display bug; authoritative receipts correctly show full budget complete.

[Full40epoch SSC learning curves](SSC_STAGE040_CURVES.png). All recorded losses/accuracies are finite. Full-budget summary wording has now been corrected against each authoritative receipt; original summaries remain in each seed summary_history/ directory. Numeric evidence and historical source records were not changed.

The prepared cost/gain-control changes are now applied after the SSC queue closed. Actual-checkout checks:77tests, real-manifest3arm dry run, unrelated-working-directory dry run and formatting checks passed. The new fresh Baseline/FIR/gain-only30epoch queue is active under gain_control_A001_v1/, with live output gain_control_stage030_stdout.log. Official tests remain untouched.

Same-seed replay audit: saved baseline computational settings match the original primary run, but some epoch metrics differ (see GAIN_CONTROL_STARTUP_AUDIT.json). Local PyTorch documentation flags CUDA scatter-reduction nondeterminism; this is not an isolated proof of the observed cause. Repeated seed42is not an independent seed, and small single-seed differences remain preliminary. Fresh3arm controls remain matched within this new source version.

## Primary gain-control study: completed30epoch screen

All three arms start fresh under the same updated source; repeated seed42is not an independent seed. Primary0/3is development, not final confirmation.

| Arm | Selected completed epoch | Training accuracy at selection | Clean held-out accuracy | Validation loss at selection |
|---|---:|---:|---:|---:|
| SE_adLIF | 18 | 96.84% | 81.04% | 0.714408 |
| FIR_SE_adLIF | 24 | 98.84% | 81.96% | 0.837363 |
| GAIN_SE_adLIF | 27 | 99.32% | 82.87% | 1.014091 |

FIR-Baseline +0.91743pp; gain-only-Baseline +1.83486pp; FIR-gain-only -0.91743pp.

No timing-specific accuracy benefit is established: a static gain-only control did better here. Gain-only validation loss is substantially worse, so its accuracy gain is not an unqualified improvement. Both pass the accuracy threshold and have better late-curve means than baseline, motivating only the already predefined6/10pair-sensitivity diagnostic. No hyperparameter tuning or pair search on that diagnostic. All secondary weights start fresh; primary weights saw6/10. ECG remains pending and official tests stay outside development.

Fixed6/10gain-sensitivity queue is active under gain_sensitivity_A001_v1/ (Baseline/FIR/gain-only, all fresh,30epochs, same seed42). Shortlist was frozen from primary accuracy evidence with worse-loss concerns retained. Gain_sensitivity_stage030_stdout.log records live output. No settings are tuned on this pair. ECG remains pending.

[Primary gain-control learning curves](GAIN_CONTROL_STAGE030_CURVES.png). Gain-only improves primary accuracy but has worse selected loss; the curves and pending primary-only calibration diagnostic qualify interpretation. The predefined6/10check is active; no test results are available.

## Gain-control predefined6/10pair sensitivity: completed

All3arms freshly trained for30epochs on the secondary8speakers; no primary weights reused.

| Arm | Selected completed epoch | Training accuracy at selection | Clean held-out accuracy | Validation loss at selection |
|---|---:|---:|---:|---:|
| SE_adLIF | 26 | 98.44% | 86.81% | 0.587373 |
| FIR_SE_adLIF | 18 | 98.01% | 88.59% | 0.460486 |
| GAIN_SE_adLIF | 29 | 99.71% | 88.83% | 0.511097 |

FIR-Baseline +1.77993pp; gain-only-Baseline +2.02265pp; FIR-gain-only -0.24272pp.

This is predefined pair sensitivity, not independent final-test or across-seed confirmation. The primary-only confidence diagnostic is next; no settings are tuned on6/10. ECG remains pending and SSC FIR regression remains reported.

## Primary0/3confidence diagnostic (no fitting/TTA/test)

All selected checkpoint accuracies were reproduced to floating-point rounding. Confidence is derived from the ordinary saved readout/masks; ECE uses15fixed equal-width bins.

| Arm | Cross-entropy | ECE15 (%) | Brier (sum over classes) | Mean confidence on wrong predictions (%) |
|---|---:|---:|---:|---:|
| SE_adLIF | 0.714408 | 11.01 | 0.306321 | 79.06 |
| FIR_SE_adLIF | 0.837363 | 12.21 | 0.300939 | 82.65 |
| GAIN_SE_adLIF | 1.014092 | 11.70 | 0.289137 | 83.52 |

Gain-only cross-entropy is worse, but its Brier score is better; calibration evidence is mixed rather than uniformly worse. Wrong predictions are more confident. ECE is a bin-dependent single-development-split estimate, not independent uncertainty. These diagnostics neither change predictions nor replace selected accuracy records.

**Next candidate A002:** learned input gain as a factorized input-conditioning/optimization hypothesis. It adds no independent representational capacity beyond a linear first projection; no timing-specific or novelty claim. Both predefined speaker-pair accuracy gains justify only matched cross-dataset validation screens. SSC FIR failure remains evidence, ECG is pending, and no two-dataset/test improvement is claimed.

A002 transfer preregistration: A002_TRANSFER_PREREGISTRATION.json. Fresh matched SSC10epoch controls are next; conservative real-training batch1024confirmation passed (finite gradients,38.6%headroom). ECG50epoch screening remains planned after its own data/calibration checks. The final three-dataset assessment is unchanged; no test data have been opened.

## A002 SSC transfer: baseline screen complete, candidate running

The fresh matched baseline completed its declared10epoch screen. Validation-selected
accuracy is77.908027% at completed epoch10 (checkpoint index9), validation loss
0.724866 and training accuracy79.926956%. This is preliminary validation evidence,
not a completed40epoch result or official-test score. Both best/last checkpoint
hashes,730optimizer updates and all10distinct CSV validation epochs were verified;
see `transfer_A002_gain_v1/SE_adLIF/SSC/seed_42/stages/010/VERIFICATION.json`.

The gain-only candidate started fresh under the same dataset-specific settings.
Its result and the advancement decision remain pending. No early cross-arm
ranking or extension is justified yet. The GPU remains reserved for this queue;
ECG calibration/training waits for it to finish.

ECG's established seed42row split is frozen in `ECG_A002_SPLIT_AUDIT.json`:
588training/30validation records, no shared row or identical input across splits,
all6classes represented on both sides. Patient identities are unavailable;
record separation does not establish patient independence. Original argmax label
semantics, all1301timesteps and negative end markers remain unchanged. Class-sensitive
diagnostics must not treat correlated timesteps as independent records.

## A002 SSC10epoch comparison complete: extend matched arms to40

| Arm | Selected completed epoch | Train accuracy at selection | Clean validation accuracy | Validation CE at selection |
|---|---:|---:|---:|---:|
| Baseline | 10 | 79.93% | 77.91% | 0.724866 |
| Gain-only A002 | 9 | 79.99% | 78.49% | 0.702578 |

Gain minus baseline is **+0.58110pp**, exceeding the predeclared practical
screen threshold+0.5pp. The gain arm's selected CE is also lower. Both arms
completed10epochs; selected checkpoints differ from final epochs as shown.
This is a single-seed preliminary validation result, not significance evidence
or an official-test improvement. Best/last hashes, selected epochs, optimizer
steps and10distinct CSV validation epochs were checked against receipts.
The saved resolved-config audit confirms only intended mechanism/provenance
differences. [Learning curves](A002_SSC_STAGE010_CURVES.png) and
`A002_SSC_STAGE010_COMPARISON.json` retain unrounded evidence.

**Decision: EXTEND.** Resume both arms from their own last checkpoints to the
declared40epoch budget, retaining the full scheduler and all10epoch snapshots.
This restores optimizer/scheduler state; it does not establish uninterrupted
RNG equivalence. No settings changed. Resume preflight passed for both arms;
the prior queue was terminal and no other training job was live at launch.
New output is `a002_ssc_stage040_stdout.log`/`a002_ssc_stage040_stderr.log`;
launch receipt is `A002_SSC_STAGE040_LAUNCH.json`. Additional compute estimate
is5580seconds plus startup, based on about93seconds per epoch per arm.
ECG waits for this queue and its own calibration; B002 remains only a conditional
hypothesis. No official test data were opened and no test ranking selected this
extension.

Final generalization endpoints are predeclared in
`FINAL_TEST_GENERALIZATION_ENDPOINTS.json`: alongside overall ordinary test
accuracy, report SHD seen/unseen-speaker and per-speaker diagnostics using
membership in the actual fitting split. This requires future prediction/row
alignment checks and does not authorize test access during development.
No subgroup implementation or subgroup test result is claimed yet.

The later canonical SHD* sample split is now explicitly frozen in
`CANONICAL_SHD_SPLIT_MANIFEST.json` and verified against the exact
`torch.utils.data.random_split` operation used by the current datamodule:
6,525fitting/1,631validation recordings, split seed42, no row overlap,
all20classes on both sides. All10training speakers occur on both sides;
canonical validation therefore does not establish unseen-speaker transfer.
This freezes data indices only, not the final mechanisms or hyperparameters.
Future canonical arms require fresh training and a runtime split-identity check;
speaker-development weights remain ineligible. No official test was opened.

## A002 SSC40: baseline complete, gain resume active

Baseline completed all40epochs. Its validation-selected checkpoint is completed
epoch37 (index36): training88.679904%, validation81.454766%, validation
CE0.642571. At final epoch40, training is88.570207% and validation79.531109%.
The late10epoch validation mean is80.770464%. Best/last checkpoint hashes,
selected epoch,2920final optimizer updates and all40distinct CSV validation
epochs were verified. The10epoch checkpoint snapshots remain unchanged.
See `transfer_A002_gain_v1/SE_adLIF/SSC/seed_42/stages/040/VERIFICATION.json`.

The gain-only arm is now fully resumed from its own last checkpoint, with the
same40epoch ceiling and saved optimizer/scheduler state. Its full-budget score
and the transfer decision remain pending. Do not compare its10epoch score with
the baseline40epoch score as a matched result. No official-test score exists
for this study, and no new source changes were applied during this queue.

## A002 SSC40 comparison complete

| Arm | Selected completed epoch | Training accuracy at selection | Clean validation accuracy | Validation CE at selection |
|---|---:|---:|---:|---:|
| Baseline | 37 | 88.68% | 81.45% | 0.642571 |
| Gain-only A002 | 24 | 85.39% | 81.55% | 0.624876 |

Gain minus baseline is **+0.10019pp**, below the predeclared practical+0.5pp
threshold. The earlier10epoch+0.58110pp advantage did not persist at that size
over the full budget. Selected validation CE is lower, but this is only a small
single-seed nominal accuracy difference, not a robust or official-test gain.
Late10epoch means are80.77046% baseline and80.85162% gain-only. Final epoch
accuracy is79.53111% baseline and81.08406% gain-only; checkpoint selection uses
the best validation checkpoint, not whichever final metric looks better.

Both arms completed40epochs with40distinct validation CSV epochs,2920final
updates and verified best/last checkpoint hashes. The10epoch snapshots are
unchanged. Resolved configs match except the intended gain/provenance fields.
Unrounded evidence is in `A002_SSC_STAGE040_COMPARISON.json` and the per-seed
stage040verification files; [full learning curves](A002_SSC_STAGE040_CURVES.png)
show the matched pause/resume boundary. Independent training seeds completed:1;
across-seed sample SD is unavailable. Repeated controls are not extra seeds.

**Decision:** No SSC extension beyond40 and no claim of practical cross-dataset
improvement. Preserve this weak positive result alongside A001's negative SSC
transfer. Perform the already planned current-source ECG calibration, then fresh
matched50epoch ECG controls for bounded transfer information before B investment
or replication. `ECG_A002_CALIBRATION_PREREGISTRATION.json` freezes that check.
The SSC queue is terminal; its exact source/package/HEAD/diff evidence is retained
in `A002_SSC_SOURCE_SNAPSHOT.zip/.json` and `A002_SSC_TRACKED_DIFF.txt`.
Official tests remain outside idea selection and were not opened.

## Current-source ECG calibration passed; A00250epoch screen active

Disposable real-training calibration (seed117, all1301timesteps, original labels
and masks, LR0.01) selected batch256. No calibration weights/optimizer state
initialize the main comparison.

| Training batch | Samples/second | Reserved VRAM | Updates/epoch |
|---|---:|---:|---:|
| 64 | 14.30 | 0.240GiB | 9 |
| 128 | 28.70 | 0.441GiB | 4 |
| 256 | 57.21 | 0.873GiB | 2 |

All losses/gradients were finite; selected-batch reserved-memory headroom is
96.36%. Validation16/32/64gave identical39030timestep predictions. Batch16CE
differed by9.97e-6;32and64agree exactly and each form one actual30record batch.
Preserve validation64. The exact calibration source, logs, config/source
identity and completion record are saved beside this report.

Both ECG2arm dry runs passed from the repository and an unrelated working
directory. Fresh matched Baseline/gain-only50epoch training is active under
`transfer_A002_gain_v1/`; seed42, split42, train256, validation64, dropout0.15,
LR0.01, original timestep loss and full400epoch scheduler. The baseline has
recorded completed epochs; gain-only follows. Output is
`a002_ecg_stage050_stdout.log`/`a002_ecg_stage050_stderr.log`; prerequisites and
launch provenance are in `A002_ECG_STAGE050_PREREGISTRATION.json` and
`A002_ECG_STAGE050_LAUNCH.json`. Estimated combined duration is20-25minutes.
Two updates/epoch may affect convergence: assess baseline learning at the
declared screen, never reject on its first weak epochs or change settings midrun.
Record separation is verified; patient independence is not. Official tests stay
closed. This is a50/400epoch development screen, not full-budget evidence.

## A002 ECG50screen complete: late learning, matched extension to100

| Arm | Selected completed epoch | Training accuracy at selection | Clean validation accuracy | Validation CE | Macro class recall |
|---|---:|---:|---:|---:|---:|
| Baseline | 50 | 36.20% | 35.56% | 1.422722 | 25.55% |
| Gain-only A002 | 50 | 64.62% | 64.16% | 1.003091 | 59.95% |

The nominal accuracy difference is+28.59851pp. This shows a large early
optimization/learning-speed difference under this batch/budget, not converged
generalization or a final-test gain. Both models selected the final epoch,
validation loss was still decreasing, and only100optimizer updates completed.
Baseline remained near the31.19% majority-class rate until late in the screen.
A small training/validation gap here does not establish successful generalization.

Read-only clean validation reproduced both logged checkpoint scores and losses.
Full class confusions, supports, original record IDs and predictions are retained
beside each seed in `VALIDATION_DIAGNOSTICS_STAGE050.json` and `diagnostic_rows/`.
Gain improves macro recall but is not uniformly better: C0andC5recall are lower
than baseline; both models have very poor C0recall. C0..C5denote original label
columns, with no inferred clinical names. Timesteps are correlated and patient
independence is not established. These remain seed42validation-only findings.

Checkpoint hashes, all50distinct CSV validation epochs, selected indices and
100optimizer steps were verified. Both effective datamodule splits match the
frozen audit; configs differ only by declared gain/provenance fields. The
unrounded comparison is `A002_ECG_STAGE050_COMPARISON.json` with
[learning curves](A002_ECG_STAGE050_CURVES.png); the exact read-only diagnostic
source is `ECG_STAGE050_DIAGNOSTIC_SOURCE.txt`.

**Decision: EXTEND matched50->100.** Both arms resume their own last checkpoints,
retaining the400epoch scheduler and unchanged settings; all50epoch snapshots are
preserved. This restores optimizer/scheduler state, not proven uninterrupted RNG
equivalence. The100epoch resume preflight passed for both arms and the queue was
launched after confirming no other training process. Decision/launch records are
`A002_ECG_STAGE100_DECISION.json` and `A002_ECG_STAGE100_LAUNCH.json`; live output
is `a002_ecg_stage100_stdout.log`. Estimated additional time is about20minutes.
No automatic escalation to400, no B training/final replication, and no official
test access are authorized by this preliminary result.

## Operational interruption during ECG100screen

The first100epoch launcher and all Python training children disappeared while
baseline progress ended at77completed epochs. No Python exception was present
in its stderr; the termination cause is not established. Repeated native process
checks confirmed no training remained. The own `last.ckpt` was readable at
checkpoint index76/global step154with Adam and scheduler state, and its77CSV
validation epochs were unique and complete. Its observed input hash and recovery
decision are recorded in `A002_ECG_STAGE100_RECOVERY077_DECISION.json`.

The unchanged100epoch resume preflight passed and the same queue was resumed
from each model's own last checkpoint, preserving original logs. Recovery output
is `a002_ecg_stage100_recovery077_stdout.log`; epochs78/79and optimizer steps
156/158confirm baseline continuation. No settings or source were changed.

This is a departure from the predeclared shared-pause condition: baseline has
an additional unplanned77epoch interruption. Core split/batch/optimization/budget
settings remain matched, but restart histories do not. Treat this screen as
exploratory and report the operational deviation; require fresh frozen
confirmation before final architectural/generalization claims. A moving
`last.ckpt` advances during continuation; the observed77epoch input hash records
lineage, while immutable stage050checkpoints/receipts remain preserved.

The recovered baseline has now completed100epochs. Selected completed epoch96:
training79.112524%, clean validation77.612096%, CE0.644082; final validation is
77.350754%. Verification found exactly100unique validation epochs with no
duplicated/missing history, correct192selected/200final optimizer steps, valid
best/last hashes and unchanged stage050snapshots. See its
`stages/100/VERIFICATION.json`. This remains100/400epoch exploratory evidence,
including the recorded extra restart. Gain-only has resumed its own50epoch
state toward100; wait for that matching-budget score before comparison.

## A002 ECG100comparison complete; transfer stage closed

| Arm | Selected completed epoch | Training accuracy at selection | Clean validation accuracy | Validation CE | Macro class recall |
|---|---:|---:|---:|---:|---:|
| Baseline | 96 | 79.11% | 77.61% | 0.644082 | 73.64% |
| Gain-only A002 | 99 | 81.71% | 78.69% | 0.598180 | 74.67% |

Gain minus baseline is+1.08122pp. Read-only validation reproduced both checkpoint
scores/losses, with full class confusions and original-row predictions saved.
Both100epoch histories and best/last hashes verify;50epoch snapshots are intact.
This is100/400epoch, single-seed exploratory evidence with unequal restart
histories from the baseline77epoch interruption, not final generalization or
uninterrupted equivalence. The28.60pp early difference at50shrunk substantially
as baseline learned, illustrating why optimization adequacy matters.

See `A002_ECG_STAGE100_COMPARISON.json`,
[100epoch curves](A002_ECG_STAGE100_CURVES.png), and
`A002_TRANSFER_STAGE_COMPLETION.json`. A002has positive development-speaker
pair results, weak SSC40evidence (+0.10pp, below practical threshold), and a
preliminary ECG100advantage. No two-dataset final-test improvement is established.
All236files in the existing A002source archive still match current source;
reuse that verified archive without duplicating it. No training is active.

**Next bounded idea:** B002training-only label smoothing, epsilon0.1, targeting
the remaining primary-speaker confidence/CE concern. Preregistered in
`B002_SCREEN_PREREGISTRATION.json`: fresh matched Baseline/A/A+B on0/3, seed42,
30epochs, clean A+B-Aaccuracy threshold+0.5pp, ordinary validation CE, no test
access. It is selected for implementation/checks, not implemented or trained yet.
Existing DTH was reviewed as an alternative from source; no empirical ranking
or new DTH experiment is claimed. Final four-arm/three-dataset replication and
ordinary official tests remain pending.


## B002 primary speaker development: three-arm30screen

Baseline81.834865%, A82.324159%, A+B86.727828%; A+B minus A+4.403669pp. Ordinary validation CE at selection:0.922076/0.815030/0.603719. Late10epoch accuracy means:80.599390/79.571866/85.461774%. A alone is+0.489295pp, below+0.5threshold. B-only control is required before claiming gain contributes. One development pair and seed, no official tests or canonical paper comparison. See `B002_PRIMARY_STAGE030_COMPARISON.json` and `.md`; all checkpoint/config/CSV checks passed.


## B002 four-arm control revises the combination claim

B-only LS0.1 reached86.544341% clean primary validation versus gain+LS86.727828%, baseline81.834865% and gain82.324159%. B-only minus baseline+4.709476pp; AB minus B+0.183487pp; interaction-0.305808pp. Added gain fails the+0.5pp conditional threshold; no two-idea complementarity established. LS-only has comparable ordinary CE and late accuracy to the combination. Shortlist only LS-only for predefined6/10sensitivity with fresh baseline; retain all four original records and do not search pairs or tune epsilon. No official tests/canonical ablation yet. Authoritative evidence: `B002_PRIMARY_STAGE030_FOUR_ARM_COMPARISON.json` and `.md`.


## B002 fixed6/10sensitivity complete

Baseline84.304208%, LS-only89.482200%; delta+5.177993pp, validation only. Supports cross-dataset validation screening of LS-only after disposable current-objective calibration. No gain complementarity, independent seed replication or official-test improvement established. See `B002_LS_SENSITIVITY_STAGE030_COMPARISON.json`/`.md`.


## B002 SSC matched validation screen launched

Current-source Baseline and LS-only disposable calibration passed at train/val1024:1044.98/1037.54samples/sec,39.82/39.91%reserved-memory headroom, finite losses/gradients and identical1024validation-probe predictions for256vs1024batch. States discarded; old full grids preserved. Fresh seed42Baseline/LS-only10epoch SSC screen now launched under full40schedule, eight workers/LR0.006 unchanged, official tests disabled. See `B002_SSC_STAGE010_PREREGISTRATION.json` and `B002_LS_SSC_STAGE010_LAUNCH.json`. No SSC outcome yet.


## B002 SSC10inconclusive: matched full-state extension

Baseline78.429013%, LS78.809738%, delta+0.380725pp; ordinary selected validation CE0.703566/0.797377, LS worse. Both select epoch10 and still improve over8->10(accuracy+1.442742/+1.122129pp, CE-0.056133/-0.039753). No positive transfer claim. Verified checkpoint hashes/configs/10uniqueepochs, unchangedsource; both own last states now fully resume to original40budget,10snapshots retained, no tuning or official tests. See `B002_SSC_STAGE010_COMPARISON.json` and `B002_SSC_STAGE040_DECISION.json`.


## B002 primary CPU diagnosis

LS gains on both primary speakers on CPU(+4.27/+5.12pp), corrects133baseline errors but loses57cases;171remain wrong under both. CE/Brier/wrong-confidence improve but ECE worsens due underconfidence. CPU net score counts differ from retainedGPUcounts(-7baseline/-8LS); scores are not replaced. GPUreplay is preregistered after SSC closes. See `B002_PRIMARY_DIAGNOSTIC_INTERPRETATION.md` and raw hash-bound arrays. No official tests or new tuning.


## B002 SSC40completed: no declared practical gain

Baseline81.524897%, LS81.955713%, delta+0.430816pp below+0.5threshold. Ordinary selected CE0.625711/0.681354, LS worse. Full matched40budget complete,40unique epochs per arm and old10snapshots verified. No SSC extension or official-test claim. See `B002_SSC_STAGE040_COMPARISON.json`/`.md`. Next: primary GPUreplay after queue terminal, then matched ECG calibration/50screen; full goal remains incomplete.


## B002 ECG50fresh matched screen launched

Current-source Baseline/LS-only calibration passed:57.82/57.72samples/sec,96.44%memory headroom, full1301steps, finite gradients, identical validation predictions16vs64. States discarded. Fresh seed42Baseline/LS-only50screen under original400schedule, train256/val64/LR0.01/dropout0.15, fixed588/30rows. SSC40negative/below-threshold outcome preserved. No patient independence, test results or two-idea claim. See `B002_ECG_STAGE050_PREREGISTRATION.json`/`B002_LS_ECG_STAGE050_LAUNCH.json`.


## B002 ECG50 review and matched100 extension

Both arms completed50: baseline35.5649501%, LS35.3753537% clean validation; LS-minus-baseline -0.1895964pp, not a gain. Ordinary CE1.42272/1.43497; macrorecall25.55%/24.88%. Both best at50 and still improving with online training accuracy~36%; matched controls remain underlearned after100updates. Original1.0pp threshold retained. Predefined full-state resume50->100 launched after successful dry-run and both own last checkpoint SHA/epoch49/globalstep100/optimizer/scheduler checks. Unchanged400schedule and shared settings; preserved50snapshots. Official tests excluded. See B002_ECG_STAGE100_DECISION.json and launch receipt; no automatic400epoch advancement.


## B002 ECG100 closed: no declared transfer gain

Both arms reached100/400epochs through own full-state resume. Baseline selected77.476299% at completed99, LS76.938254% at100: **LS-minus-baseline -0.538045pp**, below original+1.0pp criterion. Ordinary validation CE0.639904/0.682718 and macroclassrecall73.5813%/72.5953% also favor baseline. Online train at selected epochs79.8199%/79.3812%; these are not clean checkpoint train reevaluations. All100uniqueepochs, exact best/last hashes/configs, lastglobalstep200, optimizer/scheduler, prior50snapshots and selected validation inference verified. Training queue terminal; no official tests. See B002_ECG_STAGE100_COMPARISON.json/.md and curves.

**Decision: revise cross-dataset hypothesis.** LS remains promising SHD speaker-development evidence (+4.70948pp primary0/3; +5.17799pp fixed6/10), but SSC40+0.43082pp misses its+0.5pp screen and ECG100regresses. Gain+LS is not established as complementary. Do not extend this branch automatically to400or replicate seeds. Preserve all outcomes and study a distinct justified mechanism before final architecture freeze. Full goal is incomplete: canonical four-arm/full-budget/three-seed/test/cost evidence remains pending. See B002_TRANSFER_STAGE_COMPLETION.json.


## C001 distinct internal-threshold hypothesis: primary30 launched

B002closed with promising SHD but below-thresholdSSC/negativeECG; do not claim two-dataset improvement. New fresh4arm Baseline/DTH/DTH+LS/LS primary0/3seed42 tests internal activity-dependent thresholds instead of input gain, with fixed0.1smoothing. Established components; no novelty/synergy assumed. All4share1024batch/LR0.01/dropout0.15/full300schedule,30ceiling. Require combination to improve over each component(+0.5pp practical gate), supported by ordinary validation CE/curves; gap shrinkage insufficient. Historical tested results are disclosed reused evidence, not model-selection endpoints.

34nontrainingchecks passed including CPU/CUDA gradients and exact eval/state/RNG parity for DTH+LS; runner dryruns passed from root and externalTEMP. Real342step calibration passed all4,2warmup+3measuredsteps; DTH/DTH+LS~664/628samples/sec and~49.6%memoryheadroom; all1635validationrecords gave0changedpredictions256vs1024. Disposable seed117states discarded.250files/sourcearchive frozen before launch. New main runs start seed42fresh. See C001_DTH_LS_PREREGISTRATION.json, C001_IMPLEMENTATION_VERIFICATION.json and launch receipt. No official tests or further stage automatically authorized by a positive early curve.


## C001 primary30 closed: conditional combination gain, DTH-alone failure

Fresh seed42matched4arm primary0/3: Baseline83.302754%, DTH80.978596%, DTH+LS86.911315%, LS85.382265%. Online train at selected epochs99.43/98.96/97.95/99.45%; ordinary validation CE0.9106/1.0900/0.5627/0.6307; selected completed epochs30/28/22/27. Combination-LS+1.529050pp; DTH-baseline-2.324158pp; nominal interaction+3.853208pp does not prove synergy. Combination-LS speaker0-0.76670pp, speaker3+4.43213pp; pooled gain is uneven. Late10mean combination85.54%vsLS84.32%. Both static thresholds and nonzero activity gains learned; this does not isolate dynamic causality.

All4?30uniqueepochs, selected/best/last SHA/configs, globalstep180, source250hashes and clean GPUvalidation inference verified. Full prediction arrays/per-speaker evidence/curves retained. Original A=DTH/B=LS labels remain; DTH-alone did not pass. Baseline->LS->LS+DTH is only a revised conditional progression. No75extension or final claim.

Predefined6/10shortlist-only sensitivity launched fresh Baseline/LS/DTH+LS, seed42, same30budget/sharedsettings, no primaryweights/epsilon/threshold/pair tuning. Longest342steps within current4arm calibration; fresh6920/1236rows verified; root/externalTEMPdryruns and negative shortlist/seed guards passed. Only added study definition; parent250sourcefiles unchanged, new251file snapshot frozen. Estimate15-22minutes. See C001_PRIMARY_STAGE030_DECISION.json, C001_SENSITIVITY_STAGE030_PREREGISTRATION.json and launch receipt. Official tests remain outside development; final frozen comparison incomplete.


## C001 fixed6/10sensitivity30 complete: modest below-threshold increment

Fresh seed42 matched Baseline85.679615%, LS87.864077%, DTH+LS88.268608%; selected completed epochs26/25/25. Online train at selected epochs98.76/99.35/99.22% (not clean train reevaluations); ordinary validation CE0.612529/0.587949/0.550309. Combination-LS **+0.404531pp**, below unchanged+0.5pp practical criterion; loss improves-0.037640 and late10mean improves+0.291261pp. Speaker6increment+0.223714pp, speaker10+0.506971pp.26fixes/21losses=5net correct of1236,124wrong underboth. No significance/population claim or pair-tuning/extension.

All3?30uniqueepochs, own selected/last hashes/configs/globalstep180/optimizer/scheduler, source251files and exact clean GPUvalidation reproduction verified. Primary+1.52905pp overLS was uneven and standaloneDTH failed; both pairs remain one seed of development, not two independent training seeds. Keep all gains/failures. Stop this pair at30; no SSC/ECG or final freeze launched. Next is a primary-only static-versus-dynamic threshold mechanism review/control if nonduplicative, because learned static thresholds shifted along with nonzero dynamic gains. See C001_SENSITIVITY_STAGE030_COMPARISON.json/.md, decision and C001_DISCOVERY_STAGE_COMPLETION.json. No local training remains; full frozen-four-arm/test/cost goal incomplete.


## C002 primary-only bounded static-threshold mechanism control launched

C001combo-LSprimary+1.52905pp butfixed6/10+0.40453ppmisses+0.5criterion; no pair extension/tuning or heavytransfer. Existing train_thr is not an exact matched boundedper-neuronDTHcontrol; LT-DTHlearnstracedecays, FPchangesfrequency. C002compares fresh LS/static-LS/dynamicDTH-LS, primary0/3seed42,1024batch/LR0.01/dropout0.15/full300schedule,30ceiling. Static cell subclassesDTH, freezes both gainparameterszero and rejects nonzero-gain checkpoints; static logits remaintrainable. Same scan/states retained, no inference-speedclaim. This is an established-component mechanism ablation, not novelty or final2idea evidence.

39nontrainingchecks passed (CPU/CUDA forward/common-gradient/RNG/state parity, frozenzero/statictrace independence, unsafe load rejection, config/default regressions); root/externalTEMPdryruns passed. Real342step disposable3arm calibration: LS~895samples/sec,static~668,dynamic~641; memoryheadroom65.5/57.4/49.6%. Frozenstaticgains remainzero/no-gradient throughall5realupdates; all1635validationpredictions identical256vs1024. Disposable117states discarded.256filesfrozen beforefresh42queue launch; estimate18-25minutes. See C002_STATIC_THRESHOLD_PREREGISTRATION.json, C002_IMPLEMENTATION_VERIFICATION.json and launchreceipt. No official tests, no prior checkpoints used, and noautomatic nextstage.


## C002 primary30 mechanism control closed: static hypothesis favored

Fresh seed42 LS86.727828%, boundedstatic+LS87.767583%, dynamicDTH+LS87.033641%; selected completedepochs29/25/25. Online train99.50/99.30/99.28%(not clean trainreevaluations); ordinaryvalidationCE0.579465/0.567237/0.614121. Static-LS **+1.039755pp**, dynamic-static **-0.733942pp**, dynamic-LS+0.305814pp misses+0.5criterion. Static gain is+0.87623pp on speaker0 and+1.24654pp on3; late10meanstatic86.7401%vsLS85.2355%. Staticfixes43/losses26versusLS; dynamicfixes31/losses43versusstatic. All3?30uniqueepochs, best/last SHA/configs/globalstep180/optimizer/scheduler, source256files, exactcleanGPUvalidation andfrozenzerostaticgains verified. No dynamic-specific benefit supported in this screen; no novelty/synergy/cross-dataset/testclaim.

Same-seed LScontrol differed1.34556pp from C001despite matching effective settings; repeats are not independent seeds and actual initial-state hashes were not saved. Bounded syntheticreduction diagnostic after training: CPUrepeatsexact; ordinaryCUDAreducedscoremaxdiff2.67029e-5,inputgradientmaxdiff1.39698e-9,0predictionchanges; diagnosticdeterministicCUDArepeatsexact. This is a possible numerical contributor,notproven cause oftraining-scorevariation. No trainingprecision/determinism changes made. See C002_REPEATED_SEED_CONTROL_OBSERVATION.json and C002_REDUCTION_REPEATABILITY.json.

**Next:** fresh matched staticfullfactorial primary screen (Baseline/static-only/LS/static+LS) before transfer/freeze; preserve allnegativeDTH andLStransfer findings. Consider initial-statehash metadata without RNG/model changes. C001secondarypair remainsclosed; no tuning/extension. No local training remains, officialtestsuntouched, finalfullbudgetthree-seedfour-arm/test/costgoal incomplete. See C002_PRIMARY_STAGE030_DECISION.json and C002_MECHANISM_STAGE_COMPLETION.json.


## C003 fresh staticfullfactorial primary30 launched

C002favored boundedstatic+LSoverLS(+1.03976pp)anddynamic(-0.73394pprelative tostatic), but standalone static effect/factorial interaction remainedunknown. Fresh C003Baseline/static-only/static+LS/LS tests thosecontrasts, samefrozen0/3manifest/trainingseed42,1024batch/LR0.01/dropout0.15/full300schedule,initial30ceiling. Original0.5pppractical gates andordinaryCE/curves retained; noforcedsynergy, historicalcohortmix orsecondarytuning. No precision/determinism settings changed; syntheticnumericaldrift doesnotprovetheactualrepeatspreadcause.

Newopt-in INITIAL_MODEL_STATE.json recordsfreshpre-update per-tensorvalue/dtype/shapehashes, trainability, CPU/current-GPURNGhashes andexecutionflags; skipsfullresume andrejectsconflictingoverwrite.45nontrainingchecks passed includingCPUCUDAno-RNG/weight/flagmutation, scalar/bfloatbuffers, callbackgate,full-size4architecturecommoninitialtensors. Root/externalTEMPdryruns passed. Disposable4arm real342step calibrationpassedfinitegradients/frozenzerogains,57-65%memoryheadroom,0validationpredictionchanges256vs1024.117states discarded.261filesfrozen beforefresh42queue, estimate20-30minutes. See C003_STATIC_FACTORIAL_PREREGISTRATION.json, implementationverification andlaunchreceipt. Noofficialtests/finalcomparisonclaim.


## C003 staticfullfactorial30 closed: reject static branch under this screen

Freshseed42 matched Baseline81.651378%, static-only78.899086%, static+LS86.605507%, LS87.400609%; selectedcompletedepochs27/22/22/27. Online train97.92/98.62/98.62/98.84%(notcleancheckpointtrainreevaluation); ordinaryvalidationCE0.9093/0.9471/0.5814/0.5600. Static-baseline **-2.752292pp**; combination-LS **-0.795102pp**. Combination hasworseCE/late10meanthanLS andlosesgroundonbothspeakers. Positiveinteraction+1.95719pp onlyreflectssmoothingrescuingweakerstaticarm; it doesnotestablishsynergy orcombinedsuperiority. C002positiveincrementdidnot holdinthisfreshcohort; no cherrypicking.

All4?30uniqueepochs,ownbest/lastSHA/configs/globalstep180/optimizer/scheduler, source261files, frozenzerostaticgains,actualinitialcommonweights/CPU-CUDARNG/executionflagmatches andcleanGPUvalidationreproduction verified. InitialfullstatehashesmatchSEvsLSandstatic-onlyvsstatic+LS. Rawpredictions/per-speaker/curves retained. No75extension,secondarytuning,SSC/ECGtransfer orofficialtests. This is a30screenrejection,notuniversalproofagainststaticthresholds.

**Next:** distinctinput-conditionedgeneralizationmechanism afterduplicate/prior-artreview, notanotherstaticthreshold/gain tweak. LSremainsstrong SHDdevelopmentcomponent, but SSC0.43ppbelowthreshold/ECG100-0.54pp remainnegative transfer evidence. Two complementaryideas andfinalfrozenfour-arm/fullbudget/three-seed/test/cost evidence remainunestablished. See C003_PRIMARY_STAGE030_DECISION.json and C003_FACTORIAL_STAGE_COMPLETION.json.


## C004 distinctper-framecurrentnormalization initial4armprimary30launched

AfterrejectedC003staticbranch, testinput-conditionedfeedforwardcurrent: c=linear(x,W,b),d=c-b,c_new=c+tanh(g)*nonaffineLayerNorm(d,eps1e-5,lastfeatureaxis). Onezeroinitgate/layer; rawcurrent/SEstates/recurrence/resetretained, nobatch/timepooling,zero-inputcurrentbiaspreserved,no mask/timewarp/truncation. LayerNorm/ReZero/SNNnormalizationarepriorart;no novelty/theoreticalguaranteeclaim. Speaker-dependentcurrentvariationishypothesis,notproven cause.

Freshseed42 Baseline/RCN/RCN+LS/LS same0/3manifest,1024batch/LR0.01/dropout0.15/full300schedule/30ceiling,initialhashesenabled,no numericalsettingchange. CG/CG+LSparameter-matchedcurrentgaincontrolsregisteredupfrontandcalibratedbutnotlaunched. Theirbounded0..2gainrangeisnotallnormalizationeffectivescales;learnedscalediagnostics/separatelyfrozencontrolmaybeneededbeforemechanismattribution. Noforcedsynergy orautomatictransfer.

53nontrainingchecks passed:CPUCUDAzero-gateforward/scan/stateidentity,manualLN/gradientoracle,causality/batch/quietness,gainanalyticcontrol,finiteimmediategategradient,originalsmoothingmasks/ordinaryevalCE,6full-sizecommoninitialtensorhashes/configguards. Root6modelandexternalTEMP4modeldryrunspassed. All6disposable real342stepcalibrationspassedfinitegradients/quietbiaspreservation/0changedvalidationpredictions256vs1024;norm~778/786samples/sec and57.5%memoryheadroom.117statesdiscarded.269filesfrozenbefore4armqueue;estimate20-30minutes. See C004_CURRENT_NORMALIZATION_PREREGISTRATION.json, implementationverification andlaunchreceipt. Officialtestsremainoutside development.


## C004 current-normalization four-arm30 closed: reject this branch

Fresh matched seed42, primary0/3 clean validation: Baseline **83.119267%**, A **82.996941%**, A+B **86.911315%**, B-only smoothing **88.012230%**. Selected completed epochs22/16/16/30; online train98.24/96.31/96.83/99.74% (dropout/changing weights, not clean checkpoint training evaluation). Ordinary validation CE0.76285/0.72023/0.54401/0.53101. A-baseline **-0.122327pp**, AB-A **+3.914374pp**, B-baseline **+4.892963pp**, AB-B **-1.100916pp**, interaction **-0.978589pp**. Combination late10mean is2.452599pp below B. Per-speaker AB-B:0+0.109529pp,3-2.631579pp. No test/general-population/synergy claim.

All4 fresh actual common initial tensors and RNG/settings, exact30 unique epoch histories, own best/last identities/configs/optimizer/scheduler/180finalupdates, source269files and selected-checkpoint ordinary GPU validation were verified. Curves inspected. A/AB best at16 with no subsequent accuracy improvement; B continues learning to30. Do not extend failed normalization merely because the stronger B control is still improving. No75extension, gain-control training, sensitivity check, SSC/ECG transfer or official tests. Retain registered unrun controls and unapplied isolated cost-accounting draft as such. See C004_PRIMARY_STAGE030_DECISION.json and C004_FACTORIAL_STAGE_COMPLETION.json. Two complementary mechanisms and final canonical/fullbudget/three-seed/test/cost evidence remain unestablished.


## C005 dropout consistency initialfourarm primary30 launched

Distinct train-only objective: two independent original dropout forwards, average supervised CE plus alpha1 symmetricKL on original valid target blocks. B remains epsilon0.1 smoothing. Ordinary single-pass inference/SE dynamics/state keys unchanged. R-Drop is established prior art, no noveltyclaim. Fitting-only160row diagnostic showed3 baseline/1 smoothing disagreements, modest susceptibility not causal evidence. Alpha1 fixed before that diagnostic.

58 nontraining checks passed, including CPU/CUDA KL value/both-gradient oracle, masking/int32 targets, zero-alpha two-pass oracle, no extra state/RNG/eval behavior, all6full-size matchinginitialstates and guards/defaultregressions. All6root andinitial4externalTEMPdryruns passed. Full342frame real512/1024calibrations passedfinitegradients/15%headroom and0changed256vs1024ordinaryvalidationpredictions;1024selected~2xfasterpersample,32.5%two-graphreservedheadroom.117statesdiscarded.277sourcefilesfrozen/archiveverified. Fresh42 initial4 Baseline/RDrop/RDrop+LS/LSqueue launched viaofficialrunner,1024batch/LR.01/dropout.15/30ceiling/full300schedule. Estimated30-45minutes; updates6/epoch,extra2passcostexplicit. Alpha0two-passcontrolsregistered/calibratedbutnotlaunched; requiredbeforeattributingapositivegain specificallytoKL. Officialtestsdisabled. SeeC005preregistration/calibration/implementation/snapshot/launchreceipts.


## C005 initialfactorial30 verified: combined increment rejected; standalone needs two-pass control

Freshseed42 clean primary0/3 validation: Baseline82.140672%, RDrop83.853209%, RDrop+LS86.299694%, LS88.440365%. A-base+1.712537pp; B-base+6.299692pp; AB-A+2.446485pp; AB-B-2.140671pp; interaction-3.853208pp. A selected ordinaryCE0.944279 vs baseline0.946418 (nearlyunchanged), late10mean+0.850151pp. A improves speaker0 by6.790800pp but worsens speaker3 by4.709141pp: not uniform speaker robustness. Combination underperforms LS in accuracy/CE/late10 andbothspeakers. All4checkpointpredictions/ordinaryCE/configs/source277/actualinitialstate-RNGflags/30epochs/best-last/optimizer/scheduler/180updatesverified; RDropKLnonzero/finite andloggedtotaldecompositionverified. Curves inspected.

Reject combinedincrement underthisscreen; no75/secondary/transfer/test. StandaloneA merits only the alreadyregistered alpha0two-passsupervisedcontrol toseparateKLfromstochasticgradientaveraging. NoKLspecificbenefitormulti-datasetgainclaim. SeeC005_PRIMARY_STAGE030_DECISION.json andC005_A_ONLY_CONTROL_PREREGISTRATION.json.


## C005 alpha0 control verified: shortlist A alone for predefined6/10

Alpha0two-passCE81.039755% versus RDrop83.853211% (exactreproducedproportions): **+2.813456pp**; control-baseline **-1.100917pp**. RDrop-controlordinaryCE **-0.106498**, late10mean **+1.602446pp**, speaker0 **+4.709748pp**, speaker3 **+0.415512pp**;89fixes/43losses. Allsource277/initialfullweights-RNG/configs/30epochs/best-last/optimizer-scheduler/180updates/alpha0loggedlossdecomposition andordinaryselectedcheckpointpredictionsverified. No standalonepopulation/testclaim; RDropstillharms3vsordinarybaseline. Combinedincrementremainsnegative.

Primaryclosed; shortlistAonly(alpha1,epsilon0) for freshownsplit Baseline/A30 onpredefined6/10. No primaryweightsreuse,secondarytuning,transfer/finaltestyet. See C005_A_ONLY_CONTROL_STAGE030_DECISION.json, C005_PRIMARY_DISCOVERY_STAGE_COMPLETION.json and C005_SENSITIVITY_STAGE030_PREREGISTRATION.json.


## C005 predefined6/10 sensitivity launched: baseline and shortlistedA only

Fresh42, fixed6920/1236officialtraining-onlyrows, fit0/1/2/3/7/8/9/11, held6/10;all20classesbothsides,nooverlap/exhaustiveindices,actualproductionloaderand128longest342frame/140featureboundverified. Existing32.5%headroom1024two-graphcalibration remainsapplicable;no weightsreused.13nontrainingtests andexternalTEMP2armdryrun passed.278sourcefiles/newYAMLonlyplusprimaryclosuredescriptionsfrozen;shortlist/split/preregistrationbytesarchivedandhashedbeforetraining. SameLR.01/dropout.15/full300schedule/30onlyceiling;no secondarytuning/extension. OfficialrunnerPID21992launched;estimate13-20minutes. No failedB/combination oralpha0sensitivitysweep. Allprimaryevidenceretained.


## C005 predefined6/10 sensitivity verified: Aonly advances to SSC calibration

Fresh42 Baseline83.818770%, RDrop85.517799% (reproducedproportions); selectedcompletedepochs27/21; online selectedepochtrain98.99/98.29% (dropout/firstview,notcleancheckpointtrain). OrdinaryCE0.654692/0.646366. LoggedA-baseline+1.699030pp (exactcount+1.699029pp);late10mean+1.116505pp;6-1.342282pp,10+3.422053pp.68fixes/47losses. Source278/protocolartifact/actualfreshinitialfullweights-RNG/configs/30epochs/best-last/optimizer-scheduler/180updates/realRDropcomponentlogs andordinaryselectedcheckpointvalidationverified. Curves inspected. No secondarytuning/extension.

QualifiesAonlyforboundedSSC10/40validationfollowingnewtwo-graphmemory/throughputcalibration. Fixedalpha1/epsilon0;newmatchedsharedbatch/LR0.006andmeanmembraneloss;canonicalSSCcontrolpreserved. B/combinedfailed;twocomplementaryideasandactualofficialtestimprovementsremainunestablished.


## C005 SSC initial10 launched after new two-graph calibration

Officialtrain75466/valid9981,35classes,disjointstoredspeakerIDs andsourceSHAverified.16nontrainingchecks passedincludingownSSCbaseline/720x2/35class/meanloss/RDropinitialstate andspawntransformsemantics. Real256/512one/two-graph300framecalibrationpassedfinitegradients;512selected(base~536samples/s,RDrop~264.5),RDrop41.15%reservedheadroom. Training1024skippedbymeasuredfootprintprojection. Full9981ordinaryvalidation256/512/1024retained0changed/repeatedpredictionsandCEdifference<1e-5;1024valselected. Sameold8workerproducer2388.9samples/s source/semanticsverifiedandreuseexplicit,bound>bothGPUconsumers. All117calstatesdiscarded.

Fresh42 Baseline/RDropSSCqueueviaofficialrunner startedPID20700,512train/1024val/LR0.006unscaled/dropout.15/full40schedule/10ceiling/147updatesperepoch.280sourcefilesand11protocolartifactsfrozen/archivesverified;root/externalTEMPdryrunspassed. Actualbaselinefreshinitialstateverifiedbeforefirstepochcompletion;noSHD/calibrationweightsorofficialtests. Estimate75-100minutes;40extensionrequiresreview. ExistingcanonicalSSCcontrolpreservedinitsownpath;newcontrolismatchedtothisseparatebatch/objectiveprotocol.
