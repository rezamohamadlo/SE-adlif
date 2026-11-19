# Active research plan: unseen-speaker generalization

This confirmation governs subsequent experiment decisions. It supersedes random-split-only idea screening and earlier exploratory official-test permission. The frozen source guides and historical records remain intact.

## Revised stage order

1. Audit current jobs and preserve their actual protocol, checkpoints, resolved configs, metrics and source evidence. Finish only the already bounded appropriate stage; completion does not authorize automatic full-budget expansion.
2. Freeze and verify official-training-only speaker manifests. Primary validation is speakers **0/3**: fit speakers **1,2,6,7,8,9,10,11**, **6,521/1,635** recordings. Predefined sensitivity validation is **6/10**: fit **0,1,2,3,7,8,9,11**, **6,920/1,236** recordings. Both splits exhaust the official training rows, have no sample/speaker overlap, and cover all 20 classes on both sides. Never regenerate rows or search for a favorable pair.
3. Screen fresh **Baseline, A, A+B**, with **B-only** to interpret complementarity, initially at training seed **42**. Match exact split, preprocessing, batch, optimizer, learning-rate schedule, dropout, checkpoint rule and screening budget; explicitly record intended mechanism/objective changes. Select on improved **clean held-out-speaker accuracy**, supported by ordinary hard-label validation loss and learning curves. A smaller train-validation gap alone is not success. Preserve negative results and previously declared thresholds.
4. Check only shortlisted candidates and their matched controls on the predefined **6/10** holdout, starting fresh on its own fitting rows. Pair 0/3 is repeatedly tuned development; 6/10 is a predefined sensitivity diagnostic with accumulated exposure, not an independent untouched confirmation. Do not tune on this pair or search other pairs.
5. Check promising ideas on **SSC and ECG**, using their verified dataset-specific validation protocols and small predefined budgets. Do not fully train every idea under both SHD protocols. Record underlearning and any justified budget extension before interpreting it; do not assume validation gains transfer.
6. Freeze mechanisms, settings and the four-arm **Baseline/A/B/A+B** comparison. Train fresh under **canonical SHD*** (20% random validation from official training, fixed split seed42), using the paper-derived baseline and matching training seeds. Train canonical weights on their own split; speaker-holdout weights cannot stand in for canonical weights. Replicate the frozen comparison and retain separate speaker-development and canonical outputs/tables.
7. Only after validation-based selection and freezing, evaluate official tests with ordinary inference, report matched-seed variation and costs, and state prior test exposure. Official tests stay outside the discovery loop; no TTA-based selection.

## Current bounded stage

The live C004 queue is **speaker-disjoint development**, not canonical SHD* or official testing. It compares fresh Baseline, residual-current-normalization A, A+B with label smoothing, and smoothing-only B on primary0/3, seed42, batch1024, LR0.01, and the same initial30-epoch ceiling under the full300-epoch schedule. Testing and test-data construction are disabled. Preserve the queue and its frozen source; review results before any extension. Registered gain-only mechanism controls are conditional, not an automatic additional queue.

## Checkpoint and control eligibility

Never assess a speaker as unseen with weights previously trained on that speaker. This excludes canonical random-split weights from clean held-out-speaker development and primary-pair weights from secondary-pair validation. Historical controls are reusable only when exact rows, exposure, relevant source behavior, architecture, preprocessing, initialization/resume mode, optimization, budget and selection all match. Repeated seed42 runs are not independent seeds. No result from one protocol is relabeled as another.

Verification: [ACTIVE_RESEARCH_PLAN_VERIFICATION.json](ACTIVE_RESEARCH_PLAN_VERIFICATION.json). Frozen source guide: [speaker-development study](../../../docs/experiments/complementary-mechanisms-study-v2.md). Current mechanism: [C004 preregistered guide](../../../docs/experiments/complementary-C004-current-normalization-v1.md).

## Latest stage closure

C004 primary30 is now complete and rejected under its declared screen: A-baseline -0.122327pp; AB-B -1.100916pp, with worse combined CE and late10mean. Launcher/training children were terminal before checkpoint inference. Preserve evidence; no extension or transfer of this branch. The current-stage paragraph above records the preceding live-stage observation, not an active queue. See [decision](C004_PRIMARY_STAGE030_DECISION.json).

## C005 progression

Primary fourarm screen and registeredalpha0control are complete. OnlyRDropA(alpha1,epsilon0) is shortlisted: +1.71ppversusbaseline,+2.81ppversusalpha0control,withspeakerheterogeneity. Smoothingcombinationfailed. Fresh Baseline/Aonly30epochpredefined6/10sensitivity isnowqueued/running; actualstatus/provenanceareinEXPERIMENT_REGISTRY.jsonandper-runSTATUS.json. No primaryweightsreuse,secondarytuning,transferorofficialtestaccess.

## C005 SSC bounded transfer

Fixed6/10screenverified+1.699ppweightedA-baselinewithlowerCEandpositive late10mean,stillheterogeneousbyspeaker. OnlyAadvances. FreshmatchedSSCofficial-validation Baseline/RDrop10queueisnowrunning,512train/1024val/LR0.006/meanmembraneloss/full40schedule. Newrealcalibrationanddata/sourcecheckscompletedbeforelaunch;nomodelcodechange. Monitoractualregistry/per-runstatus;no40extension,ECG/newB/canonicaltrainingorofficialtestaccessbeforestage review.
