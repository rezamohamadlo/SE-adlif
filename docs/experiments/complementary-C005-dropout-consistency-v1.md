# C005: dropout consistency with confidence regularization

C004 current normalization is rejected under its matched primary30 screen.
This distinct training-objective hypothesis uses ordinary SE-adLIF inference.
The study request authorizes execution through the existing official runner;
no separate launcher is introduced. Preserve earlier negative results.

For each unchanged batch, make two independent dropout-enabled forwards.
Reduce both with the original dataset-specific target-aligned temporal
aggregation. Train on the average supervised losses plus **alpha=1** times
**half the sum of both KL directions**, summed across classes and averaged
over valid target blocks. Both distributions receive gradients. No temperature
scaling, detached teacher, time-window comparison, augmentation or mask change
is used. Each batch still produces one optimizer update with the same Adam,
LR, gradient clipping and schedule. The two graphs increase training cost.

This is established [R-Drop](https://arxiv.org/abs/2106.14448), not a new neuron.
Existing MT temporal consistency compares time windows within one forward,
whereas this objective compares independent dropout passes of the same sample.
No prior R-Drop implementation was found in the targeted repository audit.
The fitting-only diagnostic used one recording per class and fitting speaker,
160 total: baseline dropout predictions disagreed on3, smoothing on1; mean
symmetric KL was about0.024. This is modest susceptibility, not proof that
dropout causes speaker shift. Alpha1 was fixed before those diagnostic results.

## Frozen screen

Fresh Baseline/A/A+B/B-only, training seed42, frozen primary0/3 manifest,
ordinary clean validation. B remains train-only uniform smoothing epsilon0.1.
Both mechanisms change training only. Evaluation uses one original forward
and ordinary hard-label CE, with identical parameter/state keys. All common
initial tensor and RNG hashes are recorded. No previous or calibration weights
initialize study runs. Keep the full300epoch scheduler with initial30ceiling;
75 only if explicitly justified by genuinely promising/inconclusive curves.

Batch1024 is proposed subject to disposable real long-sequence two-graph
calibration,15%reserved-memory headroom and numerical validation-batch parity.
LR0.01 is not scaled. Any smaller accepted batch is frozen for the entire
fresh comparison before launch. Dropout0.15, precision32-true/high, clipping1.5,
preprocessing, split, update counts and selection rules remain matched.

Advance on clean held-out accuracy, with predeclared practical contrasts
A-baseline, AB-A and AB-B of+0.5pp, supported by ordinary CE and curves.
Report every contrast and interaction; no forced synergy or gap-only success.
First-view online accuracy is logged once; train loss is average supervised
loss plus weighted KL. Log ordinary hard CE, supervised loss and raw KL
separately. Training accuracy is not clean checkpoint reevaluation.

Two-forward alpha0 controls, with/without smoothing, are registered before
candidate results. Run the relevant control only for a positive R-Drop arm,
before attributing benefit specifically to KL rather than averaging supervised
gradients from two dropout samples. Both controls retain the two graphs and
the same optimizer update count. A positive standard-baseline contrast alone
does not distinguish these effects.

Only shortlisted ideas enter the predefined6/10 diagnostic, fresh on its own
split; no pair tuning. Then check verified SSC/ECG validation before frozen
canonical SHD* own-split four-arm/full-budget/seed replication. Official tests
remain outside development. No TTA or extra inference pass is part of reports.

Preregistration, diagnostic source rows, calibration, execution receipts and
results are beside `results/experiments/complementary_speaker_dev_v2/`.
Initial estimated training cost30-60minutes, refined by measured calibration.

## Pre-launch calibration outcome

All six arms passed actual one/two-graph finite-gradient long342frame calibration at512 and1024, including two warmups plus three synchronized measured steps. Batch1024 is frozen: two-pass throughput approximately445-447samples/s versus225-226 at512; reserved-memory headroom32.5% at1024. Initial-four estimated training seconds per epoch sum41.25 at1024 versus81.38 at512 (timing estimates, not convergence evidence). All validation256vs1024 checks retained identical predictions and CE differences below1e-5. Same LR0.01, six updates/epoch. All117states discarded. Validation-only candidate selection and30/300 distinction remain unchanged.

## C005 primary closure and fixed-pair sensitivity

Primary30 verified Baseline82.14%, RDrop83.85%, RDrop+LS86.30%, LS88.44%; combination rejected. Predeclared alpha0two-pass control81.04%; RDrop-control+2.813456pp, ordinaryCE-0.106498, late10mean+1.602446pp, bothspeaker contrastspositive. RDrop still worsens speaker3 versus the ordinary baseline; no uniform robustness or testclaim. All5actualinitialweights/RNG/settings and stage30/checkpoint/config/source/validation evidence verified before closure.

Only A005alpha1,epsilon0 is shortlisted for fresh Baseline/A sensitivity on predefined6/10, original frozenindices, seed42, batch1024/LR.01/dropout.15/full300schedule/30onlyceiling. Primaryweights saw6/10 and must never initialize this check. No failedcombination/LS/alpha0pair sweep, secondary tuning or extension. Require+0.5pp with ordinaryCE/curves support; preserve either outcome and accumulatedpair-exposure caveat. VerifiedSSC/ECGvalidation remains after this gate; finalcanonicalSHD*trainingmuststartfreshonitsownsplit. Officialtestsremainoutside development. See C005_SHORTLIST.json and C005_SENSITIVITY_STAGE030_PREREGISTRATION.json beside results.

## C005 fixed-pair closure and SSC transfer preparation

Fresh6/10 sensitivity30 verified Baseline83.82%, RDrop85.52%, +1.699030pp; ordinaryCE-0.008327, late10mean+1.116505pp. Speaker6-1.342282pp and10+3.422053pp: weightedgain,notuniformspeakerrobustness. Allsource278/protocol/initialstate/config/checkpoint/epoch/ordinaryvalidationchecks passed andcurvesinspected. No secondarytuning/extension.

Aonly nowqualifiesforbounded freshSSCseed42 Baseline/RDrop10,then40onlyifjustified. Useofficialvalidation anddataset-specificmean_membrane_potentials loss;alpha1/epsilon0fixed. Actualone/two-graph256/512calibration andindependentvalidationbatchbenchmark mustprecede frozencommonbatch;LR0.006unchanged. IsolatedSSCstudyfile iscomplementary_C005_dropout_ssc_v1.yaml. ExistingcanonicalSSCcontrolremainsintact;batchchangesrequirenewmatchedcontrol. NoB/combinationadvancement,ECGlaunch,canonicalSHD*trainingorofficialtestaccess yet. SeeC005_SENSITIVITY_STAGE030_DECISION.json andc005_dropout_ssc_v1/SSC_TRANSFER_PREREGISTRATION.jsonbesideresults.

## SSC pre-launch calibration

Sourcehashes/train75466/valid9981/all35classes anddisjointstoredspeakerIDs verified;nominalIDsnotindependentlyhumanverified. Actual256/512one/two-graph300framecalibration passedfinitegradients; batch512selected, validation1024,LR0.006unscaled,147updates/epoch. RDrop512~264.5samples/s,41.1%reservedheadroom;1024trainnotattemptedbecauseprojectedtwo-graphreserveexceeds85%GPU. Full9981validation at256/512/1024 retained0changed/repeatedpredictions andCEdifferences<1e-5. All117statesdiscarded;no checkpoints exported. Samepreprocessingsource andold8workerproducer2388.9samples/s comfortablyexceed536samples/smaximumcurrentconsumption;reuseisexplicitlyold1024batchworker evidence,notanew512benchmark.16relevantnontrainingtests passed. MainSSCusesnewisolatedfreshcontrol/RDrop,hard10ceiling/full40schedule;40extensiononlyafterreview.
