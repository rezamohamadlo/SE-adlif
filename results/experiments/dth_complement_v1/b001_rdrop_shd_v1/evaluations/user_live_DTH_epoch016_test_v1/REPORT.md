# Current SHD* DTH control: explicitly requested exploratory test

Model: **DTH only**, not DTH + R-Drop. Training seed42; canonical SHD* random20percent validation with splitseed42. This is not speaker-holdout.

Validation-selected checkpoint: epoch16 (zero-based15), globalstep400. Training had completed 23 epochs at freezing. Validation accuracy: 97.731453%. Online training accuracy at selected epoch: 99.312502%.

Official-test accuracy: **92.667845%**; hard-label CE **0.39318020**; correct 2098/2264. Ordinary single-pass inference, no TTA.

Delta paper is local minus published paper mean in percentage points; computed from unrounded local accuracy. This short one-seed diagnostic is not a full300epoch multi-seed paper reproduction.

| Model | Official test accuracy (Delta paper) |
|---|---:|
| Published two-layer SE-adLIF,20runs | 93.79 +/- 0.76% |
| DTH current epoch16,one seed | 92.67% (-1.12 pp) |

The existing DTH30/DTH+RDrop30 training queue was not stopped or restarted. No DTH+RDrop SHD result exists from this diagnostic. Subsequent validation selection remains unchanged; no test-based checkpoint selection.

Checkpoint: `results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_live_DTH_epoch016_test_v1/best_validation.ckpt`. SHA256: `aa45f68226d7ce0be479a9eedd5d1e4e209b176af8e7739060fe1f2145dfe2b6`.

[Frozen plan](FROZEN_CHECKPOINTS.json) | [Full result](ALL_RESULTS.json) | [Paper Table2,page11](../../../../../../reference_paper.pdf#page=11)
