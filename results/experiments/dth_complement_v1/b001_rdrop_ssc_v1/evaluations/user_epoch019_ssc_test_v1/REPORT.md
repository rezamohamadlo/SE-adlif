# SSC DTH + R-Drop: requested epoch-19 test

Test accuracy: **80.531842%**. Hard-label test CE: **0.66630065**. Correct: 16414 / 20382. Ordinary single-pass inference; no TTA.

Selected checkpoint: epoch19 (zero-based18), validation accuracy 82.116020%. It is the archived validation-selected checkpoint from the completed20 stage, frozen before this test. Training had completed 29 epochs when this test was requested. The existing40epoch training and final-test queue were not stopped, restarted or modified.

Delta paper means local minus published paper mean in percentage points, calculated from unrounded local accuracy. This is one local seed versus five paper runs.

| Model | Test accuracy (Delta paper) |
|---|---:|
| Published two-layer SE-adLIF | 80.44 +/- 0.26% |
| DTH + R-Drop, epoch19 | 80.53% (+0.09 pp) |

Difference versus the previously tested epoch15 checkpoint: +0.201158 pp. Difference versus historical DTH10 control: +1.074477 pp; budgets differ, so this is not an architecture-only matched comparison. Prior tests were exposed; this is exploratory and does not establish significance or independent confirmation.

Checkpoint SHA256: `15561257012d2eba48286a538e04c56c5b392bf8651ca89544fcf430e460649f`.

[Frozen plan](FROZEN_CHECKPOINTS.json) | [Full test evidence](ALL_RESULTS.json) | [Paper Table2, page11](../../../../../../reference_paper.pdf#page=11)

No new training, idea or hyperparameter decision follows from this test. The separately authorized existing training to40 and its final evaluation remain scheduled.
