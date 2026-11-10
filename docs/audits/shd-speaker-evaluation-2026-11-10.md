# Speaker-holdout SHD: evaluation and next experiment

Reviewed 10 November 2026. The 24 completed runs cover four regularization
settings, DTH/LG-DTH and training seeds 42/123/456. All selected training,
validation and test values were checked against original CSV/JSON evidence.
Selected checkpoint bytes and both HDF5 dataset hashes were verified. No model
training or checkpoint inference was performed. Existing run artifacts and
comparison reports were not modified.

The study is promising as a held-out-speaker investigation. It does not yet
establish that increasing regularization improves official-test accuracy.
The strongest repeatable regularizer signal is DTH with 10% spike deletion:
validation improves in all three seeds. LG-DTH should retain its existing
dropout-0.25/deletion-5% recipe as the current validation-selected candidate.

## Verified results

Accuracies are percentages. Validation/test values are means plus/minus sample
SD over three matching training seeds. Training is the mean online training-mode
accuracy at the selected epoch; it is not a clean frozen-checkpoint evaluation.
All settings use time scaling 0.9-1.1 and channel shift +/-2 during training.

| Model | Setting | Train, selected | Final train | Validation | Official test, exploratory |
|---|---|---:|---:|---:|---:|
| DTH | A: dropout .25, deletion 5% | 99.76 | 99.48 | 90.56 +/- 0.92 | 93.14 +/- 0.34 |
| DTH | B: dropout .30, deletion 5% | 99.59 | 99.57 | 90.46 +/- 0.74 | 92.49 +/- 1.56 |
| DTH | C: dropout .35, deletion 5% | 99.40 | 99.68 | 91.05 +/- 0.97 | 93.15 +/- 2.34 |
| DTH | D: dropout .25, deletion 10% | 99.41 | 99.49 | **91.52 +/- 0.96** | 93.29 +/- 1.43 |
| LG-DTH | A: dropout .25, deletion 5% | 99.61 | 99.81 | **91.76 +/- 0.58** | **94.26 +/- 2.16** |
| LG-DTH | B: dropout .30, deletion 5% | 99.50 | 99.76 | 91.68 +/- 0.64 | 93.48 +/- 1.52 |
| LG-DTH | C: dropout .35, deletion 5% | 99.29 | 99.53 | 90.95 +/- 0.34 | 93.14 +/- 0.59 |
| LG-DTH | D: dropout .25, deletion 10% | 99.76 | 99.55 | 90.97 +/- 1.21 | 92.98 +/- 1.25 |

DTH-D versus DTH-A improves validation by +0.612/+1.346/+0.917 percentage
points across seeds 42/123/456: mean **+0.958 pp**. Official-test changes are
-1.148/+0.618/+0.972 pp, mean only **+0.147 pp**. D is the validation-selected
candidate to confirm, while A has much lower observed test variation.

LG-A has the largest validation mean and the largest observed test mean.
Its test scores are 93.68%, 96.64% and 92.45%; the high seed-123 score materially
raises the average. LG-A versus DTH-A changes test accuracy by
+0.839/+3.578/-1.060 pp. A stable architecture advantage is not established.
LG-A exceeds LG-B mean validation by just 0.082 pp, so that ordering is weak.
Increasing LG dropout to .35 reduces validation in all three matching seeds.

| Model | Change versus A | Mean validation change (pp) | Mean test change (pp), exploratory |
|---|---|---:|---:|
| DTH | B | -0.102 | -0.648 |
| DTH | C | +0.489 | +0.015 |
| DTH | D | +0.958 | +0.147 |
| LG-DTH | B | -0.082 | -0.780 |
| LG-DTH | C | -0.815 | -1.119 |
| LG-DTH | D | -0.795 | -1.281 |

These are descriptive differences, not significance tests. Three training
seeds describe initialization/run variability on one speaker split; they do
not estimate variability across held-out speaker pairs.

## Interpreting training, validation and test

Training exceeds 99% at epochs 26-37 in all runs. Validation uses entirely
held-out speakers 0 and 3, whereas training uses speakers 1,2,6,7,8,9,10,11.
The split has 6,521 fitted-training and 1,635 validation recordings. Near-perfect
training with lower speaker-held-out validation is consistent with speaker
sensitivity and memorization. The numerical gap mixes domains and measurement
modes: training has stochastic augmentation/dropout and changing weights;
validation uses clean evaluation mode. Clean training inference on the selected
checkpoint is needed for a like-for-like inference gap. Epoch-299 train accuracy
belongs to different weights and must not be paired with selected validation.

Validation accuracy generally improves and then plateaus; it does not uniformly
deteriorate with epoch. Across all 24 runs, mean validation accuracy increases
from 80.62% at epoch 9 to 86.75% at epoch 299, while validation cross entropy
increases from 0.785 to 1.320. Training cross entropy decreases from 0.125 to
0.015. This pattern is consistent with increasingly confident held-out errors;
calibration/confidence must be measured before treating that explanation as
established. Improving accuracy and worsening cross entropy can coexist.

The current best-validation scores are favorable peaks above the sustained
85-87% late-training accuracy. After epoch 50, mean absolute epoch-to-epoch
validation change is 2.206 pp. Selected maxima exceed their five-epoch
neighborhood mean by 2.843 pp on average. Selected epochs range from 94 to 297
(zero-based), so a universal short cutoff is not supported. Maximum-versus-final
differences alone cannot prove progressive overfitting; selection itself favors
maxima. Preserve the original maximum-validation-accuracy checkpoint rule for
these completed runs.

Every recorded LR is **0.01** throughout training. The inherited plateau
patience is 9999, making its schedule effectively constant within 300 epochs.
An explicitly versioned LR-decay experiment is therefore a reasonable next
optimization hypothesis; its benefit is not demonstrated by the current data.

The official test is predominantly, but not exclusively, unseen-speaker data:

| Test recording group | Speaker IDs | Recordings | Test fraction |
|---|---|---:|---:|
| Speakers absent from official training | 4,5 | 1,840 | 81.27% |
| Speakers used to fit these models | 1,2,6,7,8,9,10,11 | 339 | 14.97% |
| Speakers used for validation | 0,3 | 85 | 3.75% |

Total: 2,264 recordings. Counts were read from `extra/speaker` in the local
HDF5 files, whose hashes match the frozen split manifest. The dataset authors
also document 12 speakers, including two found exclusively in the test set:
[Zenke Lab dataset specification](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
Test accuracy higher than validation can reflect different speaker difficulty
and group composition; this ordering alone does not prove leakage.

## What can currently be claimed

A-D are matched regularizer comparisons on this one fixed split: shared
architecture within model, split seed 42, preprocessing, LR, batch 256, 300
epochs and checkpoint rule. Changes are the declared regularizers. Environment
and continuation records remain part of the original provenance.

Shuffled versus speaker-holdout comparisons change split membership,
dropout and augmentation simultaneously. They cannot isolate augmentation's
effect and are not matched paper-baseline comparisons. The published SHD*
mean is not a numeric target for this speaker protocol.

Settings were selected by the saved three-seed validation rule: DTH-D and
LG-A. B/C/D tests were observed before that merged selection, and the earlier
tests informed candidate proposals. These test scores remain exploratory;
validation-only selection afterward does not restore test independence. New
initialization seeds assess run robustness but also do not erase previous
test-set reuse.

Comparing DTH-D against LG-A compares separately tuned systems with different
deletion probabilities. To isolate the architecture difference, use one shared
frozen recipe, such as A, on both. A claim against SE-adLIF additionally needs a
reference run with that same speaker split and regularization; the current
shuffled reference does not provide that control.

## Recommended sequence

1. **Diagnose existing selected checkpoints before another sweep.** Measure
   clean training accuracy/loss and validation accuracy/loss by speaker and
   class. For the already-observed official test, retain the original aggregate
   and add speaker-wise results plus separate pooled scores for speakers 4/5,
   fitted-training speakers and validation speakers. Treat test subgroup results
   as diagnostics, not a new basis for selecting settings. No retraining is
   needed for these evaluations; they were not executed in this review.
2. **Retain a small, explicit candidate set.** Keep A on both models for a
   matched architecture comparison. Keep DTH-D as the one supported deletion
   ablation. Do not increase dropout again solely because training is near 100%.
   A confirmation of tuned systems may instead use DTH-D/LG-A, with that
   intentional regularizer difference disclosed.
3. **For another accuracy-improvement experiment, test LR decay first.** For
   example, compare A's existing constant LR with an opt-in cosine schedule
   from .01 to a small nonzero floor over the same 300 epochs, identically for
   DTH and LG-DTH. Freeze the schedule before viewing its results, use matching
   seeds, and select using validation only. This is a tuning protocol deviation,
   requiring new configs/output directories and saved resolved metadata, not a
   change to `config/paper_baseline.yaml` or existing completed runs. There is
   no guarantee of better validation or test results.
4. **Confirm the scientific claim.** For initialization robustness, freeze the
   candidate and evaluate fresh predeclared training seeds with split seed 42.
   For broader unseen-speaker robustness, predeclare additional speaker-group
   folds in a separately versioned study. If held-out-fold accuracy is used as
   an independent evaluation, reserve separate inner validation speakers for
   checkpoint/setting selection; otherwise label the folds as repeated
   validation evidence. Group-based splitting keeps all recordings of a
   speaker together; see [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
   and the [cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
   Use the supported runner for preflight/training/resume/testing/costs, inherit
   the dataset-specific paper baseline, and label all intentional deviations.
   All training remains user-launched.

No further dropout escalation or claim based on the 96.64% single-seed test
score is justified. The most defensible current candidates are LG-A and DTH-D,
with DTH-A retained as the stable regularization control.

## Evidence and reproduction

- [Current complete comparison](../../reports/speaker-based-summary.md).
- [24-run ledger](../../results/experiments/shd_speaker_combined_v1/RESULTS.json)
  and [frozen validation selection](../../results/experiments/shd_speaker_combined_v1/VALIDATION_SELECTION.json).
- [Recomputed per-run/groups/paired changes and data counts](shd-speaker-evaluation-2026-11-10/ANALYSIS.json).
- [Comparison and learning-curve figure](shd-speaker-evaluation-2026-11-10/evaluation.png),
  also available as [SVG](shd-speaker-evaluation-2026-11-10/evaluation.svg).
- [Read-only analysis script](shd-speaker-evaluation-2026-11-10.py). It reads saved
  metrics/configs/test records, hashes selected checkpoints/HDF5 files, and
  writes only this audit's derived JSON/plots. It never loads a model.

```powershell
.\.ven\Scripts\python.exe docs/audits/shd-speaker-evaluation-2026-11-10.py
```

The local environment is `.ven`; use the project environment name appropriate
to the checkout. Verification completed: all 24 histories contain epochs 0-299,
all selected metrics/test records agree with the ledger, selected checkpoint
hashes match, dataset hashes match, and generated figure was visually inspected.
