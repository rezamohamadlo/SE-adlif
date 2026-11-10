# SHD test results: shuffled split and speaker-based regularization

Snapshot: **10 November 2026**, after importing and verifying PC1/PC3 results
alongside this machine's PC2 records. This report now uses the
[combined experiment collection](../results/experiments/shd_speaker_combined_v1/README.md):
**24 completed speaker-based runs**, covering four settings, two models and
three training seeds. All official test results use **ordinary inference
without TTA** on the official SHD test set. Training and validation accuracies
are labeled separately. All local runs completed 300 training epochs; their
checkpoints were selected using validation accuracy.

**SHD*** denotes the random recording split: 80% of the official training set
for training and 20% for validation, split seed 42, without enforced speaker
separation. The four **speaker-holdout** experiments instead use validation
speakers **0 and 3**, training speakers **1, 2, 6, 7, 8, 9, 10 and 11**, and
the same frozen split seed 42. These are distinct evaluation protocols.

The four speaker-based settings are:

- **A — augmented baseline:** dropout **0.25**, spike deletion **5%**.
- **B — increased dropout:** dropout **0.30**, spike deletion **5%**.
- **C — increased dropout:** dropout **0.35**, spike deletion **5%**.
- **D — increased deletion:** dropout **0.25**, spike deletion **10%**.

All four share training-only time scaling **0.9–1.1** and channel shift **±2**.
The shuffled baselines use dropout **0.15** and no training augmentation. The
official test inputs use ordinary preprocessing throughout. Local batch size
is 256 and LR is 0.01, with no early stopping. Shared settings inherit
`paper_classification_v1`; the speaker split and regularization are explicit
experiment deviations.

## What is being compared, and how fair is it?

**Tables 1A, 1B and 2 show official test accuracy**, not training or validation
accuracy. Tables 3A-3D separate training, validation and test metrics for each
of the four speaker settings, with matching shuffled DTH/LG-DTH rows.

The four speaker-holdout settings use matching models, training seeds, speaker
split, preprocessing, optimizer settings and training budget. B and C change
dropout relative to A; D changes spike deletion relative to A. They therefore
support a controlled regularization comparison within this fixed speaker split.
Runs were collected across three PCs and recorded source revisions; seed-42
continuation history and environment provenance remain in the source records.
Matching settings do not guarantee identical random trajectories.

The shuffled-versus-speaker comparison is **descriptive, rather than a strictly
matched experiment**: the validation speakers, training/validation membership,
augmentation and dropout differ. A higher shuffled validation score does not
establish better generalization to held-out speakers. The paper comparison also
has a different run count and lacks matched paper training seeds.

For the speaker-holdout models, the 2,264 official test recordings comprise
1,840 (81.27%) from speakers
4/5, absent from official training; 339 (14.97%) from fitted-training speakers;
and 85 (3.75%) from validation speakers 0/3. Thus the aggregate is predominantly,
but not exclusively, unseen-speaker accuracy. Different speaker difficulty and
group composition can make test accuracy exceed validation accuracy. Counts
were checked against the HDF5 speaker metadata and the
[frozen dataset identity](../results/experiments/shd_speaker_combined_v1/SPLIT_MANIFEST.json).

The B/C/D official tests were observed before merged settings selection. Those
test comparisons remain exploratory. Settings selection uses the predeclared
three-seed validation rule; test accuracy is not used to select the winner.

## Table 1A — Paper, local reference and shuffled test averages

The paper benchmark and local reference describe **SE-adLIF**. The first four
rows compare the published paper, local reference and shuffled proposed models
under the random-recording SHD* validation protocol. The paper reports **20 runs**;
local results use training seeds **42, 123 and 456** (**n=3**).

Values are test accuracy (%) as mean ± SD. Local SDs use the sample definition
(n−1); the paper SD is quoted as published.

**Δ paper** = local mean test accuracy minus the published SHD* mean
**93.79%**, in **percentage points (pp)**, calculated before rounding. Means
are compared with means; SDs are not subtracted. The paper does not report
training or validation accuracy, so those metrics have no paper delta.

| Source / experiment | Model | Validation protocol | Official test, mean ± SD (Δ paper) |
|---|---|---|---:|
| Published paper | SE-adLIF (2 layers) | SHD*: random 20% | 93.79 ± 0.76 |
| Local reference | SE-adLIF | SHD*: random 20% | 93.15 ± 0.42 (-0.64 pp) |
| Shuffled baseline | DTH | SHD*: random 20% | 94.49 ± 0.52 (+0.70 pp) |
| Shuffled baseline | LG-DTH | SHD*: random 20% | 93.74 ± 0.71 (-0.05 pp) |

## Table 1B — Speaker-based versus shuffled test averages

Each row compares one speaker-based setting with the shuffled baseline for
**the same model and training seeds 42/123/456**. All means and sample SDs use
**n=3**. Values are official test accuracy (%); all eight speaker-based rows
are shown separately from the paper comparison.

**Δ shuffled** = speaker-based mean test accuracy minus the matching
shuffled model's mean test accuracy, in pp, using unrounded results. The change
is descriptive: validation membership, dropout and augmentation differ between
protocols. Speaker-holdout uses validation speakers 0/3, so these rows have no
numeric paper delta. All speaker-based test comparisons remain exploratory.

| Speaker-based setting | Model | Shuffled SHD*, mean ± SD | Speaker-holdout, mean ± SD (Δ shuffled) |
|---|---|---:|---:|
| Drop .25 / delete 5% | DTH | 94.49 ± 0.52 | 93.14 ± 0.34 (-1.35 pp) |
| Drop .25 / delete 5% | LG-DTH | 93.74 ± 0.71 | 94.26 ± 2.16 (+0.52 pp) |
| Drop .30 / delete 5% | DTH | 94.49 ± 0.52 | 92.49 ± 1.56 (-2.00 pp) |
| Drop .30 / delete 5% | LG-DTH | 93.74 ± 0.71 | 93.48 ± 1.52 (-0.27 pp) |
| Drop .35 / delete 5% | DTH | 94.49 ± 0.52 | 93.15 ± 2.34 (-1.34 pp) |
| Drop .35 / delete 5% | LG-DTH | 93.74 ± 0.71 | 93.14 ± 0.59 (-0.60 pp) |
| Drop .25 / delete 10% | DTH | 94.49 ± 0.52 | 93.29 ± 1.43 (-1.21 pp) |
| Drop .25 / delete 10% | LG-DTH | 93.74 ± 0.71 | 92.98 ± 1.25 (-0.77 pp) |

PC1 supplied seed 42; PC2/PC3 supplied their assigned seed-123/456 runs.
Repeated runs of one training seed are not counted as independent seeds.

## Table 2 — Official test accuracy: each speaker setting versus shuffled, by seed

Each cell is **test accuracy (%)**. Parentheses in A–D give **Δ shuffled**:
speaker-holdout test accuracy minus the shuffled test accuracy for the **same
model and training seed**, in pp, using unrounded values. This is a descriptive
comparison across different validation protocols and training regularization.
It does not isolate dropout from the speaker split or augmentation when
comparing against shuffled.

All cells below now have verified local test records. No test score is inferred
from a validation result.

| Model | Training seed | Shuffled SHD* | A: Drop .25 / delete 5% (Δ shuffled) | B: Drop .30 / delete 5% (Δ shuffled) | C: Drop .35 / delete 5% (Δ shuffled) | D: Drop .25 / delete 10% (Δ shuffled) |
|---|---:|---:|---:|---:|---:|---:|
| DTH | 42 | 94.70 | 92.84 (-1.86 pp) | 91.92 (-2.78 pp) | 92.01 (-2.69 pp) | 91.70 (-3.00 pp) |
| DTH | 123 | 94.88 | 93.07 (-1.81 pp) | 91.30 (-3.58 pp) | 95.85 (+0.97 pp) | 93.68 (-1.19 pp) |
| DTH | 456 | 93.90 | 93.51 (-0.40 pp) | 94.26 (+0.35 pp) | 91.61 (-2.30 pp) | 94.48 (+0.57 pp) |
| LG-DTH | 42 | 93.77 | 93.68 (-0.09 pp) | 92.09 (-1.68 pp) | 92.89 (-0.88 pp) | 91.65 (-2.12 pp) |
| LG-DTH | 123 | 93.02 | 96.64 (+3.62 pp) | 93.24 (+0.22 pp) | 93.82 (+0.80 pp) | 94.13 (+1.10 pp) |
| LG-DTH | 456 | 94.43 | 92.45 (-1.99 pp) | 95.10 (+0.66 pp) | 92.71 (-1.72 pp) | 93.15 (-1.28 pp) |

## Tables 3A-3D — Training, validation and test: each speaker setting versus shuffled

Each table compares one speaker-based setting with the same six shuffled
model/seed baselines. Adjacent rows share the model and training seed. Repeating
shuffled rows makes each comparison self-contained; it does not add new runs.

Compact labels give the dropout probability and spike-deletion percentage.
All four settings retain the shared time scaling and channel-shift augmentation;
`delete 10%` specifically means deleting 10% of spikes during training.

Train (selected epoch) is the online training-mode accuracy logged during the
epoch whose checkpoint was selected by validation. Final train is epoch-299
accuracy. Validation and test use evaluation mode with ordinary inputs.
Training uses dropout and, for A-D, stochastic augmentation. Its accuracy is
not a clean evaluation of the frozen checkpoint on the entire training set;
furthermore, final train belongs to different weights from the selected checkpoint.

All accuracies are percentages. Validation/test refer to the same selected
checkpoint. The final column is the speaker-setting test accuracy minus its
matching shuffled test accuracy, in pp, calculated from unrounded values.
Shuffled-versus-speaker differences remain descriptive because split membership
and regularization differ. These tables do not compare individual seeds with
a paper mean; the applicable mean-to-mean paper deltas are in Table 1A.

### Table 3A — Drop .25 / delete 5% versus shuffled

| Model | Seed | Protocol / setting | Train, selected epoch (%) | Final train (%) | Validation, selected (%) | Official test (%) | Δ test vs shuffled (pp) |
|---|---:|---|---:|---:|---:|---:|---:|
| DTH | 42 | Shuffled SHD* | 100.00 | 100.00 | 99.14 | 94.70 | — |
| DTH | 42 | Drop .25 / delete 5% | 99.55 | 99.39 | 90.03 | 92.84 | -1.86 |
| DTH | 123 | Shuffled SHD* | 100.00 | 99.70 | 99.08 | 94.88 | — |
| DTH | 123 | Drop .25 / delete 5% | 99.89 | 99.62 | 90.03 | 93.07 | -1.81 |
| DTH | 456 | Shuffled SHD* | 100.00 | 100.00 | 99.02 | 93.90 | — |
| DTH | 456 | Drop .25 / delete 5% | 99.83 | 99.44 | 91.62 | 93.51 | -0.40 |
| LG-DTH | 42 | Shuffled SHD* | 100.00 | 99.75 | 99.14 | 93.77 | — |
| LG-DTH | 42 | Drop .25 / delete 5% | 99.70 | 99.84 | 91.19 | 93.68 | -0.09 |
| LG-DTH | 123 | Shuffled SHD* | 100.00 | 100.00 | 99.26 | 93.02 | — |
| LG-DTH | 123 | Drop .25 / delete 5% | 99.55 | 99.78 | 92.35 | 96.64 | +3.62 |
| LG-DTH | 456 | Shuffled SHD* | 100.00 | 99.22 | 99.02 | 94.43 | — |
| LG-DTH | 456 | Drop .25 / delete 5% | 99.58 | 99.80 | 91.74 | 92.45 | -1.99 |

### Table 3B — Drop .30 / delete 5% versus shuffled

| Model | Seed | Protocol / setting | Train, selected epoch (%) | Final train (%) | Validation, selected (%) | Official test (%) | Δ test vs shuffled (pp) |
|---|---:|---|---:|---:|---:|---:|---:|
| DTH | 42 | Shuffled SHD* | 100.00 | 100.00 | 99.14 | 94.70 | — |
| DTH | 42 | Drop .30 / delete 5% | 99.37 | 99.72 | 90.89 | 91.92 | -2.78 |
| DTH | 123 | Shuffled SHD* | 100.00 | 99.70 | 99.08 | 94.88 | — |
| DTH | 123 | Drop .30 / delete 5% | 99.73 | 99.28 | 89.60 | 91.30 | -3.58 |
| DTH | 456 | Shuffled SHD* | 100.00 | 100.00 | 99.02 | 93.90 | — |
| DTH | 456 | Drop .30 / delete 5% | 99.66 | 99.70 | 90.89 | 94.26 | +0.35 |
| LG-DTH | 42 | Shuffled SHD* | 100.00 | 99.75 | 99.14 | 93.77 | — |
| LG-DTH | 42 | Drop .30 / delete 5% | 99.41 | 99.83 | 91.01 | 92.09 | -1.68 |
| LG-DTH | 123 | Shuffled SHD* | 100.00 | 100.00 | 99.26 | 93.02 | — |
| LG-DTH | 123 | Drop .30 / delete 5% | 99.72 | 99.78 | 91.74 | 93.24 | +0.22 |
| LG-DTH | 456 | Shuffled SHD* | 100.00 | 99.22 | 99.02 | 94.43 | — |
| LG-DTH | 456 | Drop .30 / delete 5% | 99.37 | 99.67 | 92.29 | 95.10 | +0.66 |

### Table 3C — Drop .35 / delete 5% versus shuffled

| Model | Seed | Protocol / setting | Train, selected epoch (%) | Final train (%) | Validation, selected (%) | Official test (%) | Δ test vs shuffled (pp) |
|---|---:|---|---:|---:|---:|---:|---:|
| DTH | 42 | Shuffled SHD* | 100.00 | 100.00 | 99.14 | 94.70 | — |
| DTH | 42 | Drop .35 / delete 5% | 99.27 | 99.53 | 90.52 | 92.01 | -2.69 |
| DTH | 123 | Shuffled SHD* | 100.00 | 99.70 | 99.08 | 94.88 | — |
| DTH | 123 | Drop .35 / delete 5% | 99.52 | 99.75 | 92.17 | 95.85 | +0.97 |
| DTH | 456 | Shuffled SHD* | 100.00 | 100.00 | 99.02 | 93.90 | — |
| DTH | 456 | Drop .35 / delete 5% | 99.42 | 99.77 | 90.46 | 91.61 | -2.30 |
| LG-DTH | 42 | Shuffled SHD* | 100.00 | 99.75 | 99.14 | 93.77 | — |
| LG-DTH | 42 | Drop .35 / delete 5% | 99.61 | 99.62 | 91.01 | 92.89 | -0.88 |
| LG-DTH | 123 | Shuffled SHD* | 100.00 | 100.00 | 99.26 | 93.02 | — |
| LG-DTH | 123 | Drop .35 / delete 5% | 99.44 | 99.45 | 90.58 | 93.82 | +0.80 |
| LG-DTH | 456 | Shuffled SHD* | 100.00 | 99.22 | 99.02 | 94.43 | — |
| LG-DTH | 456 | Drop .35 / delete 5% | 98.83 | 99.52 | 91.25 | 92.71 | -1.72 |

### Table 3D — Drop .25 / delete 10% versus shuffled

| Model | Seed | Protocol / setting | Train, selected epoch (%) | Final train (%) | Validation, selected (%) | Official test (%) | Δ test vs shuffled (pp) |
|---|---:|---|---:|---:|---:|---:|---:|
| DTH | 42 | Shuffled SHD* | 100.00 | 100.00 | 99.14 | 94.70 | — |
| DTH | 42 | Drop .25 / delete 10% | 99.42 | 99.64 | 90.64 | 91.70 | -3.00 |
| DTH | 123 | Shuffled SHD* | 100.00 | 99.70 | 99.08 | 94.88 | — |
| DTH | 123 | Drop .25 / delete 10% | 99.25 | 99.55 | 91.38 | 93.68 | -1.19 |
| DTH | 456 | Shuffled SHD* | 100.00 | 100.00 | 99.02 | 93.90 | — |
| DTH | 456 | Drop .25 / delete 10% | 99.56 | 99.28 | 92.54 | 94.48 | +0.57 |
| LG-DTH | 42 | Shuffled SHD* | 100.00 | 99.75 | 99.14 | 93.77 | — |
| LG-DTH | 42 | Drop .25 / delete 10% | 99.94 | 99.31 | 90.09 | 91.65 | -2.12 |
| LG-DTH | 123 | Shuffled SHD* | 100.00 | 100.00 | 99.26 | 93.02 | — |
| LG-DTH | 123 | Drop .25 / delete 10% | 99.67 | 99.70 | 90.46 | 94.13 | +1.10 |
| LG-DTH | 456 | Shuffled SHD* | 100.00 | 99.22 | 99.02 | 94.43 | — |
| LG-DTH | 456 | Drop .25 / delete 10% | 99.66 | 99.64 | 92.35 | 93.15 | -1.28 |

Shuffled training/validation values come from the original CSVs and matching
checkpoint cost records in [comparison evidence](../results/experiments/shd_speaker_combined_v1/comparison_evidence/shuffled/).
Speaker values come from the combined 24-run ledger and original metrics.

## Interpretation and provenance

Higher dropout has mixed effects across matching seeds. C improves both
validation and test accuracy over A for DTH seed 123, but its three-seed DTH
test mean is almost unchanged from A. B and C both have lower three-seed
LG-DTH test means than A. D has a slightly higher DTH test mean than A and a
lower LG-DTH test mean. These descriptive test differences do not establish
a consistent gain or statistical significance.

Applying the predeclared maximum three-seed **validation** mean rule, including
A as an eligible choice, selects D for DTH (91.5189% mean validation) and A
for LG-DTH (91.7635%). The [saved validation selection](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_regularization_v1/VALIDATION_SELECTION.json)
uses validation scores only.

For DTH, D improves validation over A in all three seeds (+0.61/+1.35/+0.92 pp;
mean +0.96 pp), while its mean test gain is only +0.15 pp. LG-DTH retains A;
its mean test advantage is sensitive to the 96.64% seed-123 result. These are
candidates to confirm, not established improvements. Comparing DTH-D against
LG-DTH-A also changes deletion probability; an architecture-only comparison
must use the same regularization recipe on both models.

The B/C/D tests were requested on the individual PCs **before completion of
the merged three-seed settings selection**. They remain exploratory and are
saved separately from the validation-only training records. The later merge
and validation-only selection do not turn these previously observed test
results into independent confirmation evidence.

The current reference is the canonical `results/valid/reference/SHD/` record.
Its saved aggregate retains protocol `paper_shd_reproduction` and baseline
version `paper_classification_v1`. The shuffled LG-DTH seeds 42/123 retain
their historical `controlled_v2_custom_b134d147776c` identifiers; seed 456
retains `paper_classification_v1`. Their saved SHD split, budget, batch,
learning rate and dropout were checked, without relabeling historical runs.
LG-DTH seed 456 is the full-budget protocol-corrected rerun performed after
an earlier early-stopped test result had been observed; that earlier run is
excluded here.

Only completed official-test results enter the tables. Means use the exact
available seed sets listed above, and local SDs use the sample definition (n−1).
Different sample sizes and the paper's 20-run cohort prevent treating a mean
difference as a statistical significance result. No architecture or setting
is selected by the highest test value in this report.

## Sources

- **Combined speaker-based results:** [all 24 run records and eight three-seed aggregates](../results/experiments/shd_speaker_combined_v1/RESULTS.json), [summary and per-seed tables](../results/experiments/shd_speaker_combined_v1/RESULTS.md), and [flat CSV](../results/experiments/shd_speaker_combined_v1/RUNS.csv). Accuracies are stored as unrounded fractions; this report converts them to percentages before rounding for display.
- **Collection verification:** [source inventory and 192 evidence snapshots](../results/experiments/shd_speaker_combined_v1/SOURCE_MANIFEST.json), [collection file hashes](../results/experiments/shd_speaker_combined_v1/MANIFEST.json), and [validation-only settings selection](../results/experiments/shd_speaker_combined_v1/VALIDATION_SELECTION.json). Each run record retains its original protocol, checkpoint paths and SHA-256 identities.
- **Paper:** [published two-layer SHD* benchmark and extraction source](paper-results.md).
- **Local SE-adLIF reference:** [canonical aggregate](../results/valid/reference/SHD/REFERENCE_RESULTS.json), checked against per-seed `REFERENCE_RESULT.json` records in the [canonical reference directory](../results/valid/reference/SHD/).
- **Shuffled DTH:** verified ordinary test reproduction values in the per-seed cost records: [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json).
- **Shuffled LG-DTH:** the corresponding retained records: [42](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json). Test averages retain the accepted unrounded `reproduced_test_acc` values. Matching original shuffled metric CSVs are now available under the combined comparison evidence and supply Tables 3A-3D training/validation values.
- **A:** unrounded final test rows from each seed's original `metrics.csv`, with saved configs and measurement records; aggregates: [DTH](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_v1/DTH_SE_adLIF/RESULTS.md), [LG-DTH](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_v1/LG_DTH_SE_adLIF/RESULTS.md). [Frozen speaker split](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_v1/SPLIT_MANIFEST.json).
- **B/C/D, PC1:** [seed-42 test records and evaluated checkpoint copies](../results/experiments/shd_speaker_combined_v1/artifacts/shd_regularization_test_seed42_v1/), with copied source validation records; six test checkpoints match their training selections byte for byte.
- **B/C, PC2:** the six per-run `TEST_RESULTS.json` and `MANIFEST.json` records behind the [PC2 exploratory ordinary test report](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_regularization_tests_v1/RESULTS.md).
- **C/D, PC3:** [PC3 exploratory ordinary test report](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_regularization_test_v1/RESULTS.md) and its six per-run `TEST_RESULTS.json` records.
- **Combined verification:** [transfer inventory and 18 verified run identities](../results/provenance/shd-regularization-pc123-transfer-2026-11-10.json), including all source/copy SHA-256 hashes. Validation completion/selection is retained in the [merged tuning results](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_regularization_v1/).
- **Experiment definitions:** [A](../config/shd_speaker_study/augmented_v1.yaml), [B](../config/shd_speaker_study/dropout030_v1.yaml), [C](../config/shd_speaker_study/dropout035_v1.yaml), [D](../config/shd_speaker_study/spike_delete010_v1.yaml), and the shared [paper baseline](../config/paper_baseline.yaml).

Historical exploratory [TTA results](../results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_tta_v1/RESULTS.md)
remain separately labeled inside the combined collection and are excluded from
all current comparison tables. No training or inference was performed to
update this report.
