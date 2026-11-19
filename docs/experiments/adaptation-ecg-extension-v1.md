# ECG features, aligned time resampling and descriptive associations

**Completed by the user and verified — 16 November 2026:** the
[saved-evidence review](../../results/experiments/adaptation_ecg_extension_v1/POST_RUN_REVIEW.md)
confirmed 163 output hashes, 64 source hashes, 34 passing user tests, 180 feature
rows, 72 reused plus 36 new prediction chunks, and 2160 paired rows. All four
descriptive designs used all 30 records with no exclusions, rank 9 and 21
residual degrees of freedom. The recorded execution attempt was 83.891 seconds,
excluding initial preflight. No agent tests, forward, feature extraction or
regression were repeated during verification. Patient-population inference and
stage-6 completion are not claimed; remaining diagnostics/scope audit are tracked
in the roadmap. Preparation notes below describe the preserved workflow.

Implementation prepared 16 November 2026. This is an isolated stage-6 extension,
not training or stage 7. Scientific execution and all contract tests are
user-launched. No official-test MAT is opened.

The [versioned specification](../../config/studies/adaptation_ecg_extension_v1.yaml)
inherits only the ECG section of `config/paper_baseline.yaml` through the frozen
saved configs. Its source is the completed
[four-pair v2 package](../../results/experiments/adaptation_ecg_pairs_v2/POST_RUN_REVIEW.md),
with exact manifest/completion hashes. No frozen source, model, runner,
checkpoint or result manifest is changed.

The [preflight record](../../results/experiments/adaptation_mechanism_study_v1/ECG_EXTENSION_PREFLIGHT.json)
records artifact, loader, checkpoint, numerical-contract and map-metadata checks.
It explicitly distinguishes these from unexecuted tests, features, forward and
regression. A preflight pass is not a scientific extension result.

## Single user command

From the repository root, using the local project environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_extension.run --device cuda:0 --output-root results/experiments/adaptation_ecg_extension_v1
```

The command runs preflight, 34 user contract tests, source snapshot/validation,
input features, new mapped predictions, paired reports and descriptive
associations. Resume using exactly the same command and output root. Completed
compatible chunks are verified and reused; manifest or cache mismatches stop
the queue. An OS-owned `.analysis.lock` excludes concurrent writers. Atomic JSON
publication uses the existing bounded Windows retry writer.

All 30 validation rows, seeds 42/123/456 and four parent/variant pairs remain fixed.
All 72 existing clean/deletion chunks are preserved and reused; **only 36 new
chunks** are inferred (18 model/seed checkpoints × two time scales). MT parent
predictions are shared across its three pairs. The extension contains 180
model-independent feature rows, 108 effective prediction chunks and 2160 paired
rows (540 per pair). No new clean or deletion inference is scheduled.

Scheduling defaults to the same local PC: this adds half as many inference
chunks as the completed 72-chunk queue. There is no measured extension duration
yet. No training or PC-based result folders are introduced.

## Fixed input and time contract

Use only the same training MAT and validation indices. Preserve the original
argmax mapping of all-zero class vectors to class 0 and zero burn-in. Original
`max_i`, quiet bins or zero labels do not create a new mask. The saved numerical
contract remains float32 matrix precision `high`, without AMP.

For original length T and scale s, set M = floor(T*s + 0.5). For output position
j, choose source index `floor((2*j+1)*T/(2*M))`, clipped to T−1. This is nearest
source-bin-center resampling with later-bin tie breaking. **Exactly the same
index vector selects input, temporal class label and mask.** Rebuild blocks
1…M and let the existing collator prepend the ignored dummy target.

The stored 1301-step records become 1171 steps at 0.9× and 1431 at 1.1×.
Nearest-bin selection retains the signed input alphabet {-1, 0, +1} but may
remove or duplicate events. It can also change class proportions. Report the
original/mapped counts and their difference; interpret this as joint resampling
sensitivity, not an isolated count-preserving manipulation of duration.

Every prediction and feature row binds hashes for input, mapped targets, source
targets, mask, blocks and time map. Parents and variants must share these
identities exactly; mapped and clean comparisons share original record/label
identity, even though their mapped targets and lengths differ.

## Model-independent features

For each of the six conditions (clean, three deletion realizations, 0.9×, 1.1×):

- Activity is `sum(abs(x), channels)` per stored step. Retain positive/negative
  event counts and polarity balance separately; opposing polarities do not
  cancel in the activity descriptors.
- Fast/slow contrast is the population SD of complete, end-aligned 20-step
  minus 200-step activity rates, divided by mean activity per step.
- ACF lag is the first positive lag with centered biased ACF ≤ 1/e, searching
  up to 200 steps. No crossing is explicitly right-censored, with no numeric
  imputation. Silence/constant activity and insufficient windows have reasons.
- Record event counts, scored/stored steps, rate per step and six label-class
  fractions. These are model-independent; no feature definition is chosen from
  the new outcomes.

These windows and ACF values are **model steps**. Physical milliseconds are not
established by the MAT. Do not silently equate the numerical ECG descriptors
with physical-window SHD*/SSC measurements.

## Descriptive statistical protocol

For each pair, the primary outcome is clean per-record supervised loss benefit:
parent CE minus variant CE, separately for the three matching training seeds.
MT-TC auxiliary training penalties are excluded. Predictions and intervention
reports retain every record, including those with undefined features.

Joint feature-complete clean records enter one fixed descriptive OLS design:
intercept, the two standardized features, standardized log1p event rate and
duration if nonconstant, and class fractions 1…5 (class 0 is the reference).
Standardization uses complete-cohort population SD, fixed for all three seeds.
Constant controls are omitted with reasons; a constant primary feature prevents
estimation. The original duration is expected to be constant across these rows,
so it cannot identify a duration slope.

Require full rank at relative SVD tolerance 1e-10 and at least five residual
degrees of freedom. Otherwise report `not_estimable` with the design, complete
cases and reasons; do not add ridge, impute ACF or search for replacement models.
The analysis records per-seed coefficients, mean/sample SD over three training
seeds, residual sums, leverage and leave-one-record-out coefficient sensitivity
with full-cohort scaling held fixed. Missing-feature exclusions apply only to
this descriptive model; paired evaluation coverage remains complete.

The MAT supplies no verified patient IDs. There is **no** patient/record
bootstrap, population confidence interval, p-value or independent-patient
generalization claim. Row influence is a sensitivity diagnostic. Three training
seeds and repeated realizations are not extra independent patients. All analyses
are exploratory and conditional on previously selected validation checkpoints;
this is a versioned analysis plan, not a claim of prospective preregistration.

Intervention summaries average the three deletion realizations within each seed
before mean/sample SD across the three seeds. Report per-seed values and both
record-mean and step-weighted metrics. Compare benefit changes with the clean
input from the same original record, without averaging predictions as TTA.

## Preservation, outputs and review

User execution copies the source's 96 registered files and `COMPLETE.json`
byte-for-byte into `shared/ecg_pairs_v2/`. A path index maps original paths to the
snapshot without rewriting any completed manifest. A transferred extension can
resume using its snapshot when the original source directory is unavailable.
Exact checkpoints and the training MAT still must be transferred separately in
their portable repository-relative layout.

New output layout:

- `MANIFEST.json`, `CONTRACT_TESTS.json`/`.txt`, `PROGRESS.json`;
- `shared/ecg_pairs_v2/`, `REUSED_CLEAN_ANCHORS.json`;
- `features/<condition>/rep_<rep>.json`;
- `predictions/<model>/seed_<seed>/<condition>/rep_0.json` for mapped conditions;
- `PREDICTION_INDEX.json` linking both preserved and new chunks;
- `pairs/<pair>/paired_records.csv`, `ASSOCIATIONS.json`, `SUMMARY.json`/`.md`;
- overall `SUMMARY.json`/`.md` and hash-registered `COMPLETE.json`.

Agent-safe preflight (no feature extraction, tests, forward, fitting or analysis):

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_extension.run --check-only --device cuda:0
```

Optional user contract command (already run by the main command):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ecg tests.test_adaptation_ecg_suite tests.test_adaptation_ecg_extension
```

Read-only completion verification after execution:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg_extension.run --verify-only --output-root results/experiments/adaptation_ecg_extension_v1
```

Completion means this fixed extension executed, including explicit
non-estimability outcomes where applicable. It does not automatically close
stage 6. Review results, patient-dependence limits, outstanding direct SSC window
diagnostics and the broader study/code audit before deciding that scope is
complete. Stage 7 remains separate.
