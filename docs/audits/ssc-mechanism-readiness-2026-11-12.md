# SSC mechanism-study readiness audit

Date: 12 November 2026. Scope: read-only inspection and artifact/config
preflight. No training, inference, statistical fit, resampling, dataset
constructor or tests ran. No official-test file was opened. Audit records
were written separately; no existing evidence or dataset copies were changed.

Machine-readable evidence: [SSC_PREFLIGHT.json](../../results/experiments/adaptation_mechanism_study_v1/SSC_PREFLIGHT.json).
SHD consolidation: [SHD_SUMMARY.md](../../results/experiments/adaptation_mechanism_study_v1/SHD_SUMMARY.md).

## Readiness outcome

**Local data and checkpoint/config prerequisites pass. A validation-only SSC
mechanism evaluator is not implemented yet, and clean validation inference
reproduction remains unverified.** This is permission to prepare that workflow,
not a claim that an SSC mechanism experiment has run or its anchors will pass.
No new launch command is available from this audit.

The prior roadmap's missing-SSC-data snapshot is superseded by the current
inspection. The saved configurations use `data_path: data`; Tonic's Dataset
initializer appends the actual subclass name, so `SSCWrapper` uses:

```text
data/SSCWrapper/ssc_valid.h5
```

- Official validation records: **9,981**, covering 35 class IDs.
- Aligned speaker metadata: `extra/speaker`, **256** distinct speaker groups.
- Validation SHA-256:
  `6365c36ff8c680c4b0caf27ca52faa59e14359db5810b528f40dd80e2bcb9210`.
- The retained `data/SSCWrapper/ssc_valid.h5.zip` matches the validation MD5
  declared by the installed `tonic.datasets.SSC`. Its H5 member was streamed
  and SHA-256 checked against the retained validation H5, with an exact match.
  Checksums/member identity are in the JSON audit. No files were extracted to
  disk and no dataset was downloaded or modified.
- The second copy at `data/SSC/ssc_valid.h5` has the same byte hash. Both copies
  remain unchanged. The latter is not substituted for a missing recorded path.
- Training/test H5/ZIP filenames are present in these directories, but only
  the two validation H5 files were opened/hashed by this audit. No test outcome
  or test-input content was inspected.

Validation H5 arrays `labels`, `spikes/times`, `spikes/units` and `extra/speaker`
have consistent membership lengths. Timestamp storage is variable-length
float16, unit storage uint16. Three representative validation records were
checked for aligned lengths, finite nonnegative and nondecreasing timestamps
after widening to float64. This is not a full raw-data or preprocessing audit.
No features or model predictions were generated.

## Checkpoints and matching settings

All eighteen selected checkpoints exist and match the SHA-256 recorded in their
completed cost/config evidence. Trusted local checkpoints were deserialized on
CPU to inspect saved configuration and selected epoch; no model forward or
training function was called. Checkpoint and measurement effective configs
match. Each original metric CSV has a single complete 40-epoch validation
history, and each recorded selected epoch attains the maximum saved validation
accuracy, allowing numerical tolerance and ties.

| Model | Seed 42 selected epoch | Seed 123 | Seed 456 |
|---|---:|---:|---:|
| MT | 23 | 36 | 17 |
| DTH | 19 | 39 | 11 |
| AMT | 35 | 27 | 13 |
| LT-DTH | 16 | 20 | 24 |
| SA-MT | 13 | 26 | 27 |
| MT-TC | 16 | 26 | 28 |

Actual checkpoint identities, cost/config paths, metric-file hashes and fully
resolved saved configs are in the JSON audit; these epochs do not authorize
substitution of a similarly named checkpoint. Parent pairing remains AMT/MT,
SA-MT/MT, MT-TC/MT and LT-DTH/DTH.

Checks against `config/paper_baseline.yaml`'s SSC section passed: LR .006,
batch 256, 40 epochs, dropout .15, no early stopping, two 720-neuron layers,
140 input channels, 35 classes, and matching registered neuron types and
shared architecture/optimization/preprocessing settings. Intended model
differences are retained. Ordinary objectives have `ssc_mean_loss_v1` and
`mean_membrane_potentials`; MT-TC has the distinct
`ssc_mean_loss_v1_mt_temporal_consistency_v1` objective protocol with training
JS weight .1/window 5. The eventual comparison must use common supervised
mean-membrane classification loss, excluding the JS training penalty.

SSC uses the **official validation split**, not SHD*'s 20% training holdout.
The input bin is four ms, spatial factor .2, time factor .001, minimum length
300 frames and burn-in ten frames. Do not apply SHD's 300 epochs, LR .01,
sum-softmax aggregation or validation sampling assumptions to SSC.

## Loader and provenance gates before a pilot

The current `datasets/ssc.py` override widens float16 timestamps **before**
converting seconds to microseconds. Preserve that operation order in the
diagnostic loader; Tonic's direct path must not reintroduce float16 overflow.
The audit JSON records the current wrapper source hash. It does not establish
byte-identical historical preprocessing or uninterrupted training equivalence.

Do not call `SSCLDM.setup` as a validation-only shortcut: it constructs train,
test and validation datasets. Tonic's dataset constructor can also download
missing data. A new diagnostic loader must open only the authenticated
validation H5, reproduce the saved framing/downsampling and input channel
layout, retain physical support separately from minimum/batch padding, and
preserve class/speaker IDs. Readiness inspection alone has not verified those
numerical contracts or reproduced any checkpoint's validation loss/accuracy.

`COST_RESULTS.json`'s `data_sha256` comes from test cost evaluation, not
validation-file provenance. It was not used as the expected validation hash.
The validation file is independently hashed above, with the source contexts
kept distinct. No archived summed-loss SSC records or current comparison
settings were rewritten.

## Next bounded step

Prepare a separate versioned **AMT/MT SSC validation pilot**, preserving these
checkpoint identities and the SSC baseline. First prepare the validation-only
loader and non-training contracts, then preflight. The user should launch a
small, fixed sample subset for feature/intervention diagnostics with full clean
validation anchors as a required compatibility gate. Check planned runtime
before deciding whether a long workload needs multiple PCs. Output paths
remain independent of scheduling.

Do not launch from this audit. If the clean anchors fail, inspect data, framing,
padding/masks, saved configs and aggregation rather than changing checkpoints
or accepting whichever result is higher. Feature definitions and perturbations
need an SSC-specific resolved contract; cross-dataset statistics and shared
loss-unit pooling are not yet justified. Preserve the negative/mixed SHD*
findings while extending the study.
