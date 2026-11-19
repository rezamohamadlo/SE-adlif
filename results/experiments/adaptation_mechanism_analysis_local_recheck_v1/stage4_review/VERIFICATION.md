# Stage 4 pilot review

Saved-output hashes, cohort identity, cache identities, paired inputs and deterministic input reconstruction passed. All 256 inputs were reconstructed from the authenticated SHD training file; transformed hashes and features matched saved rows for both models and all seeds. No model inference or training was run.

Checks: channel counts preserved by time warping; thinning never creates events; support lengths, padding metadata and identity transforms agree. Clean predictions agree between arms; maximum per-sample clean loss difference: 2.288818359375e-05.

Benefit = supervised loss(MT) - supervised loss(AMT). Benefit change = transformed benefit - clean benefit. Positive favors AMT. Accuracy gap = AMT - MT in percentage points. Mean +/- sample SD across three training seeds. The three deletion realizations are averaged within each seed first.

| Condition | Training seeds | Benefit | Change from clean | Accuracy gap (pp) |
|---|---:|---:|---:|---:|
| pc2 / clean | 3 | +0.003717 +/- 0.039030 | +0.000000 +/- 0.000000 | +0.000000 +/- 0.676582 |
| pc2 / time_scale_0.9 | 3 | -0.009238 +/- 0.047019 | -0.012956 +/- 0.015098 | -0.260417 +/- 0.225527 |
| pc2 / time_scale_1.1 | 3 | -0.006206 +/- 0.016131 | -0.009924 +/- 0.026613 | -0.260417 +/- 0.451055 |
| pc3 / clean | 3 | +0.003717 +/- 0.039030 | +0.000000 +/- 0.000000 | +0.000000 +/- 0.676582 |
| pc3 / delete_0.1 | 3 | +0.007811 +/- 0.025736 | +0.004093 +/- 0.014537 | -0.130208 +/- 0.130208 |

## Transformation checks

| Condition | Replicate | Removed events (%) | Mean support-length ratio |
|---|---:|---:|---:|
| pc2 / clean | 0 | 0.0000 | 1.000000 |
| pc2 / time_scale_0.9 | 0 | 0.0000 | 0.902635 |
| pc2 / time_scale_1.1 | 0 | 0.0000 | 1.102825 |
| pc3 / clean | 0 | 0.0000 | 1.000000 |
| pc3 / delete_0.1 | 0 | 10.0086 | 1.000000 |
| pc3 / delete_0.1 | 1 | 9.9908 | 1.000000 |
| pc3 / delete_0.1 | 2 | 10.0160 | 1.000000 |

## Interpretation and next step

The pilot does not establish a consistent AMT advantage across training seeds. Time-scale effects and deletion effects must be interpreted through changes relative to clean input, with negative and inconclusive effects retained. These descriptive estimates are not significance tests.

The technical checks support preparing a separate full-validation evaluation with the same checkpoints, seeds, feature definitions and transformation levels. Keep the pilot frozen; do not reuse its output directories or expand sample limits without an explicit new study configuration.

Limitations:
- Exploratory validation pilot; no causal or independent test confirmation.
- Three training seeds; perturbation replicates averaged within each seed, not counted as independent training runs.
- Time warping preserves channel event totals but changes duration, density and temporal structure together.
- Deletion changes rate, sparsity and temporal statistics together.
- Zero transforms were checked on inputs; clean saved predictions serve as the identity condition. No transformed identity inference was rerun.
- Full-validation perturbation evaluation is not implemented or completed by this review.
