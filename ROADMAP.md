# DA-LIF Project Roadmap

## Project objective

Develop a dual-adaptive spiking neuron whose firing threshold and time constant
respond to network activity. The proposed model should preserve or improve
classification accuracy while reducing spike activity and computational cost
relative to SE-adLIF.

```text
SE-adLIF reference -> Dynamic Threshold -> Dynamic Time Constant
-> Dual-Adaptive model -> Accuracy and efficiency evaluation
```

## Current status — 7 October 2026

**Phase 1 is complete.** Canonical three-seed SE-adLIF reference results are
available for SHD, SSC, and ECG/QTDB.

| Dataset | Seed 42 | Seed 123 | Seed 456 | Mean ± sample SD | Status |
|---|---:|---:|---:|---:|---|
| SHD | 95.45% | 94.70% | 93.73% | **94.63% ± 0.86%** | Complete reference set |
| SSC | 78.33% | 78.12% | 78.17% | **78.21% ± 0.11%** | Complete final-test set |
| ECG/QTDB | 88.13% | 88.72% | 88.11% | **88.32% ± 0.35%** | Complete final-test set |

The detailed report is available at
[`reports/phase-1-se-adlif-reference.md`](reports/phase-1-se-adlif-reference.md).

### Canonical result locations

```text
results/phase1_models/reference/SHD/seed_{42,123,456}
results/phase1_models/reference/SSC/seed_{42,123,456}
results/phase1_models/reference/ECG/seed_{42,123,456}
```

Only these runs should be included in Phase 1 statistics. A repeated run of an
existing seed must be labelled as verification-only and must not be counted as
an additional independent seed.

### Interpretation of Phase 1 results

- SSC and ECG used validation data for checkpoint selection followed by final
  test evaluation. Their table values are final test accuracies.
- SHD used the test split for validation and checkpoint selection. Its result
  reproduces the current reference setup but is not an unbiased final-test
  estimate.
- The old SHD `loss_agg: softmax` mode computed
  `sum_t softmax(y_t)`, which is mathematically identical to the explicit
  `sum_softmax_over_time` mode. The canonical SHD results remain valid.
- Canonical SSC runs use `summed_membrane_potentials`. Earlier temporal-mean
  SSC runs are excluded from the reference result set.
- The SHD seed-42 training summary is stale; its authoritative `metrics.csv`
  and checkpoint record 95.45% at epoch 205.

## Phase 1 — Reproduce the SE-adLIF reference

**Status: COMPLETE**

Completed work:

- Prepared SHD, SSC, and ECG/QTDB datasets.
- Ran seeds `42`, `123`, and `456` for every dataset.
- Corrected and tested temporal loss aggregation.
- Saved checkpoints, metrics, hyperparameters, and per-seed summaries.
- Recorded parameter counts and dense-forward GFLOPs estimates.
- Calculated mean accuracy and sample standard deviation.
- Consolidated valid results under a single canonical `reference` hierarchy.

Remaining publication-quality improvement, not required to begin model
development:

- Introduce a separate SHD validation split and reserve the test split for one
  final evaluation.
- Standardize scheduler and early-stopping settings across every seed and model
  before the final controlled comparison.

## Phase 2 — Freeze the comparison protocol

**Status: NEXT**

Before evaluating a proposed architecture, define a single protocol that both
SE-adLIF and the proposed model will use:

- identical preprocessing and dataset splits;
- identical seeds: `42`, `123`, and `456`;
- identical batch size, epoch budget, optimizer, and learning-rate schedule;
- identical checkpoint-selection and final-test rules;
- identical accuracy, spike, SynOps, latency, memory, and parameter reporting;
- separate development and final-evaluation result directories.

During initial development, use SHD seed 42 only. Freeze the architecture and
hyperparameters before running seeds 123 and 456.

**Deliverable:** a reproducible runner and config set shared by the reference
and proposed models.

## Phase 3 — Dynamic Threshold model

Change only the firing-threshold mechanism while retaining the SE-adLIF time
constants and the rest of the architecture.

Tasks:

1. Define the recent-activity signal.
2. Define a bounded threshold adaptation rule.
3. Implement the neuron as a separate, selectable cell.
4. Add unit tests for shape, gradients, bounds, and deterministic behavior.
5. Train on SHD seed 42.
6. Compare accuracy, spike rate, gradient behavior, and stability with SE-adLIF.

**Deliverable:** Dynamic-Threshold model and a controlled SHD ablation.

## Phase 4 — Dynamic Time Constant model

Keep the threshold mechanism unchanged and adapt only the time constant.

Tasks:

1. Define the activity-to-time-constant rule.
2. Constrain the learned/adaptive time constant to a stable range.
3. Add gradient and numerical-stability tests.
4. Train on SHD seed 42 using the frozen protocol.
5. Measure temporal accuracy, spike rate, and computational cost.

**Deliverable:** Dynamic-Time-Constant model and an independent SHD ablation.

## Phase 5 — Dual-Adaptive model

Combine Dynamic Threshold and Dynamic Time Constant in the simplest stable
form. Avoid adding attention or auxiliary networks until the direct adaptation
rules have been evaluated.

Tasks:

1. Combine both mechanisms in a new DA-LIF cell.
2. Bound threshold and time-constant ranges.
3. Verify forward behavior and surrogate-gradient flow.
4. Train on SHD seed 42.
5. Compare against SE-adLIF and both single-mechanism ablations.

**Deliverable:** the primary DA-LIF model.

## Phase 6 — SHD ablation and model selection

Use a controlled comparison table:

| Model | Accuracy | Parameters | Spike rate | SynOps | GFLOPs | Latency |
|---|---:|---:|---:|---:|---:|---:|
| SE-adLIF | 94.63% ± 0.86% | 450,760 | — | — | 0.2510 | — |
| Dynamic Threshold | — | — | — | — | — | — |
| Dynamic Time Constant | — | — | — | — | — | — |
| DA-LIF | — | — | — | — | — | — |

Select the architecture using validation performance and efficiency, not test
accuracy. Freeze the selected architecture before multi-seed evaluation.

## Phase 7 — Three-seed evaluation

Run the frozen proposed architecture with seeds `42`, `123`, and `456`.

Report:

- per-seed validation and final-test accuracy;
- mean ± sample standard deviation;
- parameter count and GFLOPs;
- spike rate and SynOps;
- training time, inference latency, and peak GPU memory.

**Deliverable:** statistically comparable SHD results.

## Phase 8 — Generalization to SSC and ECG/QTDB

Apply the frozen model and comparison protocol in this order:

1. SSC for a harder temporal speech task.
2. ECG/QTDB for a non-speech temporal-signal task.

Do not tune on the test split. Any dataset-specific hyperparameter changes must
be documented and applied consistently to the SE-adLIF comparison.

**Deliverable:** three-dataset evidence for generalization.

## Phase 9 — Energy and efficiency evaluation

Measure:

- total spike count;
- spike rate per layer and sample;
- SynOps;
- dense-equivalent FLOPs/GFLOPs;
- inference latency;
- peak GPU memory;
- total and trainable parameters.

Accuracy alone is insufficient. The central claim should show whether adaptive
neurons reduce activity or computation without unacceptable accuracy loss.

## Phase 10 — Robustness and final reporting

Evaluate:

- reduced timestep budgets;
- dropped input events;
- temporal noise;
- threshold and time-constant trajectories;
- sensitivity to adaptation bounds and coefficients.

Prepare final tables, plots, ablations, method description, and limitations.

## Immediate next actions

1. Freeze the Phase 2 comparison protocol.
2. Create a feature branch for proposed-model development.
3. Specify the Dynamic Threshold equation and allowed range.
4. Implement the Dynamic Threshold neuron and unit tests.
5. Run the first controlled SHD seed-42 experiment.
6. Compare it with the canonical SE-adLIF reference using accuracy and spike
   activity.

## Definition of success

The project succeeds if DA-LIF provides comparable or better accuracy than
SE-adLIF while measurably reducing spike activity and SynOps under the same
evaluation protocol.
