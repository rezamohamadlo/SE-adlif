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

## Current status - 14 October 2026

The controlled comparison protocol is implemented in `evaluation.yaml`,
`evaluation_protocol.py`, and `runner.py`. SHD uses 20% training-derived
validation with split seed 42; SSC uses official validation; ECG uses 5%
training-derived validation. Final test results follow validation checkpoint
selection. Architecture and neuron implementations remain separate.

Reference and proposed-model records are under `results/valid/reference/`
and `results/valid/protocol_v2/`. Current comparisons:

- [Seed-42 model comparison](reports/seed-42-model-comparison.md).
- [Paper versus reference](reports/paper-vs-reference.md).
- [Shared evaluation commands](reports/shared-evaluation-config.md).

Historical test-selected SHD reproduction statistics are retained in the
[Phase 1 report](reports/phase-1-se-adlif-reference.md); they are not current
unbiased test estimates. The old `results/phase1_models` and development
archive paths are no longer present.

Reference tuning is separate from controlled model comparisons:

- `tune_shd.py`: checkpoint continuation screening and fresh confirmation.
- `search_shd.py`: nine fresh LR/batch runs.
- `queue_shd_dropout.py`: optional dropout follow-up queue.

Their results remain under `results/experiments/`. A fine-tuning test gain
must not be interpreted as a better training-from-scratch configuration.

## Phase 1 - Reference reproduction

Historical three-seed reproduction is documented. Current training-derived
validation reference records are retained separately. Verify matched settings
and checkpoint provenance before aggregating seeds.

## Phase 2 - Comparison protocol

Implemented. Preserve the shared splits, preprocessing, seeds, budgets,
checkpoint selection and final-test rules for fair model comparisons.
Remaining work includes multi-seed confirmation of proposed models and
measured efficiency rather than inferred energy savings.

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
