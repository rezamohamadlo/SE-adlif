# MT_DTH_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.82% | 93.55% | 93.86% | 3 | 93.74% ± 0.17% |
| SSC | pending | pending | pending | 0 | pending |
| ECG | pending | pending | pending | 0 | pending |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,924 ± 0 | 452,184 ± 0 | 10,558 ± 439 | 8,528 ± 644 / 2,029 ± 284 | 9.47 ± 0.72 / 2.25 ± 0.32 |
| SSC | 0 | pending | pending | pending | pending | pending |
| ECG | 0 | pending | pending | pending | pending | pending |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.87 ± 0.38 | 0.242895 ± 0.000000 | 0.251553 ± 0.000000 | 0.252536 ± 0.000000 | 1.991 ± 0.113 |
| SSC | 0 | pending | pending | pending | pending | pending |
| ECG | 0 | pending | pending | pending | pending | pending |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->


## Comparison and provenance

MT-DTH combines MT adaptation currents with DTH thresholds in a separate implementation. All three SHD* runs completed the shared 300-epoch baseline. The mean test accuracy is **93.74 +/- 0.17%**, compared with MT **94.61 +/- 0.23%** and DTH **94.49 +/- 0.52%**. It scored lower on every matching seed and increases the counted forward arithmetic. Treat this as a negative combination ablation.

This follow-up architecture was proposed after inspecting the earlier test results. Checkpoints were selected by validation; no test-based checkpoint selection or hyperparameter sweep was performed here. Per-seed raw metrics, resolved manifests, selected-checkpoint hashes and cost records are retained. See the [full comparison](../../../reports/reference-vs-models.md).
