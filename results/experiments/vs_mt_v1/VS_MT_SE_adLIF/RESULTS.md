# VS_MT_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | pending | pending | pending | 0 | pending |
| SSC | 80.43% | pending | pending | 1 | 80.43% (n=1; SD unavailable) |
| ECG | pending | pending | pending | 0 | pending |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 0 | pending | pending | pending | pending | pending |
| SSC | 1 | 1,694,230 (n=1) | 1,692,755 (n=1) | 47,945 (n=1) | 21,730 (n=1) / 26,216 (n=1) | 10.06 (n=1) / 12.14 (n=1) |
| ECG | 0 | pending | pending | pending | pending | pending |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 0 | pending | pending | pending | pending | pending |
| SSC | 1 | 57.10 (n=1) | 1.008720 (n=1) | 1.023918 (n=1) | 1.025221 (n=1) | 1.769 (n=1) |
| ECG | 0 | pending | pending | pending | pending | pending |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->
