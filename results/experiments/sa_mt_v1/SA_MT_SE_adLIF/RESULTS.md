# SA_MT_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.17% | 93.15% | 93.86% | 3 | 93.73% ± 0.52% |
| SSC | 80.13% | 80.32% | 79.84% | 3 | 80.10% ± 0.25% |
| ECG | 88.26% | 88.65% | 88.27% | 3 | 88.39% ± 0.22% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,920 ± 0 | 452,180 ± 0 | 10,712 ± 588 | 8,374 ± 425 / 2,339 ± 179 | 9.30 ± 0.47 / 2.60 ± 0.20 |
| SSC | 3 | 1,694,230 ± 0 | 1,692,755 ± 0 | 49,151 ± 1,644 | 21,970 ± 1,585 / 27,181 ± 337 | 10.17 ± 0.73 / 12.58 ± 0.16 |
| ECG | 3 | 4,908 ± 0 | 4,830 ± 0 | 27,167 ± 950 | 10,451 ± 570 / 16,715 ± 1,021 | 22.31 ± 1.22 / 35.69 ± 2.18 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.87 ± 0.37 | 0.242895 ± 0.000000 | 0.249789 ± 0.000000 | 0.250380 ± 0.000000 | 1.588 ± 0.005 |
| SSC | 3 | 58.17 ± 2.31 | 1.008720 ± 0.000000 | 1.023918 ± 0.000000 | 1.025221 ± 0.000000 | 1.759 ± 0.016 |
| ECG | 3 | 1.45 ± 0.05 | 0.011053 ± 0.000000 | 0.014407 ± 0.000000 | 0.014696 ± 0.000000 | 42.426 ± 1.115 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->
