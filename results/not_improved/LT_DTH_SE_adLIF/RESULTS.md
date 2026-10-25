# LT_DTH_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.66% | 93.42% | 93.86% | 3 | 93.98% ± 0.63% |
| SSC | 79.95% | 80.25% | 80.31% | 3 | 80.17% ± 0.19% |
| ECG | 88.33% | 87.11% | 87.58% | 3 | 87.67% ± 0.61% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 451,488 ± 0 | 450,748 ± 0 | 10,220 ± 503 | 8,447 ± 577 / 1,774 ± 75 | 9.38 ± 0.64 / 1.97 ± 0.08 |
| SSC | 3 | 1,691,358 ± 0 | 1,689,883 ± 0 | 43,981 ± 1,696 | 19,126 ± 765 / 24,855 ± 936 | 8.85 ± 0.35 / 11.51 ± 0.43 |
| ECG | 3 | 4,772 ± 0 | 4,694 ± 0 | 29,789 ± 1,456 | 11,891 ± 2,374 / 17,898 ± 921 | 25.39 ± 5.07 / 38.21 ± 1.97 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.71 ± 0.39 | 0.242895 ± 0.000000 | 0.250372 ± 0.000000 | 0.251353 ± 0.000000 | 1.788 ± 0.005 |
| SSC | 3 | 52.34 ± 1.80 | 1.008720 ± 0.000000 | 1.025202 ± 0.000000 | 1.027365 ± 0.000000 | 1.913 ± 0.014 |
| ECG | 3 | 1.61 ± 0.13 | 0.011053 ± 0.000000 | 0.014678 ± 0.000000 | 0.015151 ± 0.000000 | 45.911 ± 1.139 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

