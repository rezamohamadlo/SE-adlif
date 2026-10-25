# AMT_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.48% | 90.64% | 93.02% | 3 | 92.71% ± 1.94% |
| SSC | 80.24% | 80.45% | 80.08% | 3 | 80.26% ± 0.18% |
| ECG | 87.65% | 87.82% | 88.02% | 3 | 87.83% ± 0.18% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,920 ± 0 | 452,180 ± 0 | 10,740 ± 376 | 8,652 ± 98 / 2,088 ± 448 | 9.61 ± 0.11 / 2.32 ± 0.50 |
| SSC | 3 | 1,694,230 ± 0 | 1,692,755 ± 0 | 49,383 ± 2,491 | 21,880 ± 1,659 / 27,503 ± 870 | 10.13 ± 0.77 / 12.73 ± 0.40 |
| ECG | 3 | 4,908 ± 0 | 4,830 ± 0 | 27,536 ± 387 | 10,657 ± 524 / 16,879 ± 366 | 22.75 ± 1.12 / 36.04 ± 0.78 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.98 ± 0.12 | 0.242895 ± 0.000000 | 0.250769 ± 0.000000 | 0.251555 ± 0.000000 | 1.845 ± 0.121 |
| SSC | 3 | 58.29 ± 3.00 | 1.008720 ± 0.000000 | 1.026077 ± 0.000000 | 1.027811 ± 0.000000 | 2.020 ± 0.112 |
| ECG | 3 | 1.48 ± 0.03 | 0.011053 ± 0.000000 | 0.014879 ± 0.000000 | 0.015260 ± 0.000000 | 45.960 ± 1.061 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

