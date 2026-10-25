# MT_TC_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.74% | 92.84% | 93.82% | 3 | 93.80% ± 0.95% |
| SSC | 80.12% | 80.05% | pending | 2 | 80.08% ± 0.05% |
| ECG | 87.78% | pending | pending | 1 | 87.78% (n=1; SD unavailable) |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,200 ± 0 | 451,460 ± 0 | 10,628 ± 514 | 8,488 ± 468 / 2,140 ± 301 | 9.43 ± 0.52 / 2.38 ± 0.33 |
| SSC | 2 | 1,692,790 ± 0 | 1,691,315 ± 0 | 47,125 ± 35 | 20,701 ± 900 / 26,424 ± 865 | 9.58 ± 0.42 / 12.23 ± 0.40 |
| ECG | 1 | 4,836 (n=1) | 4,758 (n=1) | 28,498 (n=1) | 12,179 (n=1) / 16,319 (n=1) | 26.00 (n=1) / 34.84 (n=1) |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.88 ± 0.34 | 0.242895 ± 0.000000 | 0.249006 ± 0.000000 | 0.249597 ± 0.000000 | 1.544 ± 0.077 |
| SSC | 2 | 55.78 ± 0.65 | 1.008720 ± 0.000000 | 1.022190 ± 0.000000 | 1.023493 ± 0.000000 | 1.845 ± 0.154 |
| ECG | 1 | 1.56 (n=1) | 0.011053 (n=1) | 0.014033 (n=1) | 0.014322 (n=1) | 45.783 (n=1) |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

