# LG-DTH-SE-adLIF results

Final test accuracy from completed runs. SHD* uses 20% of the official training set for validation with split seed 42. Checkpoints are selected by validation accuracy before official test evaluation.

| Dataset | Seed 42 | Seed 123 | Seed 456 | Completed | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.77% | 93.02% | 93.51% | 3 | 93.43% ± 0.38% |
| SSC | pending | pending | pending | 0 | pending |
| ECG | pending | pending | pending | 0 | pending |

**The three-seed SHD* aggregate is descriptive: seed 456 enabled early stopping, while seeds 42 and 123 disabled it. It is not a fully matched three-seed paper-baseline result.**

| SHD* seed | Test accuracy | Test loss | Early stopping | Training epochs completed |
|---|---:|---:|---|---:|
| 42 | 93.77% | 0.570668 | Disabled | 300 |
| 123 | 93.02% | 0.637240 | Disabled | 300 |
| 456 | 93.51% | 0.403493 | Enabled | 105 |

| Settings / provenance | Record |
|---|---|
| Shared settings | Batch 256; initial LR 0.01; maximum 300 epochs; dropout 0.15; gradient clipping 1.5; scheduler patience 9999, factor 0.9 |
| Seeds 42 and 123 | Shared settings match `config/paper_baseline.yaml`, SHD section; original protocol metadata remains in their saved configs |
| Seed 456 | Protocol `paper_classification_v1_custom_af62d1bc3d42`; early-stopping patience 50, minimum delta 0.001; validation-selected test checkpoint `epoch=74-step=1875.ckpt` |
| Matched baseline subset | Seeds 42 and 123 only: 93.40% ± 0.53% sample SD (n=2) |
| Relocation | Completed seed 456 moved to `SHD/seed_456/`; see [relocation record](SHD/seed_456/RUN_RELOCATION.json). The older interrupted baseline attempt is preserved under `results/experiments/archived_runs/`. |

Sources: [seed 42 metrics](SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [seed 123 metrics](SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [seed 456 metrics](SHD/seed_456/logs/mlp_snn/version_0/metrics.csv). Statistics use unrounded test accuracies and sample SD across distinct training seeds. Completed epoch counts follow the saved training summaries.
