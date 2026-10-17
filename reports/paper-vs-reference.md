# Paper versus local reference

Snapshot: 17 October 2026. Final test accuracy (%); all selected paper rows are two-layer SE-adLIF. Local mean and sample SD use training seeds 42, 123 and 456. Differences are local minus paper mean in percentage points (pp), computed from unrounded local metrics; paper means are published rounded values.

| Dataset | Paper mean ± SD | Reference mean ± sample SD | Difference (pp) | Baseline compatibility |
| --- | ---: | ---: | ---: | --- |
| SHD* | 93.79 ± 0.76 | 93.15 ± 0.42 | -0.64 | Matched saved settings |
| SSC | 80.44 ± 0.26 | 78.21 ± 0.11 | -2.23 | Matched saved settings |
| ECG | 88.18 ± 0.36 | 88.32 ± 0.35 | +0.14 | Unmatched budget / stopping; descriptive only |

| Dataset | Seed 42 | Seed 123 | Seed 456 | Evidence |
|---|---:|---:|---:|---|
| SHD* | 92.93 | 92.89 | 93.64 | [Canonical results](../results/valid/reference/SHD/REFERENCE_RESULTS.json) |
| SSC | 78.33 | 78.12 | 78.17 | [Seed 42 metrics](../results/valid/reference/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv) / [Seed 123 metrics](../results/valid/reference/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv) / [Seed 456 metrics](../results/valid/reference/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | 88.13 | 88.72 | 88.11 | [Seed 42 metrics](../results/valid/reference/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv) / [Seed 123 metrics](../results/valid/reference/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv) / [Seed 456 metrics](../results/valid/reference/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |

## Local reference activity and computational cost

| Dataset | Estimated dense GFLOPs / validation sample, local reference | Average padded validation timesteps/sample | Evidence |
|---|---:|---:|---|
| SHD* | 0.2619 | 293.31 | [Reference summary](../results/valid/reference/SHD/seed_42/TRAINING_SUMMARY.md) |
| SSC | 1.0087 | 300.00 | [Reference summary](../results/valid/reference/SSC/seed_42/TRAINING_SUMMARY.md) |
| ECG | 0.0111 | 1301.00 | [Reference summary](../results/valid/reference/ECG/seed_42/TRAINING_SUMMARY.md) |

These are local-reference estimates, not published paper values; the paper does not report dense GFLOPs for these datasets. Each estimate agrees across the three saved seed summaries. They count feed-forward and recurrent matrix multiply-adds as two FLOPs each, using padded validation lengths. Neuron-state updates, loss computation, backward propagation and data loading are excluded. The SHD* test-shape estimate (0.2429 GFLOPs/sample) in the seven-model report uses different sequence lengths and is labeled separately. The ECG budget/stopping mismatch remains applicable to its accuracy results.

| Dataset | Total parameters | Trainable parameters | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |
|---|---:|---:|---|---|---|---|---|---|
| SHD* | 450,760 | 450,020 | 10,366 ± 726 | 8,434 ± 688 / 1,932 ± 120 | 9.37 ± 0.76 / 2.14 ± 0.13 | 9.76 ± 0.51 | 5.52 ± 0.00 | 1.241 ± 0.011 |
| SSC | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| ECG | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

SHD* activity and timing are measured from all three saved reference checkpoints; SSC/ECG measurements remain pending. Activity, operation and timing values will be reported as mean +/- sample SD across matching seeds. The ECG protocol mismatch described above still applies. These are local-reference measurements; do not attribute them to the paper. Published parameter counts remain in [paper-results.md](paper-results.md); paper spike/cost values have not been extracted for this comparison. The measurement definitions and limitations are in the [seven-model report](reference-vs-models.md#spike-activity-and-inference-cost).

| Qualification | Meaning |
|---|---|
| Shared authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml); matching saved settings do not establish exact paper reproduction. |
| SHD* | Canonical reference: batch 256, 300 epochs, early stopping disabled; 20% training-derived validation, split seed 42. Retired dated SHD reference records are excluded. |
| SSC | Retained reference settings match the SSC baseline section. |
| ECG | Seed 42 used a 383-epoch budget instead of 400; seeds 123/456 enabled early stopping. The retained three-seed mean is descriptive, not a fully matched baseline result. |
| Uncertainty | The local and paper results use different sample sizes. A positive mean difference alone does not demonstrate superiority. |

[Paper source table](paper-results.md) · [Reference versus proposed models](reference-vs-models.md)
