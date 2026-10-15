# Paper versus local reference

Snapshot: 15 October 2026. Final test accuracy (%); all selected paper rows are two-layer SE-adLIF. Local mean and sample SD use training seeds 42, 123 and 456. Differences are local minus paper mean in percentage points (pp), computed from unrounded local metrics; paper means are published rounded values.

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

| Qualification | Meaning |
|---|---|
| Shared authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml); matching saved settings do not establish exact paper reproduction. |
| SHD* | Canonical reference: batch 256, 300 epochs, early stopping disabled; 20% training-derived validation, split seed 42. Retired dated SHD reference records are excluded. |
| SSC | Retained reference settings match the SSC baseline section. |
| ECG | Seed 42 used a 383-epoch budget instead of 400; seeds 123/456 enabled early stopping. The retained three-seed mean is descriptive, not a fully matched baseline result. |
| Uncertainty | The local and paper results use different sample sizes. A positive mean difference alone does not demonstrate superiority. |

[Paper source table](paper-results.md) · [Reference versus proposed models](reference-vs-models.md)
