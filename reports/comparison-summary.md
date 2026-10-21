# Paper, reference, DTH, MT and MT-DTH comparison

## Test accuracy

Snapshot: 21 October 2026. Accuracy in percent; **bold** marks local means/values higher than the paper mean in the same row; the paper column is not bolded.

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.79 ± 0.76 | 93.15 ± 0.42 | **94.49 ± 0.52** | **94.61 ± 0.23** | 93.74 ± 0.17 |
| SSC | 80.44 ± 0.26 | 80.47 ± 0.20 | 79.95 ± 0.46 | 80.07 ± 0.14 | Not run |
| ECG | 88.18 ± 0.36 | **88.32 ± 0.35** | 87.63 ± 0.31 | 87.81 ± 0.49 | Not run |

All local entries are mean +/- sample SD across training seeds 42, 123 and 456 (n=3). SSC uses ssc_mean_loss_v1. Paper entries retain published mean/SD: SHD* and ECG n=20, SSC n=5. Different sample sizes mean bold values do not establish statistical significance.

SSC test accuracy includes the newly imported seeds 123 and 456. The following per-seed values produce the SSC means above; aggregates are calculated from unrounded metrics.

| SSC model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Mean +/- sample SD (%) |
|---|---:|---:|---:|---:|
| Reference | 80.69 | 80.33 | 80.38 | 80.47 +/- 0.20 |
| DTH | 80.46 | 79.84 | 79.56 | 79.95 +/- 0.46 |
| MT | 80.01 | 79.98 | 80.23 | 80.07 +/- 0.14 |

Sources: [SSC per-seed results](../results/valid/SSC_MEAN_COMPARISON.md) and [reference aggregate](../results/valid/reference/SSC/REFERENCE_RESULTS.json).

ECG reference results are accepted historical evidence with unmatched stopping/budget: 383 epochs for seed 42 and early stopping at 283/185 epochs for seeds 123/456, versus 400 epochs for DTH/MT. The ECG row is descriptive, not a fully controlled comparison.

Sources: [published paper results](paper-results.md), [paper versus reference](paper-vs-reference.md), [reference versus models](reference-vs-models.md), and [per-seed evidence](../results/valid/COMPARISON_STATUS.md).

MT-DTH SHD* is a matched three-seed follow-up combination experiment, selected after earlier model results were inspected. It did not improve on DTH or MT. See its [per-model records](../results/not_improved/MT_DTH_SE_adLIF/RESULTS.md). SSC/ECG have not been run.

## Inference time

Snapshot: 21 October 2026. Model forward time in milliseconds per sample; lower is faster.

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 1.24 ± 0.01 | 1.61 ± 0.04 | 1.67 ± 0.03 | 1.99 ± 0.11 |
| SSC | Not reported | 1.41 ± 0.07 | 1.74 ± 0.16 | 1.70 ± 0.14 | Not run |
| ECG | Not reported | 33.87 ± 4.54 | 43.92 ± 0.31 | 44.67 ± 0.53 | Not run |

All entries are mean ± sample SD across training seeds 42, 123 and 456 (n=3). Each seed value is the saved forward-time measurement, using three timing repeats and two warm-up batches. Across-seed SD also reflects measurement/session variation.

These are batch-amortized forward times on the official test data, excluding loading, transfer, loss and prediction aggregation; they are not single-request or end-to-end latency. Saved settings use an NVIDIA GeForce RTX 4090, PyTorch 2.6.0+cu124, CUDA 12.4 and high matmul precision. Batch sizes are 256 for SHD*/SSC and 64 for ECG. Dataset fingerprints match across measured models within each dataset.

Measurements come from separate sessions and evaluator versions. Python is 3.11.0 except ECG DTH seeds 42/123 (3.11.9); matching GPU model does not establish identical machine load or clock conditions. Treat differences as descriptive measurements rather than a controlled hardware benchmark. ECG reference timing varies substantially across seeds (38.64, 33.38 and 29.60 ms/sample); its sample SD is 4.54 ms/sample.

SSC timings use the current mean-loss checkpoints, including the verified seed-42 reference; archived summed-loss measurements are excluded. Paper timing is unavailable in the project comparison records, so no values are bolded as improvements over the paper. MT-DTH has only SHD* results. ECG reference retains the historical stopping/budget differences described in the [test-accuracy table](#test-accuracy).

DTH+TTA and MT+TTA inference times have not been measured. The values above are ordinary forward inference; TTA latency cannot be obtained from its accuracy results or assumed to be exactly three times these values.

### Per-seed measurements and sources

Values are in seed order 42, 123, 456 for every dataset. Links point to the underlying saved cost records, including checkpoint identities and measurement settings.

| Dataset | Model | Per-seed ms/sample | Cost records by seed |
|---|---|---:|---|
| SHD* | Reference | 1.23 / 1.24 / 1.25 | [42](../results/valid/reference/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/reference/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/reference/SHD/seed_456/COST_RESULTS.json) |
| SHD* | DTH | 1.66 / 1.60 / 1.58 | [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | MT | 1.64 / 1.68 / 1.68 | [42](../results/valid/MT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | MT-DTH | 1.91 / 1.94 / 2.12 | [42](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SSC | Reference | 1.32 / 1.46 / 1.44 | [42](../results/valid/reference/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/reference/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/reference/SSC/seed_456/COST_RESULTS.json) |
| SSC | DTH | 1.61 / 1.92 / 1.70 | [42](../results/valid/DTH_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | MT | 1.56 / 1.85 / 1.69 | [42](../results/valid/MT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| ECG | Reference | 38.64 / 33.38 / 29.60 | [42](../results/valid/reference/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/reference/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/reference/ECG/seed_456/COST_RESULTS.json) |
| ECG | DTH | 43.81 / 43.67 / 44.26 | [42](../results/valid/DTH_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | MT | 45.26 / 44.52 / 44.23 | [42](../results/valid/MT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |

## Parameters, activity and computational cost

The tables below use the same saved cost records linked in [per-seed measurements and sources](#per-seed-measurements-and-sources). Parameter counts are exact and agree across the available seeds. Activity and compute entries are mean ± sample SD across seeds 42, 123 and 456 for every dataset (n=3). SD describes across-seed variation, not uncertainty in an operation-counting formula.

Paper parameter counts retain the rounded values published in [Table 2](../reference_paper.pdf#page=11); the paper does not separate trainable parameters or report the classification activity/compute measures below. No values are bolded against unavailable or rounded paper cost baselines. Here, "Not reported" means no matching numeric result for the two-layer SHD*/SSC/ECG classification models in the supplied paper, including its appended supplementary material. The paper does discuss sparse activity and reports audio-bottleneck spike rates (Table 3, PDF page 15: SE-adLIF 5.2 kSp./s), which are not classification hidden-layer sparsity or spikes/sample. Rounded classification parameter counts come from Table 2 (PDF page 11); trainable versus total counts are not separately tabulated.

### Total parameters

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | 0.45M | 450,760 | 451,484 | 452,200 | 452,924 |
| SSC | 1.6M | 1,689,910 | 1,691,354 | 1,692,790 | Not run |
| ECG | 4.6k | 4,692 | 4,768 | 4,836 | Not run |

### Trainable parameters

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 450,020 | 450,744 | 451,460 | 452,184 |
| SSC | Not reported | 1,688,435 | 1,689,879 | 1,691,315 | Not run |
| ECG | Not reported | 4,614 | 4,690 | 4,758 | Not run |

### Hidden spikes per sample

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 10365.8 ± 726.0 | 10513.9 ± 612.6 | 10637.4 ± 608.6 | 10557.6 ± 438.8 |
| SSC | Not reported | 45301.2 ± 1640.2 | 45601.5 ± 4030.9 | 49500.0 ± 580.7 | Not run |
| ECG | Not reported | 26000.4 ± 2012.7 | 30057.7 ± 2822.3 | 26665.3 ± 378.0 | Not run |

### Event-driven synaptic-operation estimate (million SynOps/sample)

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 9.76 ± 0.51 | 9.78 ± 0.31 | 9.86 ± 0.42 | 9.87 ± 0.38 |
| SSC | Not reported | 53.34 ± 1.69 | 53.51 ± 4.73 | 58.35 ± 0.32 | Not run |
| ECG | Not reported | 1.40 ± 0.13 | 1.61 ± 0.17 | 1.42 ± 0.04 | Not run |

### Hidden spike sparsity (%)

Sparsity is 100 minus the recorded firing percentage: the percentage of neuron-timestep positions without a spike. Each cell reports **L1 / L2**, using the stored sequence extent including minimum padding and burn-in. For each seed and layer, sparsity = 100 × (1 - total hidden spikes / (stored sample-timesteps × neurons in that layer)). Added padding to form a batch is excluded; configured minimum padding and burn-in are included. This measures spike sparsity, not zero-valued weights or input sparsity. It does not imply that the dense GPU implementation skips the corresponding operations. All datasets show mean ± sample SD across three seeds.

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 90.63 ± 0.76 / 97.86 ± 0.13 | 90.76 ± 0.26 / 97.56 ± 0.48 | 90.67 ± 0.62 / 97.52 ± 0.10 | 90.53 ± 0.72 / 97.75 ± 0.32 |
| SSC | Not reported | 91.14 ± 0.32 / 87.88 ± 0.47 | 91.18 ± 1.14 / 87.71 ± 0.72 | 89.89 ± 0.12 / 87.20 ± 0.37 | Not run |
| ECG | Not reported | 78.30 ± 2.92 / 66.19 ± 1.38 | 75.17 ± 3.99 / 60.66 ± 2.86 | 78.77 ± 2.62 / 64.30 ± 2.49 | Not run |

![Reference hidden-layer spike sparsity by dataset](figures/reference-layer-sparsity.png)

[Download vector figure](figures/reference-layer-sparsity.svg). Reproduce from saved measurements with `.\.ven\Scripts\python.exe scripts/plot_reference_sparsity.py`; no inference or training is performed.

### Dense matrix cost (GFLOPs/sample)

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 0.24 ± 0.00 | 0.24 ± 0.00 | 0.24 ± 0.00 | 0.24 ± 0.00 |
| SSC | Not reported | 1.01 ± 0.00 | 1.01 ± 0.00 | 1.01 ± 0.00 | Not run |
| ECG | Not reported | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | Not run |

### Matrix plus scalar arithmetic cost (GFLOPs/sample)

| Dataset | Paper | Reference | DTH | MT | MT-DTH |
|---|---:|---:|---:|---:|---:|
| SHD* | Not reported | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 |
| SSC | Not reported | 1.02 ± 0.00 | 1.03 ± 0.00 | 1.02 ± 0.00 | Not run |
| ECG | Not reported | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | Not run |

Hidden spikes sum the two hidden layers over the stored sequence, including configured minimum padding and burn-in; output-layer activity is excluded. Counts depend on input duration and should be compared within each dataset.

SynOps estimate ideal event-driven dense fan-out additions from input events and hidden spikes; they do not measure operations actually executed by the dense GPU implementation. ECG analog-input projection is excluded from SynOps and counted separately as dense MACs in the source records.

Dense matrix cost counts each matrix multiply-add as two FLOPs. The arithmetic table includes this matrix cost plus counted scalar arithmetic in the neuron forward pass; it excludes nonlinear functions and comparisons. Both use the executed batch-padded sequence extent. They exclude loading, transfer, memory traffic, loss and temporal prediction aggregation. These are computational estimates, not measured energy, monetary cost or full end-to-end FLOPs.

All tables describe ordinary inference without TTA. The historical ECG reference stopping/budget caveat in the accuracy section also applies to the measured checkpoints.

SSC seeds 123/456 were imported and hash-verified on 21 October 2026; [transfer provenance](../results/provenance/ssc-local-import-2026-10-21.json). Saved measurements were aggregated without rerunning inference.
