# Paper versus local reference

Snapshot: 20 October 2026. Reference test accuracy is reported separately by dataset. SHD* and ECG use three local seeds; SSC mean loss currently has seed 42 only (no across-seed SD). Differences are local minus published rounded paper means, using unrounded local metrics. The three ECG reference runs are accepted historical evidence under the reporting agreement, with unmatched stopping/budget disclosed.

| Dataset | Paper mean ± SD | Reference mean ± sample SD | Difference (pp) | Baseline compatibility |
| --- | ---: | ---: | ---: | --- |
| SHD* | 93.79 ± 0.76 | 93.15 ± 0.42 | -0.64 | Matched saved settings |
| SSC (mean loss) | 80.44 +/- 0.26 (n=5) | 80.69 (n=1; SD unavailable) | +0.25 | Released-code mean loss; original diagnostic provenance retained |
| ECG | 88.18 ± 0.36 | 88.32 ± 0.35 | +0.14 | Accepted historical; unmatched budget/stopping, descriptive only |

| Dataset | Seed 42 | Seed 123 | Seed 456 | Evidence |
|---|---:|---:|---:|---|
| SHD* | 92.93 | 92.89 | 93.64 | [Canonical results](../results/valid/reference/SHD/REFERENCE_RESULTS.json) |
| SSC (mean loss) | 80.69 | pending | pending | [Selected checkpoint result](../results/valid/reference/SSC/seed_42/REFERENCE_RESULT.json) |
| ECG | 88.13 | 88.72 | 88.11 | [Seed 42 metrics](../results/valid/reference/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv) / [Seed 123 metrics](../results/valid/reference/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv) / [Seed 456 metrics](../results/valid/reference/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |

## Published paper versus measured local reference: spikes and dense computation

| Dataset | Paper hidden spikes/sample | Local reference hidden spikes/sample, mean and sample SD | Paper dense GFLOPs/sample | Local reference dense GFLOPs/test sample |
|---|---|---:|---|---:|
| SHD* | Not reported | 10,366 ± 726 | Not reported | 0.242895 |
| SSC (archived sum loss) | Not reported | 9,038 ± 349 | Not reported | 1.008720 |
| ECG | Not reported | 26,000 ± 2,013 | Not reported | 0.011053 |

Not reported means the classification metric is absent from the paper; it does not mean zero. Local spike measurements sum L1 and L2 across each stored sequence, including preprocessing minimum padding and burn-in, excluding additional batch padding. Dense GFLOPs use the actual batch-padded official test lengths and count matrix operations only. Local results use three training seeds; the ECG training-setting mismatch remains. The paper's audio-compression spike rates are not used as classification values. Broader arithmetic/logical estimates and measurement evidence appear below.

## Local reference activity and computational cost

| Dataset | Estimated dense GFLOPs / validation sample, local reference | Average padded validation timesteps/sample | Evidence |
|---|---:|---:|---|
| SHD* | 0.2619 | 293.31 | [Reference summary](../results/valid/reference/SHD/seed_42/TRAINING_SUMMARY.md) |
| SSC (archived sum loss) | 1.0087 | 300.00 | [Reference summary](../archive/ssc_summed_loss_20261019/reference/SSC/seed_42/TRAINING_SUMMARY.md) |
| ECG | 0.0111 | 1301.00 | [Reference summary](../results/valid/reference/ECG/seed_42/TRAINING_SUMMARY.md) |

These are local-reference estimates, not published paper values; the paper does not report dense GFLOPs for these datasets. Each estimate agrees across the three saved seed summaries. They count feed-forward and recurrent matrix multiply-adds as two FLOPs each, using padded validation lengths. Neuron-state updates, loss computation, backward propagation and data loading are excluded. The SHD* test-shape estimate (0.2429 GFLOPs/sample) in the seven-model report uses different sequence lengths and is labeled separately. The ECG budget/stopping mismatch remains applicable to its accuracy results.

| Dataset | Total parameters | Trainable parameters | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |
|---|---:|---:|---|---|---|---|---|---|
| SHD* | 450,760 | 450,020 | 10,366 ± 726 | 8,434 ± 688 / 1,932 ± 120 | 9.37 ± 0.76 / 2.14 ± 0.13 | 9.76 ± 0.51 | 5.52 ± 0.00 | 1.241 ± 0.011 |
| SSC (archived sum loss) | 1689910 | 1688435 | 9,038 ± 349 | 8,654 ± 343 / 383 ± 36 | 4.01 ± 0.16 / 0.18 ± 0.02 | 18.83 ± 0.50 | 12.16 ± 0.00 | 1.368 ± 0.099 |
| ECG | 4692 | 4614 | 26,000 ± 2,013 | 10,164 ± 1,367 / 15,836 ± 646 | 21.70 ± 2.92 / 33.81 ± 1.38 | 1.40 ± 0.13 | 2.69 ± 0.00 | 33.874 ± 4.540 |

Activity and timing are measured on the official test sets from all three saved reference checkpoints per dataset. Activity, operation and timing values are reported as mean +/- sample SD across matching seeds. The ECG protocol mismatch described above still applies. These are local-reference measurements; do not attribute them to the paper. Published parameter counts remain in [paper-results.md](paper-results.md); paper spike/cost values have not been extracted for this comparison. The measurement definitions and limitations are in the [seven-model report](reference-vs-models.md#spike-activity-and-inference-cost).

| Qualification | Meaning |
|---|---|
| Shared authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml); matching saved settings do not establish exact paper reproduction. |
| SHD* | Canonical reference: batch 256, 300 epochs, early stopping disabled; 20% training-derived validation, split seed 42. Retired dated SHD reference records are excluded. |
| SSC (archived sum loss) | Historical summed-loss measurements only; excluded from current mean-loss accuracy evidence. |
| ECG | Accepted historical evidence: seed 42 completed 383 epochs; seeds 123/456 enabled early stopping and completed 283/185 epochs. Descriptive comparisons only. |
| Uncertainty | The local and paper results use different sample sizes. A positive mean difference alone does not demonstrate superiority. |

[Paper source table](paper-results.md) · [Reference versus proposed models](reference-vs-models.md)

## Test-set forward computation

| Dataset | Test sequences | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Matrix + all logical scalar GOp/sample |
|---|---:|---:|---:|---:|
| SHD* | 2,264 | 0.242895 ± 0.000000 | 0.247824 ± 0.000000 | 0.248414 ± 0.000000 |
| SSC (archived sum loss) | 20,382 | 1.008720 ± 0.000000 | 1.019584 ± 0.000000 | 1.020884 ± 0.000000 |
| ECG | 141 | 0.011053 ± 0.000000 | 0.013455 ± 0.000000 | 0.013740 ± 0.000000 |

Entries are mean ± sample SD across three seeds, using actual batch-padded test lengths. The arithmetic estimate adds counted add/subtract/multiply/divide, negation, reciprocal, sum/mean reduction and bias operations in the cell forward code to the matrix estimate. Nonlinear functions (e.g. exp/sigmoid) and comparisons are excluded from that arithmetic GFLOP figure; they are included as one logical operation each in the last column. Memory traffic is excluded from all columns. That logical total is a counting convention, not hardware FLOPs or energy. Loss and temporal prediction aggregation are excluded from both estimates. Thus neither column should be called an exact full end-to-end inference FLOP count.

SSC inputs are framed event counts. ECG inputs are analog: its first projection is counted as dense MACs, not spike events. The ECG SynOps activity-table cell therefore covers hidden-driven connections only; its analog first-projection MACs are stored separately. All hidden spikes count stored pre-batch sequence lengths (including configured minimum padding and burn-in); padded totals are retained separately. ECG accuracy is per labeled timestep, while the compute denominator is one input sequence. Dataset-specific batch sizes and lengths mean cross-dataset runtimes are not matched-workload comparisons.

SSC/ECG checkpoint hashes, resolved configs, per-seed counts, timing repeats and reproducible command: [measurement records](../results/valid/MEASUREMENTS.md). Existing validation estimates remain separately labeled above. The ECG training-budget/stopping mismatch is unchanged. No checkpoint was retrained or selected using these test results.

Accuracy tables were checked against canonical per-seed metrics on 18 October 2026; no numerical corrections were needed. See the [local evidence audit](../docs/audits/results-audit-2026-10-18.md) for artifact availability and the separate proposed-model SSC/ECG limitations.

The [SSC reproduction fidelity audit](../docs/audits/ssc-reproduction-audit-2026-10-18.md) found a paper/released-code disagreement: our saved reference uses temporal summation, while the pinned authors' SSC config uses a temporal mean. This is a confirmed objective difference, not a demonstrated explanation of the accuracy gap. The audit also records a current NumPy/Tonic dependency conflict; it does not infer the historical training environment from today's installation.

The [SHD* fidelity investigation](../docs/audits/shd-reproduction-audit-2026-10-18.md) found no confirmed implementation error explaining its -0.64 pp gap: loss semantics match released code, six checkpoints confirm the intended settings, selected checkpoint hashes match, and 12 sampled train/validation cache entries agree with fresh preprocessing. Author-run provenance, execution details and the three-versus-twenty-seed comparison remain limitations.

## Active SSC mean-loss reference - seed 42

| Protocol | Local test (%) | Completed seeds | Sample SD | Paper SSC mean +/- SD (%) | Local minus paper (pp) |
|---|---:|---:|---|---:|---:|
| ssc_mean_loss_v1 | 80.69 | 1 | unavailable | 80.44 +/- 0.26 (n=5) | +0.25 |

Validation-selected epoch 18 (82.22% validation). This reuses the tested diagnostic checkpoint, with original provenance preserved. One local seed versus five paper seeds is descriptive only; it is not a multi-seed confirmation. See the [canonical SSC record](../results/valid/reference/SSC/RESULTS.md). Historical summed-loss values above remain separate.

## Retained historical SSC accuracy

The archived summed-loss reference was 78.21 +/- 0.11% (seeds 42/123/456: 78.33/78.12/78.17%). Its difference from the paper's rounded mean was -2.23 pp. Computation/activity tables marked archived sum loss describe those original checkpoints; mean-loss reference cost measurements remain pending. No archived measurement was reassigned to the new checkpoint.

The current reference contributes 7/9 accepted results to the [27-run grid](../results/valid/COMPARISON_STATUS.md): four matched plus three accepted historical ECG results. The project total is 21/27 accepted, including 18 matched. See the [ECG acceptance record](../results/valid/reference/ECG/RESULTS.md).
