# Paper versus local reference

Snapshot: 21 October 2026. Reference test accuracy is reported separately by dataset. All three datasets use three local training seeds (42, 123, 456). Δ paper = local minus the published rounded paper mean, in percentage points (pp), using unrounded local metrics; differences appear in parentheses after local accuracy. Per-seed deltas compare each run with the paper mean; aggregate deltas compare means. The three ECG reference runs are accepted historical evidence under the reporting agreement, with unmatched stopping/budget disclosed.

| Dataset | Reference mean ± sample SD (Δ paper) | Paper mean ± SD | Baseline compatibility |
|---|---:|---:|---|
| SHD* | 93.15 ± 0.42 (-0.64 pp) | 93.79 ± 0.76 | Matched saved settings |
| SSC (mean loss) | 80.47 +/- 0.20 (n=3) (+0.03 pp) | 80.44 +/- 0.26 (n=5) | Released-code mean loss; original diagnostic provenance retained |
| ECG | 88.32 ± 0.35 (+0.14 pp) | 88.18 ± 0.36 | Accepted historical; unmatched budget/stopping, descriptive only |

| Dataset | Seed 42 (Δ paper) | Seed 123 (Δ paper) | Seed 456 (Δ paper) | Evidence |
|---|---:|---:|---:|---|
| SHD* | 92.93 (-0.86 pp) | 92.89 (-0.90 pp) | 93.64 (-0.15 pp) | [Canonical results](../results/valid/reference/SHD/REFERENCE_RESULTS.json) |
| SSC (mean loss) | 80.69 (+0.25 pp) | 80.33 (-0.11 pp) | 80.38 (-0.06 pp) | [Three-seed reference aggregate](../results/valid/reference/SSC/REFERENCE_RESULTS.json) |
| ECG | 88.13 (-0.05 pp) | 88.72 (+0.54 pp) | 88.11 (-0.07 pp) | [Seed 42 metrics](../results/valid/reference/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv) / [Seed 123 metrics](../results/valid/reference/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv) / [Seed 456 metrics](../results/valid/reference/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |

## Published paper versus measured local reference: spikes and dense computation

| Dataset | Local reference hidden spikes/sample, mean and sample SD | Paper hidden spikes/sample | Local reference dense GFLOPs/test sample | Paper dense GFLOPs/sample |
|---|---:|---|---:|---|
| SHD* | 10,366 ± 726 | Not reported | 0.242895 | Not reported |
| SSC (mean loss) | 45,301 +/- 1,640 | Not reported | 1.008720 +/- 0.000000 | Not reported |
| ECG | 26,000 ± 2,013 | Not reported | 0.011053 | Not reported |

Not reported means the classification metric is absent from the paper; it does not mean zero. Local spike measurements sum L1 and L2 across each stored sequence, including preprocessing minimum padding and burn-in, excluding additional batch padding. Dense GFLOPs use the actual batch-padded official test lengths and count matrix operations only. Local results use three training seeds; the ECG training-setting mismatch remains. The paper's audio-compression spike rates are not used as classification values. Broader arithmetic/logical estimates and measurement evidence appear below.

## Local reference activity and computational cost

| Dataset | Estimated dense GFLOPs / validation sample, local reference | Average padded validation timesteps/sample | Evidence |
|---|---:|---:|---|
| SHD* | 0.2619 | 293.31 | [Reference summary](../results/valid/reference/SHD/seed_42/TRAINING_SUMMARY.md) |
| SSC (mean loss) | 1.0087 | 300.00 | [Reference summary](../results/valid/reference/SSC/seed_42/TRAINING_SUMMARY.md) |
| ECG | 0.0111 | 1301.00 | [Reference summary](../results/valid/reference/ECG/seed_42/TRAINING_SUMMARY.md) |

These are local-reference estimates, not published paper values; the paper does not report dense GFLOPs for these datasets. Each estimate agrees across the three saved seed summaries. They count feed-forward and recurrent matrix multiply-adds as two FLOPs each, using padded validation lengths. Neuron-state updates, loss computation, backward propagation and data loading are excluded. The SHD* test-shape estimate (0.2429 GFLOPs/sample) in the seven-model report uses different sequence lengths and is labeled separately. The ECG budget/stopping mismatch remains applicable to its accuracy results.

| Dataset | Total parameters | Trainable parameters | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |
|---|---:|---:|---|---|---|---|---|---|
| SHD* | 450,760 | 450,020 | 10,366 ± 726 | 8,434 ± 688 / 1,932 ± 120 | 9.37 ± 0.76 / 2.14 ± 0.13 | 9.76 ± 0.51 | 5.52 ± 0.00 | 1.241 ± 0.011 |
| SSC (mean loss) | 1,689,910 | 1,688,435 | 45,301 +/- 1,640 | 19,127 +/- 695 / 26,174 +/- 1,015 | 8.86 +/- 0.32 / 12.12 +/- 0.47 | 53.34 +/- 1.69 | 12.16 +/- 0.00 | 1.406 +/- 0.071 |
| ECG | 4692 | 4614 | 26,000 ± 2,013 | 10,164 ± 1,367 / 15,836 ± 646 | 21.70 ± 2.92 / 33.81 ± 1.38 | 1.40 ± 0.13 | 2.69 ± 0.00 | 33.874 ± 4.540 |

Activity and timing are measured on the official test sets from all three saved reference checkpoints per dataset. Activity, operation and timing values are reported as mean +/- sample SD across matching seeds. The ECG protocol mismatch described above still applies. These are local-reference measurements; do not attribute them to the paper. Published parameter counts remain in [paper-results.md](paper-results.md); paper spike/cost values have not been extracted for this comparison. The measurement definitions and limitations are in the [seven-model report](reference-vs-models.md#spike-activity-and-inference-cost).

| Qualification | Meaning |
|---|---|
| Shared authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml); matching saved settings do not establish exact paper reproduction. |
| SHD* | Canonical reference: batch 256, 300 epochs, early stopping disabled; 20% training-derived validation, split seed 42. Retired dated SHD reference records are excluded. |
| SSC (mean loss) | Current mean-loss measurements use the same validation-selected checkpoints as the accuracy table; seed-42 diagnostic provenance is retained. |
| ECG | Accepted historical evidence: seed 42 completed 383 epochs; seeds 123/456 enabled early stopping and completed 283/185 epochs. Descriptive comparisons only. |
| Uncertainty | The local and paper results use different sample sizes. A positive mean difference alone does not demonstrate superiority. |

[Paper source table](paper-results.md) · [Reference versus proposed models](reference-vs-models.md)

## Test-set forward computation

| Dataset | Test sequences | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Matrix + all logical scalar GOp/sample |
|---|---:|---:|---:|---:|
| SHD* | 2,264 | 0.242895 ± 0.000000 | 0.247824 ± 0.000000 | 0.248414 ± 0.000000 |
| SSC (mean loss) | 20,382 | 1.008720 +/- 0.000000 | 1.019584 +/- 0.000000 | 1.020884 +/- 0.000000 |
| ECG | 141 | 0.011053 ± 0.000000 | 0.013455 ± 0.000000 | 0.013740 ± 0.000000 |

Entries are mean ± sample SD across three seeds, using actual batch-padded test lengths. The arithmetic estimate adds counted add/subtract/multiply/divide, negation, reciprocal, sum/mean reduction and bias operations in the cell forward code to the matrix estimate. Nonlinear functions (e.g. exp/sigmoid) and comparisons are excluded from that arithmetic GFLOP figure; they are included as one logical operation each in the last column. Memory traffic is excluded from all columns. That logical total is a counting convention, not hardware FLOPs or energy. Loss and temporal prediction aggregation are excluded from both estimates. Thus neither column should be called an exact full end-to-end inference FLOP count.

SSC inputs are framed event counts. ECG inputs are analog: its first projection is counted as dense MACs, not spike events. The ECG SynOps activity-table cell therefore covers hidden-driven connections only; its analog first-projection MACs are stored separately. All hidden spikes count stored pre-batch sequence lengths (including configured minimum padding and burn-in); padded totals are retained separately. ECG accuracy is per labeled timestep, while the compute denominator is one input sequence. Dataset-specific batch sizes and lengths mean cross-dataset runtimes are not matched-workload comparisons.

SSC/ECG checkpoint hashes, resolved configs, per-seed counts, timing repeats and reproducible command: [measurement records](../results/valid/MEASUREMENTS.md). Existing validation estimates remain separately labeled above. The ECG training-budget/stopping mismatch is unchanged. No checkpoint was retrained or selected using these test results.

Accuracy tables were checked against canonical per-seed metrics on 18 October 2026; no numerical corrections were needed. See the [local evidence audit](../docs/audits/results-audit-2026-10-18.md) for artifact availability and the separate proposed-model SSC/ECG limitations.

The [SSC reproduction fidelity audit](../docs/audits/ssc-reproduction-audit-2026-10-18.md) found a paper/released-code disagreement: the archived summed-loss reference uses temporal summation, while the pinned authors' SSC config uses a temporal mean. This is a confirmed objective difference, not a demonstrated explanation of the accuracy gap. The dated audit also records a NumPy/Tonic dependency conflict at that snapshot; it does not infer the historical training environment from today's installation.

The [SHD* fidelity investigation](../docs/audits/shd-reproduction-audit-2026-10-18.md) found no confirmed implementation error explaining its -0.64 pp gap: loss semantics match released code, six checkpoints confirm the intended settings, selected checkpoint hashes match, and 12 sampled train/validation cache entries agree with fresh preprocessing. Author-run provenance, execution details and the three-versus-twenty-seed comparison remain limitations.

## Active SSC mean-loss reference - three seeds

| Model | Seed 42 test (%, Δ paper) | Seed 123 test (%, Δ paper) | Seed 456 test (%, Δ paper) | n | Mean +/- sample SD (%, Δ paper) |
|---|---:|---:|---:|---:|---:|
| Reference | 80.69 (+0.25 pp) | 80.33 (-0.11 pp) | 80.38 (-0.06 pp) | 3 | 80.47 +/- 0.20 (+0.03 pp) |
| DTH | 80.46 (+0.02 pp) | 79.84 (-0.60 pp) | 79.56 (-0.88 pp) | 3 | 79.95 +/- 0.46 (-0.49 pp) |
| MT | 80.01 (-0.43 pp) | 79.98 (-0.46 pp) | 80.23 (-0.21 pp) | 3 | 80.07 +/- 0.14 (-0.37 pp) |

All nine runs completed 40 epochs with batch 256, LR 0.006, mean membrane-potential loss and no early stopping. Checkpoints were selected by validation accuracy. Reference seed 42 retains its diagnostic provenance; the comparison was planned after its test result was observed. Means and sample SD use independent training seeds 42, 123 and 456. No archived summed-loss results are included.

[Canonical SSC records](../results/valid/reference/SSC/RESULTS.md). All three mean-loss cost records are available; historical summed-loss measurements above remain separate.

## Retained historical SSC accuracy

The archived summed-loss reference was 78.21 +/- 0.11% (seeds 42/123/456: 78.33/78.12/78.17%). Its difference from the paper's rounded mean was -2.23 pp. Current computation/activity tables describe mean-loss checkpoints; archived summed-loss measurements remain separate; mean-loss reference cost measurements are available in the seed directories. No archived measurement was reassigned to the new checkpoint.

The current reference contributes 9/9 accepted results to the [27-run grid](../results/valid/COMPARISON_STATUS.md): six matched plus three accepted historical ECG results. The project total is 27/27 accepted, including 24 matched. See the [ECG acceptance record](../results/valid/reference/ECG/RESULTS.md).
