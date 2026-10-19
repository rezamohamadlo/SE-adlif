# DTH_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Target comparison protocol: paper_classification_v1; original per-run provenance is retained. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.70% | 94.88% | 93.90% | 3 | 94.49% ± 0.52% |
| SSC | 78.22% | 78.31% | 78.04% | 3 | 78.19% ? 0.14% |
| ECG | 87.81% | 87.81% | pending | 2 | 87.81% ? 0.00% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 451,484 ± 0 | 450,744 ± 0 | 10,514 ± 613 | 8,320 ± 239 / 2,194 ± 434 | 9.24 ± 0.26 / 2.44 ± 0.48 |
| SSC | 3 | 1,691,354 ± 0 | 1,689,879 ± 0 | 9,432 ± 404 | 9,001 ± 369 / 431 ± 59 | 4.17 ± 0.17 / 0.20 ± 0.03 |
| ECG | 2 | 4,768 ± 0 | 4,690 ± 0 | 29,270 ± 3,493 | 11,530 ± 2,628 / 17,740 ± 865 | 24.62 ± 5.61 / 37.88 ± 1.85 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.78 ± 0.30 | 0.242895 ± 0.000000 | 0.250372 ± 0.000000 | 0.251353 ± 0.000000 | 1.610 ± 0.040 |
| SSC | 3 | 19.37 ± 0.56 | 1.008720 ± 0.000000 | 1.025202 ± 0.000000 | 1.027365 ± 0.000000 | 1.899 ± 0.015 |
| ECG | 2 | 1.57 ± 0.23 | 0.011053 ± 0.000000 | 0.014678 ± 0.000000 | 0.015150 ± 0.000000 | 43.741 ± 0.103 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

## Archived unmatched ECG result

| Dataset | Seed | Historical test accuracy | Reason excluded | Archived record |
|---|---:|---:|---|---|
| ECG | 42 | 88.35% | Early stopping enabled; paper baseline disables it | [Original run](../../../archive/unmatched_early_stopping_20261017/DTH_SE_adLIF/ECG/seed_42) |

All configs, checkpoints, metrics and test evidence were retained. Their hashes
are recorded in the [archive manifest](../../../archive/unmatched_early_stopping_20261017/ARCHIVE_MANIFEST.json).
The current ECG table includes the imported full-budget runs; remaining seeds are pending. Historical early-stopped runs above are excluded.

## Evidence audit - 18 October 2026

SSC seeds 123/456 now have verified original checkpoints imported on 19 October 2026. SSC seed 42 has matching saved settings but retains its original `controlled_v2` label without baseline-version metadata. See per-seed audit notes. ECG now has two full-budget completions imported on 19 October 2026; archived early-stopped scores remain excluded.

## ECG transfer - 19 October 2026

Two full 400-epoch runs were moved from the user-provided external transfer directory, with every original file hash verified. No existing active seed directory was replaced. Older early-stopped records remain archived. The current SE-adLIF ECG reference still has unmatched stopping/budget settings, so these proposed-model completions do not establish a matched reference comparison. See the [transfer record](../../../docs/transfers/ecg-transfer-2026-10-19.md).

## SSC transfer - 19 October 2026

Seeds 123/456 were moved from external storage after hash verification. Previous local seed records were archived, with DTH metrics omitted from the transfer preserved in the active folders. All three SSC training seeds are now complete for the summed-loss protocol. These are not the mean-loss diagnostic results. See the [transfer record](../../../docs/transfers/ssc-transfer-2026-10-19.md).
