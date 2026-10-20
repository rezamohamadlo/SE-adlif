# DTH_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.70% | 94.88% | 93.90% | 3 | 94.49% ± 0.52% |
| SSC | 80.46% | pending | pending | 1 | 80.46% (n=1; SD unavailable) |
| ECG | 87.81% | 87.81% | 87.27% | 3 | 87.63% ± 0.31% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 451,484 ± 0 | 450,744 ± 0 | 10,514 ± 613 | 8,320 ± 239 / 2,194 ± 434 | 9.24 ± 0.26 / 2.44 ± 0.48 |
| SSC | 1 | 1,691,354 (n=1) | 1,689,879 (n=1) | 45,402 (n=1) | 19,019 (n=1) / 26,383 (n=1) | 8.81 (n=1) / 12.21 (n=1) |
| ECG | 3 | 4,768 ± 0 | 4,690 ± 0 | 30,058 ± 2,822 | 11,631 ± 1,867 / 18,427 ± 1,338 | 24.83 ± 3.99 / 39.34 ± 2.86 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.78 ± 0.30 | 0.242895 ± 0.000000 | 0.250372 ± 0.000000 | 0.251353 ± 0.000000 | 1.610 ± 0.040 |
| SSC | 1 | 53.34 (n=1) | 1.008720 (n=1) | 1.025202 (n=1) | 1.027365 (n=1) | 1.605 (n=1) |
| ECG | 3 | 1.61 ± 0.17 | 0.011053 ± 0.000000 | 0.014678 ± 0.000000 | 0.015150 ± 0.000000 | 43.915 ± 0.310 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->


## Archived unmatched ECG result

| Dataset | Seed | Historical test accuracy | Reason excluded | Archived record |
|---|---:|---:|---|---|
| ECG | 87.81% | 87.81% | 87.27% | 3 | 87.63% ? 0.31% |

All configs, checkpoints, metrics and test evidence were retained. Their hashes
are recorded in the [archive manifest](../../../archive/unmatched_early_stopping_20261017/ARCHIVE_MANIFEST.json).
The current ECG table includes the imported full-budget runs; remaining seeds are pending. Historical early-stopped runs above are excluded.

## Evidence audit - 18 October 2026

SSC seeds 123/456 now have verified original checkpoints imported on 19 October 2026. SSC seed 42 has matching saved settings but retains its original `controlled_v2` label without baseline-version metadata. See per-seed audit notes. ECG now has two full-budget completions imported on 19 October 2026; archived early-stopped scores remain excluded.

## ECG transfer - 19 October 2026

Two full 400-epoch runs were moved from the user-provided external transfer directory, with every original file hash verified. No existing active seed directory was replaced. Older early-stopped records remain archived. The current SE-adLIF ECG reference still has unmatched stopping/budget settings, so these proposed-model completions do not establish a matched reference comparison. See the [transfer record](../../../docs/transfers/ecg-transfer-2026-10-19.md).

## SSC transfer - 19 October 2026

Seeds 123/456 were moved from external storage after hash verification. Previous local seed records were archived, with DTH metrics omitted from the transfer preserved in the active folders. All three SSC training seeds are now complete for the summed-loss protocol. These are not the mean-loss diagnostic results. See the [transfer record](../../../docs/transfers/ssc-transfer-2026-10-19.md).

## SSC archive - 19 October 2026

All prior summed-loss SSC seeds were moved to `archive/ssc_summed_loss_20261019/`. The active SSC row is pending. The completed mean-loss reference diagnostic remains separate, with 82.22% best validation accuracy and no test evaluation. New mean-loss training needs explicit separate configs; the existing runner defaults were not changed by this archive. See the [archive guide](../../../docs/transfers/ssc-summed-loss-archive-2026-10-19.md).

SSC mean-loss seed-42 training and testing completed under `ssc_mean_loss_v1`, with shared config/environment/source checks against the promoted reference. Seed-42 final results are shown above; seeds 123/456 remain pending. Use the [official runner guide](../../../docs/experiments/ssc-mean-comparison.md), which selects mean loss automatically for SSC.

## ECG completion import - 20 October 2026

The previously missing ECG seed was copied from user-provided external storage with all file hashes verified. All three ECG training seeds now have full 400-epoch/no-early-stopping results and cost evidence. The historical SE-adLIF ECG reference is still unmatched. See the [copy record](../../../docs/transfers/ecg-copy-2026-10-20.md).

## Accepted comparison grid - 20 October 2026

This model has 7/9 accepted matched runs: all SHD* and ECG seeds plus SSC mean-loss seed 42. SSC seeds 123/456 remain missing locally. The project has 21/27 accepted results; three ECG reference records are accepted historical evidence with unmatched stopping/budget, so ECG reference comparisons are descriptive. See the [current grid](../COMPARISON_STATUS.md). Dated archive/transfer notes above describe earlier states.
