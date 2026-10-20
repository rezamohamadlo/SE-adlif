# MT_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
Dataset losses/protocols: SHD: sum_softmax_over_time (paper_classification_v1); SSC: mean_membrane_potentials (ssc_mean_loss_v1); ECG: summed_membrane_potentials (paper_classification_v1)
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.88% | 94.52% | 94.43% | 3 | 94.61% ± 0.23% |
| SSC | 80.01% | pending | pending | 1 | 80.01% (n=1; SD unavailable) |
| ECG | 87.26% | 88.19% | 87.98% | 3 | 87.81% ± 0.49% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,200 ± 0 | 451,460 ± 0 | 10,637 ± 609 | 8,406 ± 556 / 2,232 ± 88 | 9.33 ± 0.62 / 2.48 ± 0.10 |
| SSC | 1 | 1,692,790 (n=1) | 1,691,315 (n=1) | 48,897 (n=1) | 22,132 (n=1) / 26,766 (n=1) | 10.25 (n=1) / 12.39 (n=1) |
| ECG | 3 | 4,836 ± 0 | 4,758 ± 0 | 26,665 ± 378 | 9,945 ± 1,226 / 16,720 ± 1,166 | 21.23 ± 2.62 / 35.70 ± 2.49 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.86 ± 0.42 | 0.242895 ± 0.000000 | 0.249006 ± 0.000000 | 0.249597 ± 0.000000 | 1.668 ± 0.026 |
| SSC | 1 | 58.09 (n=1) | 1.008720 (n=1) | 1.022190 (n=1) | 1.023493 (n=1) | 1.563 (n=1) |
| ECG | 3 | 1.42 ± 0.04 | 0.011053 ± 0.000000 | 0.014033 ± 0.000000 | 0.014322 ± 0.000000 | 44.670 ± 0.529 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->


## Archived unmatched ECG result

| Dataset | Seed | Historical test accuracy | Reason excluded | Archived record |
|---|---:|---:|---|---|
| ECG | 87.26% | 88.19% | 87.98% | 3 | 87.81% ? 0.49% |

All configs, checkpoints, metrics and test evidence were retained. Their hashes
are recorded in the [archive manifest](../../../archive/unmatched_early_stopping_20261017/ARCHIVE_MANIFEST.json).
The current ECG table includes the imported full-budget runs; remaining seeds are pending. Historical early-stopped runs above are excluded.

## Evidence audit - 18 October 2026

SSC seeds 123/456 now have complete metrics and checkpoints imported on 19 October 2026. SSC seed 42 has matching saved settings but retains its original `controlled_v2` label without baseline-version metadata. See per-seed audit notes. ECG now has two full-budget completions imported on 19 October 2026; archived early-stopped scores remain excluded.

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
