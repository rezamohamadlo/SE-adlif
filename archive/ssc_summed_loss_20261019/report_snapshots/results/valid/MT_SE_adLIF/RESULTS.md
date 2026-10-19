# MT_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Target comparison protocol: paper_classification_v1; original per-run provenance is retained. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.88% | 94.52% | 94.43% | 3 | 94.61% ± 0.23% |
| SSC | 77.60% | 78.14% | 77.46% | 3 | 77.73% ? 0.36% |
| ECG | pending | 88.19% | 87.98% | 2 | 88.08% ? 0.15% |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* | 3 | 452,200 ± 0 | 451,460 ± 0 | 10,637 ± 609 | 8,406 ± 556 / 2,232 ± 88 | 9.33 ± 0.62 / 2.48 ± 0.10 |
| SSC | 3 | 1,692,790 ± 0 | 1,691,315 ± 0 | 9,410 ± 406 | 9,004 ± 328 / 406 ± 94 | 4.17 ± 0.15 / 0.19 ± 0.04 |
| ECG | 2 | 4,836 ± 0 | 4,758 ± 0 | 26,487 ± 307 | 9,379 ± 1,040 / 17,108 ± 1,347 | 20.02 ± 2.22 / 36.53 ± 2.88 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* | 3 | 9.86 ± 0.42 | 0.242895 ± 0.000000 | 0.249006 ± 0.000000 | 0.249597 ± 0.000000 | 1.668 ± 0.026 |
| SSC | 3 | 19.35 ± 0.53 | 1.008720 ± 0.000000 | 1.022190 ± 0.000000 | 1.023493 ± 0.000000 | 1.916 ± 0.018 |
| ECG | 2 | 1.39 ± 0.02 | 0.011053 ± 0.000000 | 0.014033 ± 0.000000 | 0.014322 ± 0.000000 | 44.376 ± 0.206 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

## Archived unmatched ECG result

| Dataset | Seed | Historical test accuracy | Reason excluded | Archived record |
|---|---:|---:|---|---|
| ECG | 42 | 87.23% | Early stopping enabled; paper baseline disables it | [Original run](../../../archive/unmatched_early_stopping_20261017/MT_SE_adLIF/ECG/seed_42) |

All configs, checkpoints, metrics and test evidence were retained. Their hashes
are recorded in the [archive manifest](../../../archive/unmatched_early_stopping_20261017/ARCHIVE_MANIFEST.json).
The current ECG table includes the imported full-budget runs; remaining seeds are pending. Historical early-stopped runs above are excluded.

## Evidence audit - 18 October 2026

SSC seeds 123/456 now have complete metrics and checkpoints imported on 19 October 2026. SSC seed 42 has matching saved settings but retains its original `controlled_v2` label without baseline-version metadata. See per-seed audit notes. ECG now has two full-budget completions imported on 19 October 2026; archived early-stopped scores remain excluded.

## ECG transfer - 19 October 2026

Two full 400-epoch runs were moved from the user-provided external transfer directory, with every original file hash verified. No existing active seed directory was replaced. Older early-stopped records remain archived. The current SE-adLIF ECG reference still has unmatched stopping/budget settings, so these proposed-model completions do not establish a matched reference comparison. See the [transfer record](../../../docs/transfers/ecg-transfer-2026-10-19.md).

## SSC transfer - 19 October 2026

Seeds 123/456 were moved from external storage after hash verification. Previous local seed records were archived, with DTH metrics omitted from the transfer preserved in the active folders. All three SSC training seeds are now complete for the summed-loss protocol. These are not the mean-loss diagnostic results. See the [transfer record](../../../docs/transfers/ssc-transfer-2026-10-19.md).
