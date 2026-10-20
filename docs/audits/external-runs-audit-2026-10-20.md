# External ECG results inspection - 20 October 2026

Read-only inspection of user-provided external storage `F:/reza/SE-adlif`; no files moved or overwritten. Six ECG runs found, with three training seeds each for DTH and MT. All have 400 validation epochs, last checkpoints at epoch 399/step 3600, early stopping disabled, batch 64, LR 0.01, validation fraction 0.05 and fixed split seed 42. No duplicate validation epochs were found. Selected checkpoint callback scores match best validation metrics, and all six cost-record checkpoint hashes match.

| Model | Seed | Test (%) | Local status |
|---|---:|---:|---|
| DTH_SE_adLIF | 123 | 87.81 | Existing configs/metrics/checkpoint match |
| DTH_SE_adLIF | 42 | 87.81 | Existing configs/metrics/checkpoint match |
| DTH_SE_adLIF | 456 | 87.27 | Missing locally; new completion |
| MT_SE_adLIF | 123 | 88.19 | Existing configs/metrics/checkpoint match |
| MT_SE_adLIF | 42 | 87.26 | Missing locally; new completion |
| MT_SE_adLIF | 456 | 87.98 | Existing configs/metrics/checkpoint match |

DTH seed 456 and MT seed 42 are the two newly available completions. The four other runs duplicate previously imported evidence and must not be counted again. After a verified import, both models would have three ECG seeds locally. The SE-adLIF ECG reference remains unmatched in budget/stopping; these completions do not fix its comparison limitations. ECG uses per-timestep target blocks, so its `summed_membrane_potentials` setting is not the SSC temporal-sum issue.

[Detailed evidence](external-runs-audit-2026-10-20.json). No import or main-report updates were performed.
