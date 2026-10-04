# SE-adLIF Training Summary — Seed 42

## Run status

- **Status:** Stopped; resumable and not yet complete
- **Progress:** 109 epochs completed (`0` through `108`) out of 300
- **Dataset:** SHD
- **Model:** Two-layer SE-adLIF
- **Seed:** 42
- **Device:** CUDA GPU

## Results so far

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **94.96%** | 108 |
| Validation loss at best accuracy | **0.3053** | 108 |
| Latest training accuracy | **99.97%** | 108 |
| Latest training loss | **0.00129** | 108 |
| Latest logged gradient norm | **0.2307** | 106 |
| Learning rate | **0.01** | Constant |

The current best SE-adLIF validation accuracy is approximately **5.08 percentage points higher** than the earlier LIF result of **89.89%**.

## Best checkpoint

```text
ckpt/epoch=108-step=1635.ckpt
```

The latest resumable state is stored under `ckpt/last*.ckpt`. The launcher automatically selects the newest one.

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epochs | 300 |
| Batch size | 512 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| LR reduction factor | 0.9 |
| LR scheduler patience | 9999 (effectively disabled) |
| Hidden layers | 2 |
| Neurons per hidden layer | 360 |
| Dropout | 0.15 |
| Gradient clipping | 1.5 |
| `tau_u` range | 5–25 |
| `tau_w` range | 60–300 |
| `q` | 120 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

From the repository root:

```powershell
.\run_phase1_se_adlif.ps1 -Mode reference -Resume 1
```

## Evaluation note

The current dataset configuration uses `validate_on: test`. Therefore, the reported validation accuracy is measured on the SHD test split and is also being used for checkpoint selection. It is suitable for reproducing the current reference setup, but it is not an unbiased final test result. A separate validation split should be used for the final publishable experiment.

This summary is an intermediate snapshot. It must be updated after all 300 epochs and again after seeds 123 and 456 complete.
