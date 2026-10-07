# SHD_MT_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Manually stopped; best checkpoint evaluated and run finalized
- **Progress:** 157 epochs logged through validation (`0` through `156`) out of 300; interrupted during epoch 157
- **Dataset:** SHD
- **Model:** SHD_MT_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **93.73%** | 104 |
| Validation loss | **0.3555** | 104 |
| Training accuracy | **99.99%** | 104 |
| Training loss | **0.000170** | 104 |
| Gradient norm | **0.0425** | 104 |
| Learning rate | **0.01** | 104 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,200** |
| Trainable parameters | **451,460** |
| Average padded validation timesteps/sample | **281.11** |
| Estimated dense forward GFLOPs/sample | **0.2510** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=104-step=1575.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 512 |
| Initial learning rate | 0.01 |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.15 |
| Hidden layers | 2 |
| Neurons per hidden layer | 360 |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
.\minimal_runner.ps1 -Mode ours -ModelVariant MT_SE_adLIF -Datasets SHD -Seeds 42 -Epochs 300 -BatchSize 512 -EarlyStopping 0 -LrSchedulerPatience 9999 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/model_development
```

## Evaluation note

This run uses `validate_on: test`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

After manual interruption, the best checkpoint from epoch 104 was evaluated
directly on the test split without additional training. Early stopping was
disabled during this run; the stop was manual. SHD uses the same test split
for checkpoint selection and this evaluation.

| Metric | Value |
|---|---:|
| Test accuracy | **93.73%** |
| Test loss | **0.3555** |
