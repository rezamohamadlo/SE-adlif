# SSC_SE_adLIF_loss_diagnostic Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training finished
- **Progress:** 40 epochs completed (`0` through `39`) out of 40
- **Dataset:** SSC
- **Model:** SSC_SE_adLIF_loss_diagnostic
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **82.22%** | 18 |
| Validation loss | **0.6016** | 18 |
| Training accuracy | **86.42%** | 18 |
| Training loss | **0.429068** | 18 |
| Gradient norm | **0.2554** | 18 |
| Learning rate | **0.006** | 18 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **1,689,910** |
| Trainable parameters | **1,688,435** |
| Average padded validation timesteps/sample | **300.00** |
| Estimated dense forward GFLOPs/sample | **1.0087** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=18-step=5586.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 40 |
| Batch size | 256 |
| Initial learning rate | 0.006 |
| LR scheduler | plateau |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.15 |
| Hidden layers | 2 |
| Neurons per hidden layer | 720 |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
.\.venv\Scripts\python.exe runner.py --models SE_adLIF_loss_diagnostic --datasets SSC --seeds 42
```

## Evaluation note

SSC uses official validation for checkpoint selection and the separate official test set for final evaluation.

## Final test evaluation

The selected validation checkpoint was evaluated without further training.

Checkpoint: `epoch=18-step=5586.ckpt`

| Metric | Value |
|---|---:|
| Test accuracy | **80.69%** |
| Test loss | **0.6706** |
