# SHD_SA_MT_SE_adLIF Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SA_MT_SE_adLIF
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **95.34%** | 257 |
| Validation loss | **0.1693** | 257 |
| Training accuracy | **99.14%** | 257 |
| Training loss | **0.032866** | 257 |
| Gradient norm | **1.2332** | 257 |
| Learning rate | **0.00012552424** | 257 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,920** |
| Trainable parameters | **452,180** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=257-step=6450.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 256 |
| Initial learning rate | 0.0004 |
| LR scheduler | plateau |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 10 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.1 |
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
.\confirm_samt_shd.py -Mode ours -ModelVariant SA_MT_SE_adLIF -Datasets SHD -Seeds 456 -Epochs 300 -BatchSize 256 -EarlyStopping 0 -LrSchedulerPatience 10 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/experiments/samt_shd_first_trial_fresh_v1
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **89.31%** |
| Test loss | **0.4867** |
