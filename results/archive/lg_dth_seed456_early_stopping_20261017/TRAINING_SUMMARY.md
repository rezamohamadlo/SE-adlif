# SHD_LG_DTH_SE_adLIF Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 105 epochs completed (`0` through `104`) out of 300
- **Dataset:** SHD
- **Model:** SHD_LG_DTH_SE_adLIF
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **98.59%** | 74 |
| Validation loss | **0.0854** | 74 |
| Training accuracy | **99.91%** | 74 |
| Training loss | **0.003439** | 74 |
| Gradient norm | **0.0164** | 74 |
| Learning rate | **0.01** | 74 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,924** |
| Trainable parameters | **452,184** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=74-step=1875.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 256 |
| Initial learning rate | 0.01 |
| LR scheduler | plateau |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | true |
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
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 456 --epochs 300 --batch-size 256 --lr 0.01 --early-stopping true --early-stopping-patience 50 --early-stopping-min-delta 0.001 --lr-scheduler-patience 9999 --lr-scheduler-factor 0.9 --logdir "results/experiments/paper_classification_v1_custom_af62d1bc3d42"
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **93.51%** |
| Test loss | **0.4035** |
