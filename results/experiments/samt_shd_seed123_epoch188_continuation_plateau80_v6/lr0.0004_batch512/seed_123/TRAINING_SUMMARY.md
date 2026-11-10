# SHD_SA_MT_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 80 epochs completed (`0` through `79`) out of 80
- **Dataset:** SHD
- **Model:** SHD_SA_MT_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **99.08%** | 79 |
| Validation loss | **0.0850** | 79 |
| Training accuracy | **100.00%** | 79 |
| Training loss | **0.000000** | 79 |
| Gradient norm | **0.0000** | 79 |
| Learning rate | **0.0002125764** | 79 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,920** |
| Trainable parameters | **452,180** |
| Average padded validation timesteps/sample | **296.92** |
| Estimated dense forward GFLOPs/sample | **0.2651** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=79-step=960.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 80 |
| Batch size | 512 |
| Initial learning rate | 0.0004 |
| LR scheduler | plateau |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 10 |
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
.\search_samt_shd.py -Mode ours -ModelVariant SA_MT_SE_adLIF -Datasets SHD -Seeds 123 -Epochs 80 -BatchSize 512 -EarlyStopping 0 -LrSchedulerPatience 10 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/experiments/samt_shd_seed123_epoch188_continuation_plateau80_v6
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **93.07%** |
| Test loss | **0.6784** |
