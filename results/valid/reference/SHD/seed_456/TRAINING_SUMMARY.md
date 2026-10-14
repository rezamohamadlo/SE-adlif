# SHD_SE_adLIF_paper Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SE_adLIF_paper
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **99.08%** | 243 |
| Validation loss | **0.0705** | 243 |
| Training accuracy | **100.00%** | 243 |
| Training loss | **0.000010** | 243 |
| Gradient norm | **0.0001** | 243 |
| Learning rate | **0.01** | 243 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **450,760** |
| Trainable parameters | **450,020** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=243-step=6100.ckpt
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

## Evaluate saved reference checkpoint

```powershell
.\.ven\Scripts\python.exe test_checkpoint.py --checkpoint "results/valid/reference/SHD/seed_456/ckpt/epoch=243-step=6100.ckpt"
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **93.64%** |
| Test loss | **0.5895** |

## Canonical reference provenance

Protocol: **SHD***; baseline: `paper_classification_v1`. Completed 300 epochs with batch 256 and early stopping disabled.

Original run: `results/experiments/shd_paper/multirun/2026-10-14/10-10-21/seed_456`. Saved runtime configs and checkpoint contents are retained unchanged. Current metadata: `REFERENCE_RESULT.json`.
