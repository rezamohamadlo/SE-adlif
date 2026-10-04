# Phase 1 — LIF on SHD baseline

## Run summary

| Field | Value |
| --- | --- |
| Date | 2026-10-03 |
| Experiment | `SHD_LIF` |
| Dataset | Spiking Heidelberg Digits (SHD) |
| Model | Two-layer LIF network |
| Seed | 42 |
| Epochs | 300 |
| Batch size | 256 |
| Data-loader workers | 0 (Windows multiprocessing fallback) |
| Trainable parameters | 447,000 |
| Total parameters | 448,000 |
| Best checkpoint | Epoch 269, step 8,370 |

## Metrics

| Metric | Result |
| --- | ---: |
| Best validation accuracy | 89.89% |
| Best validation loss | 0.6759 |
| Final test accuracy | 89.89% |
| Final test loss | 0.6759 |

The final test pass restored the best checkpoint successfully.

## Reproduction command

```powershell
.\.venv\Scripts\python.exe run.py experiment=SHD_LIF ++logdir=results\phase1_lif ++datadir=data ++dataset.num_workers=0
```

## Interpretation and limitations

- The run confirms that the LIF SHD baseline trains successfully after correcting SHD timestamp conversion and PyTorch 2.6 checkpoint loading.
- `config/dataset/shd.yaml` currently sets `validate_on: test`. The test split therefore selected the best checkpoint and was also used for the reported test pass. The 89.89% value is suitable for an initial reproduction baseline, but is **not** an unbiased held-out test estimate.
- Spike rate, SynOps, and runtime have not yet been recorded. They remain required Phase 1 outputs.

## Next action

Run `SHD_SE_adLIF` with the same seed, preprocessing, batch size, epoch count, and data-loader setting. Then record equivalent metrics in a comparison table.
