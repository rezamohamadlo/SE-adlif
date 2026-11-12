# LS_SE_adLIF: staged validation study

Protocol: label_smoothing_transfer_v1_ECG_classification_label_smoothing_v1; dataset label: ECG.

Completed 100 epochs (indices 0 through 99); full budget 400. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 76.938254%; selected epoch index 99. No official test evaluation.

Training loss is the smoothed objective; `train_hard_ce` is logged separately. Validation/inference use ordinary hard-label CE.

Authoritative receipt: `stages/100/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models LS_SE_adLIF --datasets ECG --seeds 42 --study-file config/studies/complementary_B002_transfer_v1.yaml --stage-ceiling 100
```
