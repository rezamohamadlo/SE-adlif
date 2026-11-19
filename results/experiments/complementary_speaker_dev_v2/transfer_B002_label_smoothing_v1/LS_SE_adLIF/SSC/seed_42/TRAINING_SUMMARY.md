# LS_SE_adLIF: staged validation study

Protocol: label_smoothing_transfer_v1_SSC_classification_label_smoothing_v1; dataset label: SSC.

Completed 40 epochs (indices 0 through 39); full budget 40. The full training budget is complete; official-test evaluation remains pending.

Best validation accuracy: 81.955713%; selected epoch index 35. No official test evaluation.

Training loss is the smoothed objective; `train_hard_ce` is logged separately. Validation/inference use ordinary hard-label CE.

Authoritative receipt: `stages/040/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models LS_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/complementary_B002_transfer_v1.yaml --stage-ceiling 40
```
