# DTH_RDROP_SE_adLIF: staged validation study

Protocol: dth_complement_rdrop_ecg30_v1_ECG_classification_dropout_consistency_v1; dataset label: ECG.

Completed 30 epochs (indices 0 through 29); full budget 400. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 79.251856%; selected epoch index 27. No official test evaluation.

Training loss averages two supervised dropout passes plus weighted symmetric KL; accuracy uses the first pass only. Hard CE, supervised loss and KL are logged separately. Validation/inference use one ordinary forward and hard-label CE.

Authoritative receipt: `stages/030/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models DTH_RDROP_SE_adLIF --datasets ECG --seeds 42 --study-file config/studies/dth_complement_B001_ecg30_v1.yaml --stage-ceiling 30
```
