# SE_adLIF: staged validation study

Protocol: paper_classification_v1_complementary_speaker_dev_v2; dataset label: ECG.

Completed 100 epochs (indices 0 through 99); full budget 400. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 77.612096%; selected epoch index 95. No official test evaluation.

Authoritative receipt: `stages/100/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF --datasets ECG --seeds 42 --study-file config/studies/complementary_transfer_A002_gain_v1.yaml --stage-ceiling 100
```
