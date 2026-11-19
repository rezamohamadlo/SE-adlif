# FIR_SE_adLIF: staged validation study

Completed 75 epochs (indices 0 through 74); full budget 300. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 98.038012%; selected epoch index 63. No official test evaluation.

Authoritative receipt: `stages/075/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models FIR_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_mechanisms_v1.yaml --stage-ceiling 75
```
