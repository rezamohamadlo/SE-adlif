# DTH_SE_adLIF: staged validation study

Protocol: dth_complement_rdrop_ssc_v1_SSC; dataset label: SSC.

Completed 30 epochs (indices 0 through 29); full budget 40. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 82.116020%; selected epoch index 18. No official test evaluation.

Authoritative receipt: `stages/030/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/dth_complement_B001_ssc_v1.yaml --stage-ceiling 30
```
