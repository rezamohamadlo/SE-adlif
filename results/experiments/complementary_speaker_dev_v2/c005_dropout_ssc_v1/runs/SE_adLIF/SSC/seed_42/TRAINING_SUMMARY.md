# SE_adLIF: staged validation study

Protocol: dropout_consistency_ssc_mean_loss_v1_SSC; dataset label: SSC.

Completed 10 epochs (indices 0 through 9); full budget 40. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 79.711449%; selected epoch index 9. No official test evaluation.

Authoritative receipt: `stages/010/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/complementary_C005_dropout_ssc_v1.yaml --stage-ceiling 10
```
