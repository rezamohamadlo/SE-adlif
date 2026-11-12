# GAIN_SE_adLIF: staged validation study

Protocol: ssc_mean_loss_v1_complementary_speaker_dev_v2; dataset label: SSC.

Completed 40 epochs (indices 0 through 39); full budget 40. The full training budget is complete; official-test evaluation remains pending.

Best validation accuracy: 81.554955%; selected epoch index 23. No official test evaluation.

Authoritative receipt: `stages/040/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models GAIN_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/complementary_transfer_A002_gain_v1.yaml --stage-ceiling 40
```
