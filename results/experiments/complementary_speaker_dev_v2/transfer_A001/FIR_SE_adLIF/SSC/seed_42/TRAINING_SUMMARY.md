# FIR_SE_adLIF: staged validation study

Protocol: ssc_mean_loss_v1_complementary_speaker_dev_v2; dataset label: SSC.

Completed 40 epochs (indices 0 through 39); full budget 40. The full training budget is complete; official-test evaluation remains pending.

Best validation accuracy: 80.923754%; selected epoch index 32. No official test evaluation.

Authoritative receipt: `stages/040/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models FIR_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/complementary_transfer_A001_v2.yaml --stage-ceiling 40
```

Status wording corrected against stages/040/RESULT.json; original retained in summary_history/. Recovery requires the matching source archived in the study root; do not bypass source guards.
