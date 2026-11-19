# FIR_THIN_SE_adLIF: staged validation study

Protocol: complementary_speaker_dev_v2_primary_0_3; dataset label: SHD speaker-disjoint development.

Completed 30 epochs (indices 0 through 29); full budget 300. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 81.223243%; selected epoch index 21. No official test evaluation.

Authoritative receipt: `stages/030/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models FIR_THIN_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_speaker_dev_v2.yaml --stage-ceiling 30
```
