# SE_adLIF: staged validation study

Protocol: shd_speaker_label_smoothing_v1_sensitivity_6_10; dataset label: SHD speaker-disjoint development.

Completed 30 epochs (indices 0 through 29); full budget 300. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 84.304208%; selected epoch index 29. No official test evaluation.

Authoritative receipt: `stages/030/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_B002_sensitivity_v1.yaml --stage-ceiling 30
```
