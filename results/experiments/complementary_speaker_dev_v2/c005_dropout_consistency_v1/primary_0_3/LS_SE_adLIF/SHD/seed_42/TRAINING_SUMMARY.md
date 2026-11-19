# LS_SE_adLIF: staged validation study

Protocol: shd_speaker_dropout_consistency_v1_primary_0_3_classification_label_smoothing_v1; dataset label: SHD speaker-disjoint development.

Completed 30 epochs (indices 0 through 29); full budget 300. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 88.440365%; selected epoch index 27. No official test evaluation.

Training loss is the smoothed objective; `train_hard_ce` is logged separately. Validation/inference use ordinary hard-label CE.

Authoritative receipt: `stages/030/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models LS_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_C005_dropout_consistency_v1.yaml --stage-ceiling 30
```
