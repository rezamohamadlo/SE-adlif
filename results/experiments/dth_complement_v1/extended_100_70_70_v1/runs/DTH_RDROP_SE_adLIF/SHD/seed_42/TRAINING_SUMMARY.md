# DTH_RDROP_SE_adLIF: staged validation study

Protocol: dth_rdrop_extended_100_70_70_v1_SHD_classification_dropout_consistency_v1; dataset label: SHD*.

Completed 100 epochs (indices 0 through 99); full budget 300. This is a preliminary screen, not full-budget completion.

Best validation accuracy: 98.896384%; selected epoch index 91. No official test evaluation.

Training loss averages two supervised dropout passes plus weighted symmetric KL; accuracy uses the first pass only. Hard CE, supervised loss and KL are logged separately. Validation/inference use one ordinary forward and hard-label CE.

Authoritative receipt: `stages/100/RESULT.json`; per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.

Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):

```powershell
.\.ven\Scripts\python.exe runner.py --models DTH_RDROP_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/dth_complement_extended_100_70_70_v1.yaml --stage-ceiling 100
```
