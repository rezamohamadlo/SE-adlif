# Controlled comparison: protocol v2

This protocol is separate from historical Phase 1 and development results.
No old checkpoints are used. All model variants share dataset-specific settings:

| Setting | SHD | SSC | ECG |
|---|---:|---:|---:|
| Hidden layers / neurons | 2 / 360 | 2 / 720 | 2 / 36 |
| Batch size | 512 | 256 | 64 |
| Initial Adam LR | 0.01 | 0.006 | 0.01 |
| Maximum epochs | 300 | 40 | 400 |
| Early stopping | patience 50 | disabled | patience 50 |
| Validation | 20% of training | official valid split | 5% of training |

Training seeds are 42, 123, 456. SHD/ECG split seed stays 42 across models and
training seeds. Checkpoints are selected by maximum validation accuracy; test
evaluation follows training. Early-stopping min_delta is 0.001, dropout 0.15,
gradient clipping 1.5, tau_u [5,25], tau_w [60,300], q 120. The LR scheduler is
effectively disabled (patience 9999); remaining preprocessing and neuron
settings are inherited from the dataset's reference experiment.

SHD caches use an opt-in namespace plus a split/preprocessing fingerprint.
Old caches remain untouched. The first controlled run populates a fresh cache;
subsequent models reuse it safely with the identical split. Existing launchers
retain their legacy cache paths when the namespace is not supplied.

## Launch candidates (27 sequential runs)

```powershell
.\.venv\Scripts\python.exe full_runner.py DTH_SE_adLIF DA_SE_adLIF MR_SE_adLIF --adaptation-update-interval 2
```

Append `--dry-run` to preflight all runs without training or writing files.
Rerun the same command to resume incomplete runs and skip completed runs.
Configuration manifests prevent incompatible checkpoint reuse.

Results are saved under `results/full_runs/protocol_v2/<model>/<dataset>/seed_<seed>/`;
MR adds `K_2` between model and dataset. Each model/K has a separate `RESULTS.md`
with final-test mean and sample SD from completed seeds only.

## Matched reference (9 additional runs)

```powershell
.\.venv\Scripts\python.exe full_runner.py SE_adLIF
```

Alternatively, prepend `SE_adLIF` to the candidate command to queue all 36 runs.
Do not combine historical test-selected SHD scores or duplicate seeds with
these controlled-protocol statistics. Three seeds provide descriptive
uncertainty, not definitive proof of a small improvement.
