# Shared evaluation runner

The root file `evaluation.yaml` defines comparison settings. The public entry
point is `runner.py`; `run.py` performs one internal training job.

```powershell
.\.venv\Scripts\python.exe runner.py --models SE_adLIF DTH_SE_adLIF DA_SE_adLIF MR_SE_adLIF MT_SE_adLIF --datasets SHD --seeds 42 123 456 --dry-run
```

Remove `--dry-run` to train. Omit `--datasets` to select SHD, SSC, and ECG;
omit `--seeds` to use the default seeds from `evaluation.yaml`.

`--resume 1` is the default: resume from `last.ckpt` and skip completed runs.
Use `--resume 0` for fresh runs only; nonempty result directories are rejected
to preserve existing artifacts. Supply a new `--logdir` to start over.

| Dataset | Epoch limit | Batch size | LR | Early stopping | Validation |
|---|---:|---:|---:|---|---|
| SHD | 300 | 512 | 0.01 | 50 checks, delta 0.001 | Random 20% of training, split seed 42 |
| SSC | 40 | 256 | 0.006 | Disabled | Official validation set |
| ECG | 400 | 64 | 0.01 | 50 checks, delta 0.001 | 5% of training, split seed 42 |

Training seeds are supplied on the command line; dataset split seeds stay
fixed across models. Scheduler patience is 9999 and factor is 0.9.
Architecture and preprocessing come from experiment/dataset configs.
Parameter counts may differ. Final testing follows validation checkpoint selection.

Results retain the existing `results/valid/protocol_v2/<model>/<dataset>/seed_<seed>`
layout (MR adds `K_2`). Rerun the same command to resume incomplete jobs and skip
completed jobs. Saved manifests reject changed settings. The launcher rename
alone is accepted for existing controlled-v2 manifests.

Legacy Python and PowerShell launchers have been removed. Use `runner.py`
for comparisons. All protocol loading uses the single root `evaluation.yaml`.
Changing the file does not change training processes already running.

## Per-run training overrides

```powershell
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets SHD --seeds 42 --epochs 100 --batch-size 256 --lr 0.005 --early-stopping true --early-stopping-patience 20 --dry-run
```

Also supported: `--early-stopping-min-delta`, `--lr-scheduler-patience`, and
`--lr-scheduler-factor`. Overrides apply to every selected model and dataset;
dataset split settings stay fixed. Without overrides the shared protocol is
unchanged. Override runs use `results/experiments/<protocol>_custom_<hash>`;
identical overrides reuse the same folder for resume. `--logdir` sets a custom
results root. Existing manifest checks still reject incompatible reuse.
