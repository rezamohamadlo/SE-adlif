# Full evaluation runner

From the project root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe full_runner.py MT_SE_adLIF
```

Supply one or more model-config suffixes. For example:

```powershell
.\.venv\Scripts\python.exe full_runner.py DTH_SE_adLIF MT_SE_adLIF
```

Models run sequentially in the supplied order. All requested models are
validated before training starts; repeated names are scheduled only once.
Supported configured models are
`SE_adLIF`, `DTH_SE_adLIF`, and `MT_SE_adLIF`. Future variants need SHD, SSC,
and ECG experiment configs and a registered neuron cell.

The runner launches SHD, SSC, and ECG sequentially, each with seeds 42, 123,
and 456. It uses the same explicit protocol for every model:

| Dataset | Hidden layers × neurons | Epoch limit | Batch | LR | Early stopping |
|---|---:|---:|---:|---:|---|
| SHD | 2 × 360 | 300 | 512 | 0.01 | Patience 50, min_delta 0.001 |
| SSC | 2 × 720 | 40 | 256 | 0.006 | Disabled |
| ECG | 2 × 36 | 400 | 64 | 0.01 | Patience 50, min_delta 0.001 |

LR scheduler patience is 9999 for all runs. Architecture and dataset-specific
loss settings are inherited from reference configs. SHD still uses the test
split for checkpoint selection; this is a reproduction/development protocol,
not an unbiased final test estimate. Historical reference runs used some
different stopping/scheduler settings and remain separate experimental records.

Outputs: `results/full_runs/<model>/<dataset>/seed_<seed>/`.
Aggregate report: `results/full_runs/<model>/RESULTS.md`.

Rerun the identical command to skip completed seeds and resume interrupted
ones from `ckpt/last.ckpt`. The resolved config must match the original run.
Existing unrecognized results or artifacts without a resumable checkpoint
cause an error rather than being overwritten. A failed subprocess stops the
suite. Aggregate statistics include final test results from completed seeds
only; mean ± sample SD is recalculated after each completion.

This runner starts fresh experiments in its own results hierarchy. It does
not automatically reuse development checkpoints or historical reference runs.
