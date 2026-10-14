# Two-layer SHD* paper reproduction

Use `experiment=SHD_SE_adLIF_paper` with the unchanged single-job `run.py`.
The original experiment, shared `evaluation.yaml`, comparison runner, and
existing results are unchanged.

The single versioned source of shared paper settings is
`config/paper_baseline.yaml`. This experiment inherits it and resolves training
and preprocessing values through its SHD section. The baseline metadata and
`paper_baseline_version` are saved in new Hydra configs and checkpoints.
Paper-documented settings are separated from local reproduction choices.
Already-running jobs retain the configuration composed at startup.

The local `reference_paper.pdf`, Table 7 (page 35), specifies SHD batch 256,
Adam LR 0.01, dropout 0.15, 300 epochs, two layers of 360 neurons, q 120,
SLAYER alpha 5/c 0.4, tau_u [5,25], tau_w [60,300], and output tau 15.
The new config inherits those architecture and preprocessing settings from
`SHD_SE_adLIF`; it updates both batch fields and disables our added early
stopping so the full listed epoch budget runs. The scheduler remains
effectively constant (patience 9999).

SHD* checkpoint selection uses 20% of training as validation. The official
test split is evaluated once after training at the best validation checkpoint.
The retained local split seed is 42; the paper's exact split indices and
software environment have not been verified. This config matches documented
settings, not a guarantee of identical results.

First run seed 42 from scratch:

```powershell
.\.ven\Scripts\python.exe run.py experiment=SHD_SE_adLIF_paper random_seed=42
```

Alternatively run three independent seeds sequentially:

```powershell
.\.ven\Scripts\python.exe run.py -m experiment=SHD_SE_adLIF_paper random_seed=42,123,456
```

Single runs use `results/experiments/shd_paper/seed_42/hydra/<date>/<time>/`.
Multiruns use `results/experiments/shd_paper/multirun/<date>/<time>/seed_<seed>/`.
Each saves its best checkpoint, last checkpoint, metrics, and final test
accuracy in `TRAINING_SUMMARY.md`. Repeating either command starts new runs
in new timestamped folders; it does not resume existing ones automatically.
Prefer uninterrupted runs for reproduction. Neither command loads historical
weights. Do not pass `evaluation_file=evaluation.yaml`, which would apply the
different controlled_v2 batch/stopping settings over this configuration.

The paper's 93.79 +/- 0.76% is a mean and SD across 20 initializations, not an
individual-seed target. Compare multi-seed averages and report uncertainty.
