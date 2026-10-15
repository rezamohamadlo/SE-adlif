# LG-DTH SHD* continuation with the paper baseline

`runner.py` resolves training settings from `config/paper_baseline.yaml` through
`config/comparison/SHD.yaml`. No historical runner or second training baseline
is needed.

Existing SHD runs with legacy controlled-v2 metadata may be reused only when
all effective configuration values match the current resolved configuration.
The comparison excludes the old protocol label and newly added baseline
metadata. An omitted `dataset.additional_test_set_validation` is equivalent to
its existing dataset default, `false`. Model settings, training and split seeds,
preprocessing, optimization, budgets and stopping rules must still match.
Unknown protocol labels and mismatched baseline versions are not migrated.

On an actual launch, compatible existing runs retain `FULL_RUN_CONFIG.json`.
`PAPER_BASELINE_COMPATIBILITY.json` records both the original config and the
resolved baseline config. This establishes configuration compatibility, not
retroactive baseline inheritance or equivalence to uninterrupted training.
The dry-run does not modify these records or launch training.

```powershell
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 42 123 456 --logdir results/valid/protocol_v2 --dry-run
```

Verified plan on 2026-10-15: seed 42 complete (skip), seed 123 resume from
`ckpt/last.ckpt`, seed 456 new (random weights and fresh optimizer).
Remove `--dry-run` to launch the sequential suite, after ensuring another
launcher is not already writing these run directories. Resume is enabled by
default. The runner maintains completion markers and the aggregate report.

These runs use batch 256, 300 total epochs, LR 0.01, disabled early stopping,
20% training-derived validation and fixed split seed 42. Official test evaluation
uses the validation-selected best checkpoint after training.

The commands use this checkout's existing `.venv`; the documented `.ven`
interpreter is absent here.
