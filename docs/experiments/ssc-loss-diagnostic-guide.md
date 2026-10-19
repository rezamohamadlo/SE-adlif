# SSC sum/mean diagnostic

Prepared on 18 October 2026; training has not been launched by the agent.

This is a separate reference-neuron experiment using the existing SE-adLIF architecture. It changes the loss aggregation, not the neuron implementation. Six jobs run sequentially: summed-loss control and released-code mean-loss candidate, each for seeds 42, 123 and 456. Every job starts with random weights and a fresh optimizer, uses the SSC paper baseline for 40 epochs, and selects by official validation accuracy. Official test evaluation is disabled. Current canonical reference/proposed records are preserved.

The [experiment config](../../config/experiment/SSC_SE_adLIF_loss_diagnostic.yaml) inherits the SSC baseline through the comparison group. The [separate launcher](../../ssc_loss_diagnostic.py) records resolved configs, baseline/deviation metadata, package versions and source hashes. It locks the output root and rejects incompatible reuse, duplicate validation epochs and unexpected test results. Full-budget jobs are skipped on subsequent launches. Resume restores a job's own `last.ckpt`; it is not weights-only continuation. Partial metrics without that checkpoint require a separate output directory.

From the repository root in PowerShell, check the six configurations without training:

```powershell
.\.venv\Scripts\python.exe ssc_loss_diagnostic.py --dry-run
```

Start the sequential queue yourself:

```powershell
.\.venv\Scripts\python.exe ssc_loss_diagnostic.py --seeds 42 123 456
```

Resume the same queue after interruption:

```powershell
.\.venv\Scripts\python.exe ssc_loss_diagnostic.py --seeds 42 123 456 --resume
```

The launcher shows training progress, validation metrics and job transitions in the same terminal, and also saves the output per seed as `console.log`. A second terminal is optional; for example, follow the saved log with:

```powershell
Get-Content results/experiments/ssc_loss_diagnostic_v1/summed_membrane_potentials/seed_42/console.log -Tail 20 -Wait
```

Outputs are under `results/experiments/ssc_loss_diagnostic_v1/<loss>/seed_<seed>/`. The root `RESULTS.md` updates after each finished/failed job; per-seed `VALIDATION_RESULTS.json` is saved for new completions. These are validation results, not final test accuracy. Per-test cost measurement is not run because no test evaluation is performed.

The available local interpreter is `.venv/Scripts/python.exe`; `.ven` does not exist here. NumPy was restored to the project's pinned 1.26.4 before preparation, and `pip check` passes. On another checkout, create the project environment and transfer/download the local SSC files while preserving their relative layout; Git does not transfer datasets or ignored checkpoints. The launcher resolves project paths from its own repository root, including when invoked outside that root or when Hydra changes the working directory.

The [audit](../audits/ssc-reproduction-audit-2026-10-18.md) records the motivating paper/released-code disagreement. Neither arm is guaranteed to close the paper accuracy gap. The experiment was planned after historical test results were observed; use validation alone for this comparison and disclose that development provenance.
