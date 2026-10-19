# SSC mean-loss comparison: reference, DTH and MT seed 42

Protocol: `ssc_mean_loss_v1`. Shared settings inherit the SSC section of `config/paper_baseline.yaml`; temporal mean loss is an explicit deviation. Other datasets and the default main runner are unchanged. This is a separate comparison, not a silent baseline-value update.

The completed reference was moved from the diagnostic into `results/valid/reference/SSC/seed_42/` with all original file hashes verified. Its original config, manifest and checkpoint remain byte-identical. Validation selected epoch 18 (82.22%); standalone official test accuracy is 80.69%. The original training protocol remains `ssc_loss_diagnostic_v1`; the new comparison explicitly reuses that record. It is one seed, not an across-seed accuracy claim. The protocol was chosen after observing earlier test results and the reference test score; retain that development provenance.

Use the official `runner.py` for DTH and MT, sequentially, from random weights with fresh optimizer state at epoch 0. The SSC config automatically selects mean loss and `ssc_mean_loss_v1`; SHD/ECG are unchanged. Normal manifest guards, resume, validation-selected final test evaluation, model aggregation and per-seed cost measurement apply. No new training was started during preparation.

Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SSC --seeds 42 --resume 0 --dry-run
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SSC --seeds 42 --resume 0
```

To continue interrupted jobs from their own `last.ckpt`:

```powershell
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SSC --seeds 42 --resume 1
```

Training progress appears in the same terminal. Outputs are `results/valid/DTH_SE_adLIF/SSC/seed_42/` and `results/valid/MT_SE_adLIF/SSC/seed_42/`. The runner updates each model's `RESULTS.md`, evaluates the best validation-selected checkpoint on the official test set after training, and saves per-seed cost records by default. Dataset-specific configs now select loss automatically; no loss flag is needed. The separate `ssc_mean_comparison.py` launcher and standalone mean experiment configs are superseded for this comparison; do not use them to start these jobs.

The old diagnostic queue is retired after promotion and refuses to restart training. The explicit path alias in `config/path_aliases.yaml` resolves the moved reference without substituting any checkpoint. The source diagnostic report links to the promoted evidence. Historical summed-loss runs remain under `archive/ssc_summed_loss_20261019/`.

Promotion manifest: [original file inventory](../../results/provenance/ssc-mean-reference-promotion-2026-10-19.json). Checkpoints and datasets are ignored by Git and must be transferred separately to another checkout with the same relative layout. The local environment is `.venv`, whose package pins were checked; no global Python is used.
