# SHD reference hyperparameter sweep

Target: SHD SE_adLIF, reference seed 42, reported final test accuracy 93.42%.
The retained best checkpoint is `epoch=162-step=1956.ckpt`, with 99.14%
validation accuracy, under
`results/valid/reference/SHD/seed_42/hydra/2026-10-12/07-17-03/ckpt/`.
It is the runner's default checkpoint. The Markdown summary names epoch 120
and 98.96%, which differ from the actual checkpoint metadata; use the saved
checkpoint and CSV metrics for source provenance.

The screening runner starts every trial from exactly the same reference weights.
It resets Adam, scheduler state and epoch numbering, then trains for 10 epochs
(or 5 with `--epochs 5`). This is weights-only continuation, rather than a full
Lightning resume that would restore the previous learning rate and epoch count.
Interrupted individual trials resume their own optimizer and scheduler states.

There are 20 trials: LRs 0.0001, 0.0003, 0.001, 0.003 and 0.01, each crossed
with these four recipes:

| Recipe | Batch | Scheduler |
|---|---:|---|
| A | 512 | Constant LR |
| B | 256 | Constant LR |
| C | 512 | ReduceLROnPlateau, validation accuracy, patience 2, factor 0.5 |
| D | 512 | Cosine decay, minimum LR = 10% of initial LR |

Architecture, dropout and preprocessing come from the saved checkpoint config.
The original training-derived 20% validation split and split seed 42 stay fixed.
Both batch size settings are updated together. Screening disables early stopping
to give each candidate the complete budget, and disables all test evaluation.
Gradient clipping stays at 1.5. Validation of the unmodified reference weights is
recorded before each trial. Smaller batches perform more updates per epoch;
these comparisons use equal epochs, not equal optimizer steps or GPU time.

From the project root:

```powershell
# Inspect all 20 recipes without loading a checkpoint.
.\.ven\Scripts\python.exe tune_shd.py --list

# Use the retained epoch-162 checkpoint (or supply --checkpoint explicitly).
.\.ven\Scripts\python.exe tune_shd.py --epochs 10 --dry-run
.\.ven\Scripts\python.exe tune_shd.py --epochs 10
```

Runs execute sequentially on the configured device. Results go to
`results/experiments/shd_lr_sweep/c01/` through `c20/`; the original checkpoint
is read only. `LEADERBOARD.md`, `LEADERBOARD.json` and `BEST_CONFIG.json`
update after each completion. Rank uses validation accuracy, with validation
loss as a tie breaker; test accuracy is never used. Negative deltas mean a trial
failed to improve the starting reference. The best screened recipe may still
fail to improve the baseline, so inspect the delta before selecting it.

Repeat the identical command to skip completed trials and resume interrupted
trials from `last.ckpt`. Changing epoch budget, recipe or source checkpoint
requires a separate `--output` directory. Use `--trial c01` to run just one
candidate. `--report` refreshes the leaderboard without training.

Short continuation measures fine-tuning performance near a converged solution;
it cannot establish the best LR or scheduler for training from scratch.
Confirm promising recipes with fresh full runs, ideally the top 2–3 plus the
existing baseline, using seeds 42, 123 and 456 before claiming an improvement.
For example, if c11 is chosen:

```powershell
.\.ven\Scripts\python.exe tune_shd.py --full-train --trial c11 --dry-run
.\.ven\Scripts\python.exe tune_shd.py --full-train --trial c11
```

Confirmation initializes weights and Adam from scratch, uses a 300-epoch limit,
restores the reference early-stopping settings, and stretches cosine decay over
the full budget. Results are isolated under `full_train/c11/seed_<seed>/`.
Test evaluation occurs once after validation checkpoint selection in each
full run. Select the final recipe using validation results, then report test
accuracy and variability across seeds. There is no guarantee of exceeding 93.42%.
