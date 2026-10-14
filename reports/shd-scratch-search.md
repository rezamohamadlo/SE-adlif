# SHD reference: LR and batch search from scratch

This separate launcher uses the existing `run.py` without changing the training
runner, optimizer implementation, or shared evaluation settings. It reads the
reference architecture from `SHD_SE_adLIF` and preprocessing from the current
SHD evaluation protocol. It requires no reference checkpoint.

Every run starts with random weights and a fresh Adam optimizer. Only initial
LR and batch size vary. The original reference control is r06 and runs first.

| Trial | Initial LR | Batch size |
|---|---:|---:|
| r01 | 0.005 | 128 |
| r02 | 0.005 | 256 |
| r03 | 0.005 | 512 |
| r04 | 0.01 | 128 |
| r05 | 0.01 | 256 |
| r06 (control) | 0.01 | 512 |
| r07 | 0.015 | 128 |
| r08 | 0.015 | 256 |
| r09 | 0.015 | 512 |

Fixed: 300-epoch limit, dropout 0.15, gradient clipping 1.5, early stopping
patience 50/min delta 0.001, and reference ReduceLROnPlateau patience 9999/factor
0.9 (effectively constant LR). The validation split is 20% of training with
split seed 42; test data is not evaluated during search. Batch size is updated
in both the model config and data module. Equal epoch budgets imply different
optimizer-step counts for different batch sizes.

From the project root:

```powershell
# Validate all nine configurations without writing results or training.
.\.ven\Scripts\python.exe search_shd.py --dry-run

# Start all nine sequentially, with training seed 42.
.\.ven\Scripts\python.exe search_shd.py
```

Results: `results/experiments/shd_scratch_search/<trial>/seed_<seed>/`.
The root contains `LEADERBOARD.md` and `LEADERBOARD.json`, ranked by validation
accuracy and then loss. Reports update after each run. Refresh live progress:

```powershell
.\.ven\Scripts\python.exe search_shd.py --report
```

Repeat the same training command to skip completed runs and resume interrupted
runs from their own `last.ckpt`, including optimizer, scheduler and epoch state.
Only one training/evaluation launcher can own an output directory at a time.
An OS-managed lock rejects duplicate launches and releases when the process
exits. The lock file can remain after exit; its presence alone is not a lock.
Existing results with different configurations cause an error. Use a different
`--output` for a new search. The original reference results and earlier sweep
are preserved in their existing directories.

After choosing promising configurations by validation (example: r03 and r06),
confirm them on additional seeds while keeping the data split fixed:

```powershell
.\.ven\Scripts\python.exe search_shd.py --trials r03 r06 --seeds 123 456
```

The report includes validation means and sample SD across completed seeds;
compare configurations with matching seed sets. Select a final configuration
on validation, then explicitly test its best validation checkpoint for each
completed seed (example: r03):

```powershell
.\.ven\Scripts\python.exe search_shd.py --test-selected --trials r03 --seeds 42 123 456
```

This test command performs no training. It saves test results in the per-run
summary and `TEST_RESULTS.json` and refreshes the search leaderboard. Test
scores do not affect validation ranking. Seed-42 test accuracy can be compared
with the historical 93.42%; repeatability requires comparison across matching
reference seeds. A better configuration is not guaranteed.

LR scheduler changes are a later, separate experiment after the LR/batch
search; this first stage deliberately retains the original scheduler behavior.

Historical reference context: these experiments used the now-retired
controlled-v2 baseline. Their saved REFERENCE_BASELINE.json preserves that
comparison; the canonical SHD reference has since been replaced.
