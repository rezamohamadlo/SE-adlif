# SA-MT seed-123 continuation: 80-epoch plateau search (v6)

Source: imported SHD* SA-MT epoch-188 validation checkpoint, seed 123.
Eight weights-only continuation trials, resetting optimizer and scheduler:
LR 0.0004 and 0.0001 crossed with dropout 0.10, 0.15, 0.25; two additional
LR 0.0004/dropout 0.15 trials with batch sizes 128 and 512 (otherwise 256).
Each trains 80 additional epochs, no early stopping, then tests its best
validation checkpoint. ReduceLROnPlateau monitors val_acc, mode max,
factor 0.9, patience 10 epochs, min LR zero. Reductions follow a plateau,
not a fixed starting epoch. Source epoch 188 is zero based; new counters
start at zero because this resets optimization rather than fully resuming.

Run from repository root with the local environment:
.\.ven\Scripts\python.exe -u search_samt_shd.py
Use --dry-run for non-training preflight. Queue proceeds automatically;
--pause-between-runs waits after each trial. Repeat an identical command
for trial resume. New output: results/experiments/samt_shd_seed123_epoch188_continuation_plateau80_v6.
Stop old launchers before starting this queue. Old evidence is preserved.
Test inspection for selection is exploratory; rankings stay validation-based.
