# SHD overnight follow-up — 2026-10-13

The active nine-run LR/batch search continues in the user's terminal without
being restarted or modified. A separate queue waits for all nine completion
markers and for the source launcher to release its output-directory lock.
No additional GPU training runs concurrently with that search.

## Control audit

The historical seed-42 Hydra config and fresh r06 manifest match on architecture,
seed, LR, batch size, epochs, dropout, surrogate settings, temporal aggregation,
checkpoint metric, scheduler factor/patience and early-stopping settings.
The only dataset difference is `additional_test_set_validation: false` versus
an omitted value, whose constructor default is already false. This does not
explain the validation discrepancy. The main runners remain unchanged.

The clean r06 run was interrupted and resumed at epoch 20. The existing runner
seeds randomness on process startup; exact equivalence of resumed random data
ordering and dropout with an uninterrupted historical run is not established.
Treat the reproduction discrepancy as unresolved rather than evidence that
LR/batch settings alone caused it.

## Runtime estimate

At the audit snapshot, r06 completed 148 validation epochs in about 2,359
seconds (~16 seconds/epoch). r01's first 11 recorded validation epochs spanned
~621 seconds (~62 seconds per epoch interval). Batch-256 speed was not yet
measured; an intermediate estimate of ~31 seconds/epoch was used.

The remaining grid is approximately 12 hours if most runs stop around 150
epochs, or 26 hours if all remaining runs reach 300. These are estimates:
file timestamps can include interruption or startup overhead, batch-128 speed
was measured early, and early stopping depends on future validation behavior.
The source grid itself may exceed the available 18-hour window.

## Follow-up experiments

Two fresh runs use reference LR 0.01, batch 512, seed 42, and the same
architecture/preprocessing/early stopping/effectively constant LR:

| Trial | Dropout |
|---|---:|
| d25 | 0.25 |
| d35 | 0.35 |

The queue's admission window lasts 18 hours from its launch. It estimates each
300-epoch dropout run at 25 seconds/epoch plus a 25% margin and five minutes
of overhead (~2.7 hours). It starts a run only if that estimate fits the
remaining time. Otherwise it exits, leaving the experiments queued. It never
interrupts the active LR/batch grid. The deadline is an admission budget, not
a guarantee of stopping GPU work at an exact time if epoch speed changes.

Results: `results/experiments/shd_dropout_search/LEADERBOARD.md`.
Queue state: `QUEUE_STATUS.json` in that directory. Live output goes to
`queue.stdout.log` and `queue.stderr.log` there.

```powershell
# Validate without training or waiting.
.\.ven\Scripts\python.exe queue_shd_dropout.py --dry-run

# Start a new 18-hour waiting/admission window (only if no queue is running).
.\.ven\Scripts\python.exe queue_shd_dropout.py --hours 18

# Watch an already-running background queue.
Get-Content .\results\experiments\shd_dropout_search\queue.stdout.log -Tail 20 -Wait
```

The separate queue holds its own lock to reject duplicates. Once the grid
finishes, it also holds the grid's lock during follow-up training. Test data
is not evaluated in either dropout experiment; test evaluation is explicit
after validation-based selection.
