# Recovery from overlapping launchers — 2026-10-13

Two search launchers were writing into the same output directories. Both were
stopped before recovery. Checkpoints were loaded and validation CSV histories
were audited:

| Trial | Validation rows | Repeated epoch numbers | Backward epoch transitions |
|---|---:|---:|---:|
| r06 | 174 | 79 | 0 |
| r01 | 297 | 120 | 118 |
| r02 | 218 | 75 | 36 |
| r03 | 72 | 0 | 0 |

r06, r01 and r02 were moved intact to
`results/experiments/shd_scratch_search_overlap_20261013-141632/` and will be
rerun from scratch. Their combined histories and completion markers are not
used for ranking. Original root reports were also copied into that archive.

r03's history is consistent. Its last checkpoint is epoch 71/global step 864,
so it will resume from that checkpoint when its turn arrives. No epochs are
reused from the overlapping histories of other configurations.

The separate launcher now holds an OS-managed output-directory lock to prevent
duplicate launchers. Existing training runners remain unchanged. The search
retains its current 300-epoch limit with early stopping; a 50-epoch staged
search has not been implemented.
