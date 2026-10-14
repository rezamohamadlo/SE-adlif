# Canonical SHD* reference replacement

The user requested replacement of the prior reference, rather than keeping
two active reference versions. The completed paper-baseline seed set was moved
from `results/experiments/shd_paper/multirun/2026-10-14/10-10-21/` into
`results/valid/reference/SHD/seed_<seed>/`.

| Seed | Current test accuracy | Best checkpoint epoch |
|---|---:|---:|
| 42 | 92.93% | 261 |
| 123 | 92.89% | 204 |
| 456 | 93.64% | 243 |

Current mean +/- sample SD: **93.15 +/- 0.42%**. Batch 256, LR 0.01,
dropout 0.15, 300 completed epochs, early stopping disabled. Validation is
20% of training with split seed 42; test uses the best validation checkpoint.

`REFERENCE_RESULTS.json` holds aggregate unrounded metrics. Each seed has
`REFERENCE_RESULT.json`, the saved runtime config, metric CSV, best/last
checkpoints and summary. Checkpoint bytes were retained unchanged; saved
callback/output paths reflect the original run location and are provenance,
not current filesystem locations. Metadata identifies the current relative
best-checkpoint path. The baseline version was assigned during promotion after
checking the completed run settings; these runs started before that metadata
field was introduced in the experiment config.

The former batch-512/early-stopped SHD tree is no longer in the project.
Only historical numerical context is retained in prior experiments'
`REFERENCE_BASELINE.json`. Their 93.42% seed-42 comparison must not be silently
replaced with the new 92.93% score. Prior proposed-model controlled-v2 SHD
results are unmatched until retrained with the paper baseline.

A temporary rollback copy of the removed tree was placed outside the project
under the Windows user Temp directory (`SE-adlif-retired-SHD-*`). It is not a
second project reference and is not tracked by Git.

The existing main runners and frozen controlled-v2 settings were not changed.
New fair comparisons must inherit `config/paper_baseline.yaml`. Historical
proposed-model artifacts and their measurements remain unchanged.
