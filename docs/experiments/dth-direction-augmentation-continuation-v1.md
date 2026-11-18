# User-requested training-distribution change: Direction-aware DTH + R-Drop

The user explicitly requested stopping B and continuing its own training with
time scaling0.9–1.1, channel shifts within±2 and independent5% spike deletion.
The original study stopped after the saved41st epoch (1025 optimizer updates).
There was no runtime graceful-stop hook; its orchestration was disabled, the
next complete last checkpoint was verified, then only its own processes were
terminated. Any work after that checkpoint is not counted as completed.

Continuation is isolated at
`results/experiments/dth_complement_v1/direction_aug_shd100_v1/`.
It resumes the run's last weights, optimizer, scheduler and epoch/step state;
this is a distribution-changing full-state continuation, not fresh training,
not weights-only fine tuning and not an unchanged-protocol resume. Stop at100
TOTAL epochs, retaining the original300-epoch schedule. Seed42, splitseed42,
batch256, LR0.01, dropout0.15, R-Drop alpha1 all remain fixed.

The existing count-frame augmenter runs only in training after deterministic
cache reads. Each sample is augmented once, then both R-Drop forwards see the
same augmented sample and independent dropout. Time scaling re-bins complete
time bins (frame-level approximation, not sub-bin event warping); collisions
sum counts. Shifts are at the140-channel model-input resolution; boundary
events are discarded, never wrapped. Deletion uses binomial thinning of counts,
so bins with multiple spikes receive independent per-spike thinning. Original
labels are retained and temporal block masks are rebuilt. Validation and test
retain the exact original deterministic transforms and masks. No TTA.

The same canonical SHD* fitting/validation sample indices are retained; no
official test sample enters training or validation. Preserve old checkpoints,
configs, metrics and intermediate tests, with explicit lineage and all config
changes recorded. Derived checkpoint metadata and callback paths are rebound
to the new isolated output; tensor/optimizer/scheduler/loop hashes must match
the source. The old best validation checkpoint remains eligible across the
whole continuation history; identify whether a selected epoch predates
augmentation. Do not claim an independently trained augmentation ablation or a
budget-matched causal gain. Tests are exploratory because already exposed.

Use runner.py exclusively for training/resume. Stop after100 and one frozen
validation-selected ordinary test. No new ideas, searches, commits or pushes.
