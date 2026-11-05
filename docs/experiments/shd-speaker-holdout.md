# SHD speaker holdout v1

**Retired at the user's request on 5 November 2026.** Results, checkpoints and
the incomplete run are preserved in the [archive](../../results/not_improved/shd_speaker_holdout_v1/README.md).
The phase commands below are historical; the launcher now rejects this retired
workflow. An intentional rerun requires a new experiment version and output root.

Protocol `shd_speaker_holdout_v1`; original run implementation **1.0.0**;
retirement guard version **1.0.1**.
See the [changelog](shd-speaker-holdout-changelog.md).

This opt-in workflow compares the existing DTH and LG-DTH architectures using
two entire training speakers for validation. Both LG-DTH local slow and local
transient corrections remain enabled in both hidden layers. It inherits
`config/comparison/SHD.yaml` and `config/paper_baseline.yaml` through
`config/comparison/SHD_speaker_holdout_v1.yaml`. The resolved baseline is
`paper_classification_v1`: 300 epochs, batch 256, LR 0.01, no early stopping,
`sum_softmax_over_time`, validation-accuracy checkpoint selection. Other
optimization, preprocessing and architecture settings are checked against each
ordinary model's fully resolved comparison config. Ordinary inference uses no TTA.

## Split definition and identity

Read `extra/speaker` and `labels` directly from `SHDWrapper/shd_train.h5`.
They must be one-dimensional integer arrays aligned to the rows of
`spikes/times` and `spikes/units`. Verify the installed wrapper's `speaker`
property against the same HDF5 array and index space. No speaker identity is
inferred from labels or recording order. Preserve the complete official
`shd_test.h5` for final testing.

Algorithm `sha256_seed_speaker_rank_v1` sorts unique integer speaker IDs in
ascending numeric order, ranks each by the hexadecimal SHA-256 of the exact
UTF-8 string `sha256_seed_speaker_rank_v1:42:<integer ID>`, breaking ties by ID,
and holds out the first two. Recording indices within each subset retain their
original ascending HDF5 row order. Split seed **42** is independent of training
seeds **42, 123, 456**. Selection never consults model performance. If either
subset lacks any class 0–19, or metadata is unavailable/incompatible, preflight
fails without trying another speaker pair or falling back to random splitting.

The frozen [split manifest](../../results/not_improved/shd_speaker_holdout_v1/SPLIT_MANIFEST.json)
stores speaker IDs, original recording indices, per-class counts, actual sizes,
validation fraction, split seed, algorithm version and dataset identity. Its
fingerprint hashes canonical sorted-key compact JSON excluding the fingerprint
field itself. Dataset identity includes SHA-256 of the complete training and
test HDF5 files and aligned training speaker/label metadata. Physical paths are
excluded from the fingerprint so identical transferred data remain compatible.
Preflight and datamodule construction recompute and compare this identity.

Caches live below `data/cache/SHD/shd_speaker_holdout_v1_<preprocessing hash>/<split fingerprint>/`;
`train`, `val` and `test` have distinct subdirectories. Both split identity and
preprocessing settings enter the namespace. Subset-local indices cannot reuse
the historical random-split cache. Unused legacy caches remain in place.

The installed Tonic loader also checks the presence of the original
`shd_train.h5.zip` and `shd_test.h5.zip` beside the extracted files. Preflight
checks these prerequisites and never downloads data.

## User commands

Run these PowerShell commands from the repository root. All training must be
launched by the user. The dry run hashes the actual local dataset, validates
metadata/class coverage, resolves all six configurations and checks existing
output/resume guards. It performs no training, inference, downloads or writes.

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase all --dry-run
```

Phase 1 runs DTH seed 42, then LG-DTH seed 42:

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase 1
```

After Phase 1 completes, Phase 2 runs DTH seed 123, LG-DTH seed 123,
DTH seed 456, then LG-DTH seed 456:

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase 2
```

Each phase may also be checked individually by appending `--dry-run`.
Phase 2 training requires completed, verified Phase 1 runs. The `all` phase
executes the same six jobs in order. No training overrides are accepted by this
frozen workflow. Change the version and output root for a different protocol.

## Outputs and recovery

All new evidence stays under
`results/experiments/shd_speaker_holdout_v1/` during execution; the tree is now
archived intact under `results/not_improved/shd_speaker_holdout_v1/`. Each run uses
`<model>/SHD/seed_<seed>/`, with its resolved `FULL_RUN_CONFIG.json`, a verified
`SPLIT_MANIFEST.json` copy, Hydra configuration, metrics, training summary,
checkpoints and per-seed `COST_RESULTS.json` / `.md`. Each model's `RESULTS.md`
reports final test accuracy, sample SD, completed seed count and runner cost
aggregates, using the distinct label **SHD speaker holdout v1**. Missing final
tests remain pending. Shared split and non-training preflight evidence stay at
the experiment root. Existing comparison reports and historical results remain
separate.

The shared runner performs preflight, training, checkpoint selection, final
testing and cost measurement. Fresh runs start with random weights and a fresh
optimizer at epoch 0. Rerunning the same phase skips completed training and
fully resumes interrupted jobs from their own `ckpt/last.ckpt`, including
optimizer/scheduler state. Final evaluation uses the best speaker-validation
checkpoint, which may differ from `last.ckpt`. Resuming compares the entire
effective config and checkpoint hyperparameters, protocol, split fingerprint,
model, training seed and dataset identity. Missing recorded checkpoints are
errors; another checkpoint is never substituted. A completed marker requires
the full epoch budget, its validation-selected checkpoint and finite test
metrics. Conflicting test histories are rejected.

Both phases hold the same OS-managed `.extension.lock` for their full duration
and wait for each child process in the visible terminal. An overlapping launcher
fails immediately. Stale lock files are harmless and must not be deleted to
bypass a live process. Ordinary `runner.py --logdir` cannot target this protected
experiment root without its protocol dispatch. Exceptions stop the queue.

Cost failures preserve training completion and stop this queue if no cost record
was produced. Retry the same phase with `--measure-existing-costs`; completed
training is skipped, so measurement repair does not retrain a model. A failure
before any resumable training checkpoint is a failed fresh start. Preserve its
artifacts for diagnosis; the runner refuses to overwrite an unrecognized or
non-resumable history. Do not call that case checkpoint continuation.

Repository artifacts are stored with relative paths and resolved with
`project_paths.py`, including when Hydra changes the working directory. Copy
the same relative results layout and exact dataset/checkpoint files to another
checkout. Git alone does not transfer ignored datasets or weights. External
dataset storage remains explicit in the resolved configuration.

## Scientific scope and verification

Held-out speakers define model selection here. The actual validation fraction
depends on speaker recording counts and need not equal 20%. This differs from
random-recording **SHD*** and does not exactly reproduce the official test set's
mixture of familiar and unfamiliar speakers. Keep its results in their own
protocol; do not treat the paper's SHD* accuracy as a matched numeric baseline.
One fixed speaker pair does not establish robustness across arbitrary held-out
speaker choices. Matching three training seeds does not establish statistical
significance or guarantee a test improvement.

Non-training tests use synthetic HDF5 metadata and trusted synthetic checkpoint
metadata; they never call `Trainer.fit` or optimizer steps. They cover split
determinism, numeric ordering and index alignment, disjointness, class coverage,
manifest/dataset mutation, cache isolation, installed-loader compatibility,
legacy split behavior, portable paths/Hydra-style working-directory changes,
full-config and checkpoint guards, phase ordering/failure stopping, locking,
aggregation and output isolation. Actual-data checks and their frozen split are
recorded in the [preflight evidence](../../results/not_improved/shd_speaker_holdout_v1/PREFLIGHT.json).
That evidence describes the original pre-training check. Subsequently the user
completed both seed-42 jobs, including final testing/costs, and partially ran
DTH seed 123 before discontinuing the experiment. See the archive for the
final inventory. No agent launched training or a training smoke test.
