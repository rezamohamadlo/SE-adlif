# SHD speaker holdout: regularization tuning v1

Current physical artifacts are consolidated under `results/experiments/shd_speaker_combined_v1/artifacts/`. Legacy paths resolve through `config/path_aliases.yaml`; see the [consolidation record](../transfers/shd-speaker-consolidation-2026-11-10.md).

An explicit opt-in workflow through `runner.py` for DTH and LG-DTH. The three
candidates inherit `config/shd_speaker_study/augmented_v1.yaml`, which overlays
`config/comparison/SHD_speaker_holdout_v1.yaml` and the dataset-specific
`config/paper_baseline.yaml` settings. The regularization overlays do not
duplicate the learning rate, batch, epoch budget, schedule or preprocessing.

| Profile | Dropout | Spike deletion | Time scaling | Channel shift |
|---|---:|---:|---|---:|
| Existing `augmented_v1` baseline (reused) | 0.25 | 5% | 0.9–1.1 | 2 |
| `dropout030_v1` | 0.30 | 5% | 0.9–1.1 | 2 |
| `dropout035_v1` | 0.35 | 5% | 0.9–1.1 | 2 |
| `spike_delete010_v1` | 0.25 | 10% | 0.9–1.1 | 2 |

Each candidate changes one regularizer relative to `augmented_v1`. Augmentation
is stochastic and applies only to training, after clean frames are cached.
Validation and test inputs use ordinary preprocessing without augmentation.
The clean preprocessing/split cache namespace remains `shd_speaker_augmented_v1`:
these candidates share the exact clean frames, while the regularization config
and versioned output directories are distinct. No existing cache is deleted.

The speaker split is frozen at split seed 42, validation speakers `[0, 3]`,
fingerprint `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
It holds out entire speakers from the official training set. The official test
set remains separate. This protocol differs from the random recording split
called SHD* in baseline comparisons; a paper accuracy is not a matched baseline.

All jobs use fresh random weights and a fresh optimizer at epoch zero: 300
epochs, batch 256, LR 0.01 and the inherited schedule, with no early stopping.
Validation runs every epoch. The checkpoint with maximum validation accuracy
is saved; interrupted jobs fully resume only their own `ckpt/last.ckpt`.
Both LG-DTH local corrections remain enabled. This is regularization tuning,
not a new matched paper comparison.

## Launch sequence

Run from the repository root, using the existing `.ven` environment in this
checkout (`.venv` may be the environment name in another checkout). All training
is launched by the user. The new candidates can run on another system while
the existing `augmented_v1` baseline finishes. Missing baseline results stay
pending and do not block training. Within one checkout, an active baseline
launcher retains its lock; the new queue refuses to overlap it or write to its
result directories.

```powershell
.\.ven\Scripts\python.exe runner.py --shd-speaker-phase all --shd-speaker-profile regularization_v1
```

There are 18 jobs. First, seed 42 runs for dropout 0.30, then dropout 0.35,
then spike deletion 10%; each variant runs DTH followed by LG-DTH. Next, dropout
0.30 runs seeds 123 and 456 for both models, then dropout 0.35 does the same,
then spike deletion 10% does the same. The existing baseline is not retrained.
All runs use a visible terminal with the normal training progress bar.

The same command skips verified completed jobs and resumes interrupted jobs.
A failed startup without a checkpoint cannot resume; preserve its artifacts
and investigate instead of overwriting them. Errors stop the queue immediately.

To pause for review after all seed-42 candidates, launch phase 1 instead of
`all`, then phase 2 later:

```powershell
.\.ven\Scripts\python.exe runner.py --shd-speaker-phase 1 --shd-speaker-profile regularization_v1
.\.ven\Scripts\python.exe runner.py --shd-speaker-phase 2 --shd-speaker-profile regularization_v1
```

Phase 2 requires all six phase-1 validation completions. Do not launch these
commands in parallel. The shared OS lock covers all variants and phases;
stale lock files do not grant access and must not be deleted to bypass a lock.

Appending `--dry-run` checks actual dataset files, split identity, all effective
configs, outputs, completion and resume metadata without training, downloads,
inference or writes. It may inspect the current baseline while it is active.

## Outputs and validation selection

All new evidence stays below
`results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_regularization_v1/<profile>/<model>/SHD/seed_<seed>/`.
Each run stores `FULL_RUN_CONFIG.json`, the baseline identity/deviations,
the frozen per-run split, Hydra config, metrics, source/environment provenance,
best/last checkpoints and `VALIDATION_RESULTS.json`/`VALIDATION_SUMMARY.md`.
Completion uses `validation_completed.txt`; no test completion marker is made.
Artifact paths are repository-relative and resolve against the checkout root,
including when Hydra changes the working directory. Transfer the exact dataset
and checkpoint files with the same relative layout; Git does not transfer them.

The experiment root contains the frozen execution order in `QUEUE.json` and
an automatically updated `VALIDATION_COMPARISON.md`. This table includes the
existing augmented baseline and only completed-seed best validation accuracies.
It reports means, sample SDs and counts separately for each model and variant.
Once all 18 new jobs finish, `VALIDATION_SELECTION.json` selects the maximum
three-seed mean best validation accuracy per model, including the existing
baseline as an eligible choice, provided all six baseline results are also
available and verified. Exact ties are retained. No winner is published from
an incomplete set of seeds or while a baseline result is missing.

## Running the new candidates on another system

Pull the code, use that checkout's Python environment and retain the matching
local SHD dataset. No baseline model weights initialize the new candidates;
every new job starts fresh. A successful dry run confirms the local dataset
identity. Git does not transfer local baseline metrics or checkpoints.

After the baseline queue finishes on the original system, copy its completed
evidence into the same repository-relative layout on the new system:
`results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_v1/<model>/SHD/seed_<seed>/` for
both models and all three seeds. For validation comparison, each seed requires
`FULL_RUN_CONFIG.json`, `SPLIT_MANIFEST.json`, `completed.txt` and
`logs/mlp_snn/version_0/metrics.csv`. Keep the best/last checkpoints on the
original system for future evaluation/resume; weights need not be copied just
to compute the validation comparison. Copy a run only after it finishes, and
complete the transfer before asking the new queue to inspect its completion
marker. Never replace a missing artifact with another seed or checkpoint.

The comparison checks the entire resolved baseline config, the exact speaker
split and all 300 validation epochs. Baseline test scores are not used in
selection. The next completed candidate automatically refreshes the comparison.
If the new queue has already finished, rerun its original command to refresh
the comparison; verified completed candidates are skipped without training.

These are validation-only tuning runs: `run_test=false` and
`validation_only=true`. Test metrics and test artifacts are rejected. No
official-test inference or test-cost measurement is performed by this queue.
After settings are selected, prepare a separate final evaluation/confirmation
workflow through the shared runner; do not manually inject test results into
the tuning directories. No training is needed to repair later cost evaluation.

The candidates were proposed after observing earlier test results, so future
official-test reuse is not a wholly independent discovery experiment. One fixed
speaker pair and three training seeds also do not measure variability across
speaker splits. New confirmation seeds can assess initialization robustness.

## Non-training verification

The targeted tests use synthetic metadata and checkpoint dictionaries; they
never fit a model or take optimizer steps:

```powershell
.\.ven\Scripts\python.exe -m unittest tests.test_shd_speaker_regularization tests.test_shd_speaker_holdout tests.test_shd_speaker_augmented tests.test_extended_seeds
git diff --check
```
