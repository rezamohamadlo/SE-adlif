# SHD* speaker holdout with augmentation and dropout

Current physical artifacts are consolidated under `results/experiments/shd_speaker_combined_v1/artifacts/`. Legacy paths resolve through `config/path_aliases.yaml`; see the [consolidation record](../transfers/shd-speaker-consolidation-2026-11-10.md).

Protocol `shd_speaker_augmented_v1`, implementation 1.0.0. This is a new,
fresh-training DTH/LG-DTH experiment. The retired
[speaker-holdout experiment](shd-speaker-holdout.md) and its checkpoints remain
archived. Both LG-DTH local slow and transient corrections stay enabled.

The shared `runner.py` handles preflight, sequential training, full resume,
validation-selected final testing, aggregation and cost measurement. The main
runner, `run.py`, model implementations, paper baseline and ordinary comparison
defaults are unchanged. The new opt-in profile extends the existing speaker
workflow and SHD data module.

## Fixed settings

Inherit SHD's dataset-specific settings from `config/paper_baseline.yaml` through
`config/comparison/SHD_speaker_holdout_v1.yaml`. Apply the separate
`config/shd_speaker_study/augmented_v1.yaml` overlay to both models:

| Setting | Value |
|---|---|
| Training mode | Random weights, fresh optimizer, epoch 0 |
| Training seeds | 42, 123, 456 |
| Split seed | 42, independent of training seed |
| Training speakers | 1, 2, 6, 7, 8, 9, 10, 11: 6,521 recordings |
| Validation speakers | 0, 3: 1,635 recordings (20.046591% of official train) |
| LR / batch / epoch budget | Baseline 0.01 / 256 / 300 |
| Early stopping | Disabled; full 300 epochs |
| Loss / selection | Baseline `sum_softmax_over_time` / maximum validation accuracy |
| Dropout | 0.25 in both hidden layers (baseline 0.15) |
| Time scale | Uniform 0.9–1.1 per training recording |
| Channel shift | Uniform integer −2…+2 of the 140 downsampled channels |
| Spike deletion | Independent probability 0.05 per spike |
| Validation / official test | Deterministic ordinary inputs, no augmentation or TTA |

These regularization strengths are user-approved starting settings, not tuned
results. Freeze this recipe before comparing the two models; use a new protocol
version and output root if changing settings after any run begins.

## Split and augmentation behavior

Use the original SHA-256 ranking of training speaker IDs with split seed 42;
it selects IDs 0 and 3 without consulting model accuracy. Preflight requires
exactly eight training and two validation speakers and all 20 classes in each
subset. The unchanged split algorithm and dataset-content fingerprint are
`shd_speaker_holdout_v1` metadata; the experiment/training protocol is separately
identified as `shd_speaker_augmented_v1`. Keeping the same split makes the
historical and new validation subsets identical without reusing trained weights.

Augmentation applies to count frames **after deterministic disk-cache reads**.
Time scaling maps whole frame indices with `floor(index * scale)` and adds
colliding counts. It preserves spike counts before spatial boundary removal
and deletion; it approximates time warping at the baseline's 4 ms bin resolution.
It does not duplicate spikes through interpolation. Shifted spikes outside the
channel range are discarded, without wraparound. Binomial thinning deletes
spikes independently even when a bin contains multiple spikes. Rebuild the
block mask to match the resulting length, retain the first-ten-timestep ignore
rule, and pad to at least the baseline 250 timesteps.

The label and original cached sample are preserved. Every training read gets
a fresh augmentation draw from PyTorch's seeded RNG. `run.py` already seeds
PyTorch and workers using each training seed. Full resume restores training
state but does not promise the same subsequent random augmentation/dropout
sequence as uninterrupted training. Mini-batch order still shuffles within the
eight training speakers; the train/validation partition never shuffles recordings
between speakers. Validation and test loaders never apply the training transform.

Event deletion and spatial perturbations are established event-data operations;
see the [Tonic transform reference](https://tonic.readthedocs.io/en/develop/autoapi/tonic/transforms/index.html).
The local frame-based implementation and numerical strengths above are explicit
experimental choices, not a claim of paper reproduction or improved accuracy.

## Commands

Run from the repository root in PowerShell using the project's virtual environment.
All training is launched by the user. Check all six jobs without training,
inference, downloads or output writes:

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase all --shd-speaker-profile augmented_v1 --dry-run
```

Launch all jobs sequentially in a visible terminal:

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase all --shd-speaker-profile augmented_v1
```

Order: **DTH-42 → LG-DTH-42 → DTH-123 → LG-DTH-123 → DTH-456 → LG-DTH-456**.
Every job completes validation-selected official testing and cost measurement
before the next begins. A failure stops the queue. For separate phases, replace
`all` with `1` for both seed-42 runs, then `2` for seeds 123 and 456. Phase 2
requires both phase-1 training/test completions.

Rerun the same command after interruption: completed runs are skipped and an
interrupted run resumes from its own `ckpt/last.ckpt`, including optimizer and
scheduler state. Final test uses the best validation-selected checkpoint.
For a cost-only failure, add `--measure-existing-costs` to the same phase command;
training/test remain complete and are not repeated to repair measurement.

## Evidence and isolation

Outputs: `results/experiments/shd_speaker_combined_v1/artifacts/shd_speaker_augmented_v1/<model>/SHD/seed_<seed>/`.
Each seed records its `FULL_RUN_CONFIG.json`, Hydra config, split manifest,
training metrics/summary, best/last checkpoints, and `COST_RESULTS.json`/`.md`.
The shared split manifest belongs at the experiment root. Each model's
`RESULTS.md` reports per-seed final test accuracy, completed-seed count, mean
and sample SD. Missing tests stay pending. Baseline version, split/training
seeds and regularization deviations are saved in each resolved run config.

Caches use `data/cache/SHD/shd_speaker_augmented_v1_<preprocessing hash>/<split fingerprint>/`
with separate train/val/test subdirectories. The new output root has its own OS
launcher lock. No archived run, current baseline result or legacy cache is reused
as new training evidence. Dataset files/archives must exist locally; preflight
never downloads. Manifests and artifact paths remain repository-relative and
resolve through `project_paths.py` when Hydra changes working directory.

Report this as **SHD* speaker holdout + augmentation**, separately from the
random-recording SHD* paper comparison. The validation split and regularization
differ, so the paper's accuracy is not a matched numeric baseline for this study.
Do not use final-test scores to tune this recipe or pick a seed.
