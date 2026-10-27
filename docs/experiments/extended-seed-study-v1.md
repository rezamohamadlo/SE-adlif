# Extended seed study v1

This study asks whether SA-MT improves on MT on ECG, and whether MT improves
on DTH on SHD*. It extends training-seed coverage from the historical seeds
42, 123, 456 to ten distinct seeds. The seven new runs per model are
validation-only research evidence. Historical test scores and these validation
scores cannot form a ten-seed test result or an independent confirmation.

The implementation starts from `8ac59d60447eeeb126ade175ea964213182635c2`.
Both PCs must use the same committed implementation and versioned configuration.
The frozen cohort is [config/seed_cohorts/extension_v1.yaml](../../config/seed_cohorts/extension_v1.yaml):

- Original seeds: `42, 123, 456`.
- Additional seeds, in execution order: `1009, 2027, 3037, 4049, 5051, 6067, 7079`.
- PC1: ECG, `MT_SE_adLIF` and `SA_MT_SE_adLIF`, 14 jobs.
- PC2: SHD* (dataset argument `SHD`), `MT_SE_adLIF` and `DTH_SE_adLIF`, 14 jobs.

## Frozen training configuration

The runner still inherits the dataset section of
[paper_baseline.yaml](../../config/paper_baseline.yaml), through the existing
comparison group. Its original `reproduction_policy.training_seeds` is unchanged.

| Setting | ECG | SHD* |
|---|---:|---:|
| Epochs | 400 | 300 |
| Batch | 64 | 256 |
| LR | 0.01 | 0.01 |
| Training-derived validation fraction | 0.05 | 0.20 |
| Dataset split seed | 42 | 42 |
| Early stopping | false | false |

Existing Adam, plateau scheduler, preprocessing, architectures and dataset-specific
losses remain selected by the existing versioned configs. Model differences remain
explicit. All new jobs start from random initialization at epoch zero with a fresh
optimizer; SA-MT never loads trained MT weights. A resumed job restores its own
`last.ckpt`, including optimizer/scheduler state. Training seed and split seed are
separate settings. Restoring a checkpoint does not guarantee the same random
data/dropout sequence as uninterrupted training.

## Commands

Run from each checkout's repository root with its own prepared `.venv` and local
dataset files. Git transfers code and small tracked records; it does not transfer
ignored datasets or weights. Check local dataset/cache locations before launching.
No shared filesystem or cross-machine absolute paths are required.

PC1:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF SA_MT_SE_adLIF --datasets ECG --seed-cohort extension_v1 --validation-only --logdir results/experiments/extended_seeds_v1/pc1_ecg
```

PC2:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF DTH_SE_adLIF --datasets SHD --seed-cohort extension_v1 --validation-only --logdir results/experiments/extended_seeds_v1/pc2_shd
```

Append `--dry-run` before launching to resolve and validate the 14-job plan without
training, inference, locks or experiment-data writes. Only the user launches training.
The commands display training progress in their terminal; each seed also retains
`out.log` and CSV metrics. They run models and seeds sequentially on each PC.

`--seed-cohort extension_v1` authorizes the ten frozen seeds and, when `--seeds`
is omitted, schedules only the seven additional seeds. An explicit `--seeds`
subset is allowed, including canonical seeds in these separate study directories.
Without this flag, only `42, 123, 456` are accepted, as before.

`--validation-only` sets `run_test=false`, enforces the full baseline budget and
validation selection, and uses independent completion and reporting paths.
It requires a repository-relative `results/experiments/...` output root.
The extension cohort requires validation-only mode. Training overrides, TTA and
test-cost backfill are rejected for this study. Ordinary runner behavior remains
unchanged without the new flags.

## Outputs and restart safety

Outputs use `<output_root>/<model>/<dataset>/seed_<seed>/`, where roots are:

```text
results/experiments/extended_seeds_v1/pc1_ecg/
results/experiments/extended_seeds_v1/pc2_shd/
```

Each seed retains:

- `FULL_RUN_CONFIG.json` and `.hydra/config.yaml`: resolved settings, baseline
  identity, cohort name and ordered ten-seed list, training and split seeds.
- `LAUNCH_ENVIRONMENTS.json`: Git commit, tracked source diff fingerprint,
  Python/package versions, OS, CUDA and GPU names, preserving resume environments.
- `ckpt/last.ckpt`, the validation-selected best checkpoint, and
  `logs/mlp_snn/version_0/metrics.csv`.
- `TRAINING_SUMMARY.md`, `VALIDATION_SUMMARY.md`, `VALIDATION_RESULTS.json` and
  `validation_completed.txt` after verified full-budget completion. Selected epochs
  are zero based; accuracies in JSON are fractions, displayed as percentages.

No test accuracy is calculated or written by this workflow. It never invokes
`trainer.test` or post-test cost measurement. `completed.txt` retains its historical
meaning of training plus final-test completion and is never written here.
The SHD additional test-validation loader is disabled.

To resume, rerun the exact original command. Matching completed validation jobs are
verified and skipped, without testing. Interrupted jobs restore their own
`ckpt/last.ckpt`. Configuration changes, missing recorded checkpoints, test artifacts
and unidentified existing output directories are rejected. `--resume 0` requires
empty seed directories. Never rename a different checkpoint to replace a missing one.
A startup failure before any checkpoint is not continuation; preserve its evidence
and choose a new explicit output root for a fresh restart. A crash between writing
the result JSON and completion marker resumes from last.ckpt and republishes completion.

An OS-managed `.extension.lock` serializes launchers sharing an output root.
It releases on process exit; do not remove a lock file to bypass a live launcher.
Keep the source commit and tracked source fingerprint fixed on resume. New code or
settings require a separate study/output root. Existing manifests are not rewritten.

## Transfer and aggregation

Stop copying until the relevant run is complete, or explicitly treat its snapshot
as incomplete. Copy PC1 and PC2 output trees to one checkout, preserving the original
relative layout. Never merge files into an existing seed directory. Compare SHA256
hashes on source and destination (`Get-FileHash -Algorithm SHA256 <file>`) before
accepting the transfer. Transfer the entire seed directory, including metrics,
manifests, environment records, completion marker, summaries and checkpoints.
Checkpoints remain required for local runner resume/skip checks; read-only aggregation
can use evidence-only copies without weights. It does not deserialize imported weights.
Shared transfer manifests belong in `results/provenance/`.

After both roots exist locally:

```powershell
.\.venv\Scripts\python.exe aggregate_extended_seeds.py --roots results/experiments/extended_seeds_v1/pc1_ecg results/experiments/extended_seeds_v1/pc2_shd --output results/experiments/extended_seeds_v1/aggregate_001
```

This writes `VALIDATION_AGGREGATE.json` and `VALIDATION_AGGREGATE.md` into a new
directory; choose a new suffix for a later snapshot. Imported roots may have
different local locations, while saved artifact paths preserve the original layout.
Roots must point directly to the directory containing the model directories.
Duplicate model/dataset/seed directories are rejected, even if identical. Configs are
recomposed against the committed configs; validation scores/selected epochs are
checked against CSV metrics. Mixed source revisions within a dataset comparison,
incompatible configs, test evidence and inconsistent completion records are rejected.

The default report expects seven seeds for each of four model/dataset groups,
including explicit missing/incomplete rows. It reports each completed seed's best
validation accuracy/selected epoch, mean, sample SD (`ddof=1`) and completed count.
Only matching completed seeds enter paired comparisons:

- ECG: SA-MT minus MT, in percentage points.
- SHD*: MT minus DTH, in percentage points.

For differences `d_i`, the 95% Student t interval is
`mean(d) +/- t(0.975, n-1) * sample_SD(d) / sqrt(n)`.
It assumes independent training seeds and approximately normal paired differences.
With fewer than two pairs the interval and sample SD are unavailable. Small samples,
one fixed validation split, exploratory comparisons and multiple comparisons limit
interpretation; intervals are descriptive and not multiplicity adjusted. No accuracy
is compared numerically with a paper test baseline in these validation reports.

`--include-canonical` adds the original three seeds to expected rows, but accepts
only cohort-tagged validation-only evidence under the same frozen configuration.
Historical test-completed runs cannot be imported as validation completions.
The original test-selected/exploratory evidence retains its original status.

## Eventual official-test evaluation

This implementation does not provide an automatic promotion to test evaluation.
Before any future official test work, freeze a separate written plan with model
configs, seed list, exact validation-selected checkpoint identities/hashes, exclusions,
paired comparisons and aggregation rules. Archive that plan and obtain an explicit
user request to evaluate those checkpoints with a separate evaluation workflow and
output root. Do not change `run_test`, create `completed.txt`, reuse this output tree,
or combine historical three-seed test scores with seven validation scores.

## Infrastructure verification

Tests use temporary config trees, saved metadata fixtures, mocked subprocesses,
static test-call guards and statistics checks. They perform no fitting, optimizer
steps or official-test evaluations. Existing runner/config/path tests are also run.
End-to-end GPU training and completion callbacks during real training remain for
user-launched experiments.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_extended_seeds tests.test_full_runner tests.test_runner_costs tests.test_runner_ssc_mean tests.test_mt_consistency_runner tests.test_paper_config tests.test_project_paths
git diff --check
```
