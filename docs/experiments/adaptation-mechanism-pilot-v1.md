# Adaptation mechanism pilot v1

Current workflow (12 November 2026): the completed local pilot has been
organized into `clean/`, `time_scaling/`, and `spike_deletion/` under
`results/experiments/adaptation_mechanism_analysis_local_recheck_v1/`.
Its `ARTIFACT_INDEX.json` maps preserved historical paths to the current
package. Do not rerun the legacy commands below into relocated directories.
See the [full-validation guide](adaptation-mechanism-full-validation-v1.md)
for current commands, sharing and verification. This document retains the
original pilot procedure; `command.txt` now prepares full validation.

This implements the first bounded SHD* validation analysis in the
[supervisor roadmap](../../supervisor-guideline-roadmap-fa.md) and
[three-PC daily plan](supervisor-three-pc-daily-plan-fa.md). It compares existing,
validation-selected AMT and MT checkpoints with frozen weights. It does not train,
access the official test set, use TTA, or start later roadmap stages automatically.

The frozen study definition is
[adaptation_mechanism_pilot_v1.yaml](../../config/studies/adaptation_mechanism_pilot_v1.yaml).
Its checkpoint and raw-data hashes are authoritative. Do not replace a missing
checkpoint with another epoch or seed. These are mechanism diagnostics using an
existing cohort, not a fresh fair classification training comparison.

## Prerequisites on each PC

Use the same code revision and the project's installed Python environment.
PC1 uses `.venv`; PC2 and PC3 use `.ven`. Each machine needs:

- `data/SHDWrapper/shd_train.h5` with the SHA-256 recorded in the study YAML.
  No test H5 is needed for this pilot.
- The validation-selected checkpoint and saved configuration/provenance for
  each of these six seed directories:

  ```text
  results/valid/MT_SE_adLIF/SHD/seed_42/
  results/valid/MT_SE_adLIF/SHD/seed_123/
  results/valid/MT_SE_adLIF/SHD/seed_456/
  results/not_improved/AMT_SE_adLIF/SHD/seed_42/
  results/not_improved/AMT_SE_adLIF/SHD/seed_123/
  results/not_improved/AMT_SE_adLIF/SHD/seed_456/
  ```

- The corresponding `COST_RESULTS.json` records and original training
  `metrics.csv` evidence used to verify each clean validation anchor.
- CUDA-capable PyTorch in that environment for the default `cuda:0` device.

Git pull transfers code and tracked metadata, but does not transfer ignored
checkpoints, raw data or training logs. Copy missing artifacts separately using
the same repository-relative layout. Verify the frozen identities; do not
overwrite a different file at the same path. The preflight reports missing or
incompatible artifacts before inference and does not download data.

## Check first, then launch

Run from the repository root. Append `--self-check --check-only` to the assigned
command first. For example, on PC2:

```powershell
.\.ven\Scripts\python.exe scripts/adaptation_analysis/pilot.py --pc pc2 --self-check --check-only
```

`--self-check` alone runs synthetic feature/transformation contracts and exits;
it does not load raw data or models. `--check-only` checks the frozen study's
local prerequisites without inference or writing a results directory. Combined,
they do both kinds of checks. The optional portable wrapper also supports this:

```powershell
.\run_adaptation_pilot.ps1 -PC pc2 -SelfCheck -CheckOnly
```

The wrapper selects `.venv` or `.ven` under its own checkout and works from
another working directory. It never uses a global Python installation.

After the checks pass, run exactly one assigned command on each machine:

```powershell
# PC1: clean feature association
.\.venv\Scripts\python.exe scripts/adaptation_analysis/pilot.py --pc pc1

# PC2: timing sensitivity
.\.ven\Scripts\python.exe scripts/adaptation_analysis/pilot.py --pc pc2

# PC3: spike-density sensitivity
.\.ven\Scripts\python.exe scripts/adaptation_analysis/pilot.py --pc pc3
```

The current [command.txt](../../command.txt) now contains full-validation commands. Historical
speaker-regularization commands remain in Git history and their experiment
guides; they are not part of this pilot. Check active processes and output paths
before launching; do not run duplicate workers for one PC's output directory.

## Bounded scope

### Local regeneration of PC2/PC3 (12 November 2026)

The user chose one physical PC to regenerate both short evaluation arms.
`--pc pc2` and `--pc pc3` select the timing and spike-density analyses;
they can run sequentially on the same machine using its `.venv` environment.
At the time of the local regeneration, `command.txt` supplied these two commands with
`--output-root results/experiments/adaptation_mechanism_analysis_local_recheck_v1`.
Both synthetic checks and prerequisite preflights passed locally; inference
has not been launched by the agent. Finish PC2 successfully before starting PC3.

This is an intentional repeat of the same frozen-checkpoint, 256-sample pilot
to produce coherent local detailed artifacts. It is not the full-validation
perturbation study and adds no independent training seeds. The imported outputs
and original local PC1 evidence remain separate. Each new arm will retain its
own manifest, detailed rows and completion hashes; verify those records before
comparing with imported summaries. The three-machine commands above document
the original arrangement; use the current local commands for this regeneration.

The defaults are **256 fixed validation samples**, the existing training seeds
**42, 123 and 456**, and one GPU worker per PC. Sample selection and the original
SHD* split seed 42 are fixed in the study definition. The CLI caps this diagnostic
subset at 512 samples. `--samples` and `--seeds` can make a smaller technical
check, but it must use a separate `--output-root`; it is not the complete cohort.

Every run first checks clean **full-validation** accuracy/loss against its saved
evidence. The full-validation anchor is a compatibility gate, not an expansion
of the perturbation study to all validation samples. A small subset's accuracy
must not be compared directly with a saved full-validation accuracy.

| PC | Arm | Fixed conditions |
|---|---|---|
| PC1 | Feature association | Clean input; short/long activity contrast and an autocorrelation-timescale proxy |
| PC2 | Timing sensitivity | Time scales 0.9, 1.0 and 1.1 |
| PC3 | Spike-density sensitivity | Deletion probability 0 and 0.1; three recorded realizations for nonzero deletion |

Paired models see the same transformed input. Transformations are evaluated
separately; predictions are not averaged. Deletion thins individual events in
count-valued frames, rather than deleting an entire occupied frame. Event count,
duration and rate before/after the transform are retained for interpretation.

The paired metric is `supervised_loss(MT) - supervised_loss(AMT)`; a positive
value favors AMT. Activity features exclude artificial padding. Feature
association remains exploratory: activity, duration and class can confound it.
Repeated transformations or samples do not create independent training seeds.

## Outputs and stopping point

The default output root is
`results/experiments/adaptation_mechanism_analysis_v1/`, with an isolated arm
directory for each PC. It does not write into existing training results.
Outputs include `MANIFEST.json`, `VALIDATION_ANCHORS.json`, `SUMMARY.json`,
`SUMMARY.md`, plots, `PROGRESS.json` and, after completion,
`PILOT_COMPLETE.json`. Detailed diagnostic rows and paired sample CSV files are
local artifacts; transfer them separately if they are ignored by Git.
To move or continue an analysis, copy the complete PC output directory,
including its ignored `diagnostic_rows/` chunks. Git summaries alone cannot
reproduce the completed-output verification or supply resumable inference data.

An identical rerun verifies the manifest and completed-output hashes before
skipping a complete pilot. An interrupted evaluation reuses complete,
identity-guarded inference chunks. This is evaluation continuation, not training
resume. Changed arguments, code or environment require a distinct output root;
do not bypass an identity mismatch by deleting guards.

Each command stops after its bounded pilot. There is no automatic full-dataset,
new-model, gradient or training queue. Review the three summaries and actual
runtime together before defining the next daily step. A day is a planning
interval, not a promise of a 24-hour runtime. Validation development does not
provide independent confirmation, and official-test history must remain a
limitation in later claims.
