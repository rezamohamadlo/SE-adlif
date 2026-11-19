# Full SHD* validation: AMT versus MT

Status (12 November 2026): user execution completed; 57 recorded output hashes,
full split coverage, prediction/feature identities, six clean anchors and saved
summary calculations verified. Runtime: approximately 5 minutes 52 seconds.
See the [post-run review](../../results/experiments/adaptation_mechanism_full_validation_v1/post_run_review/VERIFICATION.md).
Stage 5 adjusted exploratory analysis for this pair has since completed;
see the [statistics guide](adaptation-statistics-v1.md). The launch commands
below are retained for reproducibility; no rerun is required for this completed run.
This is frozen-checkpoint validation analysis, without training, official test
access, TTA, new checkpoint selection or an automatic next research stage.

The [versioned study](../../config/studies/adaptation_mechanism_full_validation_v1.yaml)
inherits the pilot's authenticated checkpoints and settings. It selects ALL
1,631 members of the original SHD* validation split (20%, split seed 42).
Training seeds remain 42, 123 and 456. Saved configurations and
`paper_classification_v1` baseline identities remain authoritative.

## Run locally

One machine was selected for this preparation. Analysis names and result
paths do not depend on the number of machines. Use the project's `.venv`.
From the repository root, check first:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_analysis/evaluate.py --check-only
```

Then launch the one bounded evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_analysis/evaluate.py
```

Both commands are in [command.txt](../../command.txt). The optional
`run_adaptation_validation.ps1` wrapper uses its own checkout and `.venv`, even
when invoked from another working directory. Append `-CheckOnly` for preflight.
The process stays in the current terminal; progress is also saved in
`PROGRESS.json`. Do not start another worker in the same output directory.
No precise runtime is promised from the smaller pilot's timing.

Prerequisites are the exact official training H5, six selected checkpoints,
their cost/config records and original training metrics, as listed in the
inherited pilot study. Preflight authenticates them locally; it does not
download data. Ignored artifacts must be transferred separately.

## Conditions and shared evidence

The evaluation computes clean input once for each model/seed. That prediction
is both the full-validation anchor and the shared comparison for every
intervention. It must match the saved validation evidence before proceeding.
Input features are computed once per condition and shared across models/seeds;
reconstructed input hashes are checked against the shared feature cache.

The fixed conditions remain clean, time scales 0.9 and 1.1, and event deletion
0.1 with three deterministic realizations. Scale 1 and deletion 0 use the shared
clean condition. Per-channel counts are preserved by warping, but duration,
density and temporal structure may change. Deletion changes rate, sparsity and
other temporal statistics together. These are not pure single-feature causal
interventions.

The output layout is independent of PC assignments:

```text
results/experiments/adaptation_mechanism_full_validation_v1/
    MANIFEST.json
    PROGRESS.json
    VALIDATION_ANCHORS.json
    COMPLETE.json
    diagnostic_rows/
        input_features_<condition>_rep_<replicate>.json
        <model>_seed_<seed>_<condition>_rep_<replicate>.json
        paired_samples.csv
    analyses/
        clean/SUMMARY.json, SUMMARY.md, benefit.png, benefit.svg
        time_scaling/...
        spike_deletion/...
```

`MANIFEST.json` saves the resolved study and each checkpoint's full saved
config, baseline identity, sample/split identity, seeds, source identities and
environment. New source hashes normalize CRLF to LF, so checkout line-ending
conversion alone does not invalidate new evaluations. Historical pilot hashes
remain unchanged. This does not excuse actual code/environment changes:
resuming requires an identical effective manifest; changed settings require
a distinct `--output-root` under `results/experiments/`.

Interrupted evaluations reuse identity- and hash-checked prediction chunks.
This is evaluation continuation, not training resume. No optimizer is created.
One input condition's tensors are held at a time. Clean predictions and shared
features are not recomputed separately for each report.

Reports show per-seed/per-condition/per-replicate values, means and sample SD
across training seeds. A separate effect table first averages perturbation
replicates inside each seed, then compares transformed benefit with clean
benefit. Three perturbation replicates do not create three training seeds.
Correlations remain descriptive; controlled Stage 5 statistical inference is
still pending. No claim of independent final-test confirmation is made.

## Share and verify

Transfer the ENTIRE result directory with its repository-relative layout,
including ignored `diagnostic_rows/`. Git transfers the summaries and tracked
metadata, not the detailed rows. Keep checkpoints and original data separately
when continuing evaluation; completed output verification needs only the
complete result package and installed analysis dependencies.

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_analysis/evaluate.py --verify-only
```

Verification does not load checkpoints, require a GPU or read datasets. It
checks completion identity and all recorded output hashes. Never regenerate a
missing chunk from a different checkpoint or remove a guard to force reuse.

The completed 256-sample local pilot is now self-contained under
`results/experiments/adaptation_mechanism_analysis_local_recheck_v1/`, with
`clean/`, `time_scaling/`, `spike_deletion/` and the two historical review
directories. Its `ARTIFACT_INDEX.json` resolves immutable historical paths to
the new layout. The original archive is not needed to verify this package.
Historical labels inside original manifests/reviews remain provenance only.

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_analysis/evaluate.py --verify-only --output-root results/experiments/adaptation_mechanism_analysis_local_recheck_v1
```

After the user runs the full evaluation, verify its completion, review the
full-validation effects, and update the roadmap before starting Stage 5 or
adding more models/datasets.
