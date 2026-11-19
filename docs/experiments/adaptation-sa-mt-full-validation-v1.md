# SA-MT versus MT: full SHD* validation

User execution completed 12 November 2026. Read-only artifact checks matched
the completion identity, all 58 output hashes, all 24 reused cache value hashes,
full prediction coverage and the six saved clean-anchor tolerances. See the
[post-run review](../../results/experiments/adaptation_sa_mt_full_validation_v1/POST_RUN_REVIEW.md).
The agent ran no inference, statistics or tests. No evaluation rerun is needed;
SA-MT-specific adjusted statistical analysis is implemented and preflight
checked, awaiting user execution; see the [statistics guide](adaptation-sa-mt-statistics-v1.md).

The [versioned study](../../config/studies/adaptation_sa_mt_full_validation_v1.yaml)
inherits the same full-validation definitions and paper-baseline authority as
the previous pairs. It pins validation-selected MT checkpoints (epochs
199/137/107 for seeds 42/123/456) and the original baseline SA-MT checkpoints
(188/188/99). SA-MT records are under `results/experiments/sa_mt_v1/`.
The separate `results/not_improved/samt_shd_first_trial_fresh_v1/` tuned trial
has different LR/dropout and is deliberately excluded. No model/checkpoint is
chosen using this analysis's test or validation outcome.

All 1,631 validation samples, split seed 42, training seeds 42/123/456, feature
definitions and perturbations remain fixed: clean, time scales .9/1.1 and
10% event deletion with three deterministic realizations. The hypothesis is
that independently learned fast/slow spike-adaptation strengths may have a
different benefit for transient versus sustained inputs. It is not assumed
true. These are frozen-model diagnostics, without training, TTA, official-test
access, tuning or automatic statistical stages.

## Run locally

From the repository root, using its `.venv`:

```powershell
# Optional preflight: no inference or result writes.
.\.venv\Scripts\python.exe scripts/adaptation_sa_mt/evaluate.py --check-only

# USER LAUNCH: reused MT records + new SA-MT evaluation.
.\.venv\Scripts\python.exe scripts/adaptation_sa_mt/evaluate.py

# USER LAUNCH after completion: verify the self-contained result package.
.\.venv\Scripts\python.exe scripts/adaptation_sa_mt/evaluate.py --verify-only
```

The same commands are in [command.txt](../../command.txt). This is a short
single-machine evaluation, with PC-independent result paths. Terminal output
and `PROGRESS.json` expose progress. One worker owns the output lock; do not
start another worker for this directory. Preflight works from a different
current working directory and authenticates exact checkpoint bytes, saved
configs, selected epochs, original metrics, data, strict state loading and
model parameter identity without a forward pass.

## Reuse and provenance

The original completed AMT/MT full-validation package provides the MT predictions
and model-independent input features. Its exact manifest/completion hashes are
pinned. Reuse requires identical full sample/split/data identities, features,
condition definitions, training seed cohort, parent checkpoints/configs/metrics,
software/GPU environment and all recorded source implementation identities.
If any guard differs, preflight fails; no replacement checkpoint, automatic
fresh-parent evaluation or relaxed tolerance is used. Transfer the original
complete package, including ignored chunks, if any required source file is missing.

During USER execution, six feature caches and eighteen MT prediction caches are
imported with their original value hashes preserved. New cache identities refer
to the new manifest; `REUSE_PROVENANCE.json` records the source paths, byte hashes,
original identities and unchanged value hashes. Original records are never
rewritten. Local conflicting files are rejected. Only the eighteen SA-MT
prediction chunks require new neural inference. The same transformed inputs
are reconstructed and checked against the shared feature-cache identities.
Both models' full clean anchors must match their original validation evidence.

The benefit is `loss(MT) - loss(SA-MT)`, positive favoring SA-MT. Per-seed values,
mean and sample SD remain separate from perturbation realizations; extra
realizations or reused MT predictions do not create training seeds or independent
experimental replications. Statistical testing and causal explanations are not
provided by these descriptive evaluation reports.

Outputs are under `results/experiments/adaptation_sa_mt_full_validation_v1/`:

```text
MANIFEST.json
REUSE_PROVENANCE.json
VALIDATION_ANCHORS.json
PROGRESS.json
COMPLETE.json
diagnostic_rows/       # 6 feature caches, 36 prediction caches, paired CSV
analyses/
    clean/
    time_scaling/
    spike_deletion/
```

After completion the package is self-contained: output verification does not
need the old AMT package, checkpoints, GPU or datasets. Preserve its relative
layout when sharing, including ignored `diagnostic_rows/`; Git alone is
insufficient. Resuming an interrupted evaluation still requires the original
reuse source and matching effective manifest. Any code/settings/environment
change requires a new `--output-root`; do not bypass identity guards.

Prepared tests, not executed by the agent:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_sa_mt tests.test_adaptation_full_validation
```

After the user completes evaluation, inspect completion evidence and per-seed
effects, then specify SA-MT statistical hypotheses and exact source identities
separately. Do not relabel AMT or LT-DTH statistical records. MT-TC and the
other datasets remain future stages.
