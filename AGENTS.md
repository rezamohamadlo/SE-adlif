# Project instructions for agents

These instructions apply throughout this repository. Keep durable policies
here. Keep main comparison reports in `reports/`; put guides and audit
documentation in `docs/`. Keep run progress, timing estimates and per-run
provenance in the relevant results directory. Shared transfer manifests belong
in `results/provenance/`.

## Reports and documentation layout

- Reserve the `reports/` root for main result comparisons:
  `paper-results.md`, `paper-vs-reference.md`, `reference-vs-models.md`, and
  `comparison-summary.md` (accuracy and inference-time tables with dataset rows
  and paper/reference/DTH/MT/MT-DTH columns).
- Put experiment guides in `docs/experiments/`, audits and reproduction
  investigations in `docs/audits/`, and transfer documentation in
  `docs/transfers/`. Do not create guides, audit logs or transfer notes in
  `reports/`.
- Keep per-run configs, metrics, summaries and measurement evidence beside
  their seed directory. Store shared cross-run transfer manifests in
  `results/provenance/`, not in `reports/`.
- Preserve existing historical archives. When relocating documentation,
  update its consumers and relative links without changing evidence values.

## Project map and configuration authority

- `run.py`: one classification training job, using Hydra and Lightning.
- `runner.py`: shared controlled comparisons, preflight, resume and aggregation.
- `ssc_validation_study.py`: explicit `runner.py --ssc-study-stage` workflow for
  PC3/PC4 SSC validation tuning and frozen confirmation. Its versioned study
  config is separate from `extension_v1`; never mix its tuning results into
  the default baseline grid or launch training as an agent.
- `measure_run_costs.py`: runner's post-test activity/compute evaluation; saves
  per-seed `COST_RESULTS.json`/`.md` without training. New completions are measured
  by default; existing completions require `--measure-existing-costs`. Cost
  failures stay separate from training completion; never retrain to repair them.
- `config/paper_baseline.yaml`: versioned paper-derived baseline for future
  fair classification comparisons and their results.
- `evaluation.yaml` and `evaluation_protocol.py`: existing controlled_v2
  comparison settings; preserve this older protocol as a separate record.
  Experiment architecture lives in `config/experiment/`;
  dataset/preprocessing defaults live in `config/dataset/`.
- `tune_shd.py`: weights-only continuation screening and fresh confirmation.
- `search_shd.py`: separate fresh LR/batch search.
- `queue_shd_dropout.py`: optional sequential dropout follow-up queue.
- `test_checkpoint.py`: standalone classification test evaluation using a
  checkpoint's saved model/dataset config; it performs no training.
- `models/ours/`: proposed neuron variants. Keep reference implementations
  separately selectable. `datasets/`, `functional/` and `module/` contain
  shared data, loss/metric and neuron-parameter utilities.
- `run_compress.py`, `generate_waves.py` and `evaluate_metrics.py`: audio
  compression workflows; do not confuse these with classification evaluation.
- `tests/`: unittest checks. `reports/`: current comparison tables:
  `paper-results.md`, `paper-vs-reference.md`, `reference-vs-models.md`, and
  `comparison-summary.md`.
  `reference-vs-models-legacy.md` separately summarizes archived results and
  their configuration/provenance limits; do not mix it into the current grid.
  `reports/archive/`: previous reports, guides, audits, document exports and
  audio measurement CSVs, including the `2026-10-15-report-consolidation/` snapshot.
  New guides and audits belong in `docs/experiments/`, `docs/audits/` and
  `docs/transfers/`; do not add documentation files to the `reports/` root.

Paper reproduction uses `config/experiment/SHD_SE_adLIF_paper.yaml` separately
from the controlled protocol. Its documented SHD settings are batch 256 and
300 epochs without the locally added early-stopping rule. Read the actual
configs for current controlled settings; do not silently change them to match
paper reproduction. Passing `evaluation_file=evaluation.yaml` applies the
controlled settings over job overrides, so do not use it for paper reproduction.

Reference records are under `results/valid/reference/`; matched proposed-model
records are under `results/valid/`. Tuning and reproduction use
separate directories under `results/experiments/`. Historical documentation is
not the authority for current paths, commands or settings.

## SHD evaluation policy

The user has chosen **SHD*** for all future SHD experiments and controlled
comparisons. The underlying dataset and existing config identifiers remain
`SHD`; label the evaluation protocol **SHD*** in reports.

- Reserve 20% of the official training set for validation
  (`dataset.validate_on: 0.2`). Keep the split seed fixed at 42
  (`dataset.random_seed: 42`) across models and training seeds.
- Select model checkpoints, hyperparameters and architectures using validation
  results. Keep the official test set separate from training and selection.
- Evaluate the official test set after validation-based selection. Report test
  results and variation across matching training seeds.
- Compare paper reproduction with the two-layer SHD* result
  **93.79 +/- 0.76% across 20 runs**, not the test-selected SHD result 95.81%.
  A paper mean is not a guaranteed target for an individual training seed.
- Retain historical test-selected SHD results only as clearly labeled
  historical records; exclude them from future SHD* controlled comparisons.
- Previous sweeps that selected candidates using test scores are exploratory,
  not independent final-test evidence. Do not silently relabel them.

## Mandatory paper-baseline inheritance

Use **`config/paper_baseline.yaml` as the single paper-derived baseline file**
for all new fair reference/proposed-model comparisons. Select its dataset-specific
section; do not apply SHD settings to SSC or ECG. Paper-documented values and
local reproduction choices are explicitly separated in this file.

- New experiment configs must inherit this file and reference its settings
  rather than duplicating LR, batch, epoch budget, preprocessing and shared
  optimization settings. `SHD_SE_adLIF_paper.yaml` demonstrates inheritance.
- Keep intended architecture/neuron differences explicit. Verify the effective
  LR, batch, split, budget and stopping rules are matched across compared models.
- Save the fully resolved config, `paper_baseline_version`, dataset section,
  training/split seeds and all deviations with each run's results. Hydra saved
  configs/checkpoint hyperparameters provide the resolved-config record; include
  baseline identity in manifests or result metadata when producing aggregates.
- Tuning may override the baseline only in separate experiment outputs. Record
  every override and label those runs as tuning or a different protocol. Do not
  describe mismatched configurations as a fair baseline comparison.
- Changes to baseline values require a new version and new result directories.
  Do not silently update completed runs, frozen controlled_v2 settings or active
  processes. Using the baseline does not prove exact paper reproduction or
  guarantee the paper's reported accuracy.
- `runner.py` applies `config/paper_baseline.yaml` by default through the
  dataset-specific `config/comparison/` group. Outputs use
  `results/valid/`; no baseline flag is required. This directory
  name does not identify the training protocol: use saved baseline metadata.
  Archive incompatible historical runs before reusing their paths.
  Explicit tuning overrides use separate experiment outputs. `evaluation.yaml`
  remains the historical controlled_v2 record and is not the runner default.

## Preserve fair comparisons

The user requires the main runners to remain unchanged for later fair
experiments. Use separate experiment configs or launchers for tuning and paper
reproduction. Do not change `run.py`, `runner.py`, shared model implementations
or `evaluation.yaml` to implement a new search unless explicitly requested.
Keep new results separate from existing runs and preserve their provenance.

For new fair model comparisons, including training-objective variants, use
`runner.py` for preflight, training, resume, final testing, aggregation and
cost measurement. Register new objectives as explicit opt-in configurations;
keep ordinary model behavior unchanged and record objective deviations and
distinct protocol metadata. A shared runner matches the controlled settings,
but does not make intentionally different training objectives identical.
Do not substitute a separate candidate launcher unless explicitly requested.

For the SSC mean-loss reference/DTH/MT comparison, use the official
`runner.py` workflow for new training, resume, final testing, aggregation and
cost measurement. Do not substitute the separate SSC diagnostic/comparison
launchers. Dataset-specific comparison configs select loss automatically: SHD uses
`sum_softmax_over_time`, SSC uses `mean_membrane_potentials` under
`ssc_mean_loss_v1`, and ECG uses `summed_membrane_potentials` on per-timestep
blocks. No loss flag is needed; `--ssc-mean-loss` is a compatibility alias.
Preserve archived summed-loss SSC records and other datasets' protocol metadata. The completed
reference seed 42 is already retained under `results/valid/reference/SSC/`;
do not retrain or overwrite it when scheduling DTH and MT.

All training must be launched by the user. Never start, resume or restart
training processes yourself, including fresh training, tuning, fine-tuning,
weights-only continuation, and training queues. When training is needed,
prepare the configs and launcher, perform non-training checks such as dry runs,
then return the exact command line for the user to execute. Requests to prepare
or run an experiment do not override this rule. Read-only inspection and
explicitly requested standalone checkpoint evaluation without training remain
allowed.

## Execution and checkpoint safety

- Use `project_paths.py` for classification artifact path serialization and
  resolution. Persist repository-relative paths for files inside the checkout;
  resolve them against the repository root at runtime, not Hydra's run directory.
  `config/path_aliases.yaml` registers previous checkout roots. Relocate only
  an explicit root prefix, preserving the artifact suffix and checkpoint identity.
  Compare normalized paths in manifest guards while retaining historical records.
  Never substitute a different checkpoint because a recorded file is missing.
- For future development and new experiment workflows, keep paths to project
  files repository-relative in configs, manifests, reports and generated commands.
  Do not introduce user names, drive letters or machine-specific checkout roots.
  Resolve paths against the repository root only when accessing files; explicit
  external storage paths are allowed when needed and must be documented.
- Before proposing next-step commands or changing file-loading/saving code,
  check that paths work from a different checkout location and when Hydra changes
  the working directory. Check that the referenced artifacts exist locally;
  Git pull does not transfer ignored checkpoints or datasets. Preserve the same
  relative results layout when transferring files between computers. Apply this
  portability convention when reusing this project's workflow in another project,
  without changing that project's instructions unless requested.
- Use the project's `.venv` Python explicitly in commands. Do not accidentally
  install dependencies into global Python. The tested Windows environment is
  described by `requirements.txt`; `environment.yml` is the legacy Conda setup.
- Before starting expensive work, inspect existing processes, output paths and
  launcher locks. Never run two launchers writing the same results directory.
  The OS-managed `.search.lock` may exist without being held; do not delete it
  to bypass an active lock. Check actual processes when a status file is stale.
- Ask for missing preferences when necessary, but do not add confirmation steps
  for work already authorized. Do not launch, stop or restart the user's active
  training merely to answer a status question.
- Preserve terminal visibility when requested. If an authorized process runs
  in the background, explain where its output is recorded and how to watch it.
- Fresh training starts with random weights and a fresh optimizer at epoch 0.
  Weights-only continuation loads model weights but resets optimizer, scheduler
  and epoch state. Full resume restores a run's own training state. State which
  mode an experiment uses; they answer different research questions.
- Resume from `last.ckpt` for continuation of an interrupted run. Use the best
  validation-selected checkpoint for final test evaluation; these may differ.
- Retain manifest guards. Do not reuse output directories after changing
  architecture, preprocessing, split, seed, budget or optimization settings.
  Use a new directory or an explicitly designed staged-budget workflow.
- Only load trusted local checkpoints with unrestricted deserialization.
  Standalone classification checkpoints must match their saved dataset and
  registered model; cross-dataset evaluation is not automatically compatible.

## Evidence and result reporting

- Use ordinary inference without test-time augmentation (TTA) in all current
  comparison reports. Do not select between ordinary and TTA results using test
  accuracy. Preserve historical exploratory TTA evidence separately.

- In all current reports and newly generated comparison tables, wherever local
  results are compared with a numeric paper baseline, append `(Δ paper)` to
  each applicable local metric column heading and a signed difference in
  parentheses after each local value, e.g. `94.49 ± 0.52 (+0.70 pp)`.
  Define Δ paper once per report as local minus paper, with explicit units.
  Accuracy differences use percentage points and unrounded local metrics
  against the published paper mean; compare means with means and clearly
  label any per-seed comparisons to the paper mean. Do not subtract SDs or
  annotate seed IDs, run counts, or the paper's own baseline with a delta.
  For other comparable metrics use their native units; differences against
  rounded paper counts must be marked approximate and identify the rounded
  baseline. Missing, unreported or incompatible paper values have no numeric
  delta. Preserve historical archives and provenance records; apply this
  presentation rule to current reports and future generated reports without
  rewriting historical evidence. Retain protocol and significance caveats.

- Verify results against `metrics.csv`, `TEST_RESULTS.json`, verified
  `RESULT.json` records where present, saved configs and checkpoint metadata.
  `TRAINING_SUMMARY.md` can lag during training or contain historical text.
  Resolve disagreements explicitly rather than copying whichever value looks
  better. Reject mixed histories from overlapping processes as valid run data.
- Clearly distinguish validation accuracy, final test accuracy, continuation
  results and fresh-training results. Label the dataset protocol SHD* in reports.
- Compare configurations across matching training seeds. Keep split seed 42
  fixed. Report per-seed values, mean, sample SD and number of completed runs.
  Do not count repeated runs of one seed as independent training seeds.
- Calculate differences in percentage points from unrounded metrics whenever
  those exist. Label rounded historical baselines and the metric being compared.
- Partial/incomplete runs are progress evidence, not completed results. Missing
  test results remain pending; do not infer them from validation accuracy.
- Test-selected rankings are exploratory. Do not present their highest score as
  an independently validated improvement. A validation improvement does not
  guarantee a test improvement, and a fine-tuning winner may fail from scratch.
- Do not promise paper-level accuracy. Exact split indices, environment, random
  initialization and uninterrupted training behavior may differ from the paper.
  Resume restores optimizer state but does not establish equivalence of the
  random data/dropout sequence to an uninterrupted run.

## Verification

Classification cross-entropy targets must use `torch.int64` (`torch.long`).
Converting integer class labels from int32 to int64 preserves their values and
does not change the split, architecture, loss definition or training protocol;
this compatibility fix does not invalidate a comparison. Keep the conversion
in the classification branch so regression targets retain their original dtype.
Verify loss equivalence and finite gradients on CPU and CUDA when available.
A failed startup with no saved training checkpoint must restart from scratch;
do not describe it as checkpoint continuation.

Preflight before training where supported:

```powershell
.\.venv\Scripts\python.exe runner.py --models SE_adLIF --datasets SHD --seeds 42 --dry-run
.\.venv\Scripts\python.exe search_shd.py --dry-run
.\.venv\Scripts\python.exe queue_shd_dropout.py --dry-run
.\.venv\Scripts\python.exe run.py experiment=SHD_SE_adLIF_paper --cfg job --resolve
```

Run tests relevant to code/config changes. For example:

The user-launch-only training rule also covers synthetic training smoke
tests. Exclude tests that call `Trainer.fit`, training entrypoints, or optimizer
training steps from agent-run discovery; leave those for the user to launch.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_search_shd tests.test_queue_shd_dropout tests.test_paper_config
git diff --check
```

For wider implementation changes, use unittest discovery. Do not start full
training as a routine test. Documentation-only changes usually need command,
path/link and formatting checks rather than model training. Report what was
actually verified and any material limitations.

## Artifacts, cleanup and Git

- Keep evaluated SHD* candidates without accuracy improvement over DTH/MT in
  `results/not_improved/`, preserving model/seed artifacts and documenting the
  tested configuration and outcome in its README. These remain valid research
  evidence. Assess SSC/ECG separately. Check this directory before proposing
  duplicate experiments; intentional reruns require a separate output directory.
- Store per-run measurements inside the actual training seed directory, beside
  `TRAINING_SUMMARY.md`, as `COST_RESULTS.json` and `COST_RESULTS.md`. Keep this
  layout for new models, datasets and launchers; do not recreate a shared
  `results/measurements/` tree by default. The runner measures new completions
  automatically unless `--no-measure-costs` is passed; use
  `--measure-existing-costs` explicitly for already completed jobs. Preserve
  checkpoint identity and existing measurement evidence when reusing records.
  Keep superseded measurements in the seed's `measurement_history/` and shared
  historical evaluator evidence in `archive/measurement_provenance/`.
  See `results/valid/MEASUREMENTS.md` for the layout and compatibility rules.
- Preserve summaries, manifests, test-result JSONs, aggregate reports and
  underlying metrics. Keep best checkpoints needed for evaluation and last
  checkpoints needed for resume. Large weights and raw metrics may remain local
  or in external storage; retain enough provenance to audit reported results.
- Keep `.venv`, datasets, preprocessing caches, verbose logs, weights, locks and
  generated launcher configs out of Git according to `.gitignore`. Generated
  `trial.yaml` is rebuilt from the manifest; ignoring it does not untrack an
  already committed copy.
- Cleanup must distinguish disposable artifacts from research evidence. Check
  references and consumers before removing code or configs. Verify dataset
  duplicates by hashes and retained extracted files before deleting a copy.
  Do not delete caches simply to tidy Git when they are already ignored.
- Before destructive file operations, verify absolute paths remain within the
  intended workspace/target directory. Respect approval-review rejections and
  report blocked actions; do not claim blocked deletions were completed.
- Preserve unrelated user edits. Inspect status before staging, rebasing or
  cleaning. Fetch and integrate remote changes safely when requested; resolve
  report conflicts using provenance while preserving useful local records.
- Commit or push only when requested. Stage intentional source/docs/results;
  exclude generated artifacts. Check for conflicts and verify the final remote
  state after pushing. Never force-push or discard user work without explicit
  authorization.
- Never push to a remote without first obtaining explicit user confirmation
  for that push. Permission to commit does not authorize pushing.
- Status updates should distinguish completed work from queued, running,
  interrupted or unverified work. Keep explanations concise and candid.

## Canonical SHD reference location

The user explicitly replaced the prior SHD reference. The sole current
reference is `results/valid/reference/SHD/seed_<seed>/`, with aggregate
`REFERENCE_RESULTS.json` and per-seed `REFERENCE_RESULT.json`. Its protocol
is paper_classification_v1 (batch 256, full 300 epochs). Do not restore the
retired controlled-v2 SHD tree or use old date-based paths as the active baseline.
Historical experiments retain their original baseline in REFERENCE_BASELINE.json;
do not retroactively substitute current reference scores into those records.
Proposed controlled-v2 SHD results remain unmatched until rerun with this
baseline. Prefer metadata/metrics over hard-coded reference numbers.


## Codex Model and Reasoning Efficiency Rules

I use ChatGPT Plus and want to maximize coding quality while minimizing unnecessary usage.

### 1. Default Settings

For most development tasks, recommend:

- **Model:** GPT-6.1 Sol
- **Reasoning:** Medium

This is the preferred balance between implementation quality, speed, and usage efficiency.

### 2. Task-Based Recommendations

Recommend model and reasoning settings according to task complexity:

| Task Type | Model | Reasoning |
|---|---|---|
| Normal coding and feature implementation | GPT-6.1 Sol | Medium |
| Complex debugging and refactoring | GPT-6.1 Sol | High |
| Research algorithms and mathematical correctness | GPT-6.1 Sol | High |
| Experimental protocol and scientific validity reviews | GPT-6.1 Sol | High |
| Extremely difficult problems unresolved by Sol | GPT-6 Astra | Medium |
| Simple scripts, documentation, formatting, and minor fixes | GPT-6 Luna | Low |

Only recommend models that are actually available in the user's Codex environment. Otherwise, suggest the closest available alternative.

### 3. Automatic Reminders

- At the beginning of a substantial new task, evaluate whether the current model and reasoning level are appropriate.
- If a different setting would significantly improve efficiency or reliability, recommend switching before beginning implementation.
- If the current settings are appropriate, continue without unnecessary reminders.
- Reevaluate when task complexity changes significantly.
- Never interrupt an ongoing task merely to suggest a model change.
- Do not repeat recommendations in every response.

### 4. Plus Usage Optimization

- Prefer Medium reasoning unless the task genuinely requires High.
- Reserve expensive models and higher reasoning levels for difficult problems.
- Use targeted code inspection and tests instead of repeatedly analyzing the entire repository.
- Avoid unnecessary tool calls, repeated searches, and redundant analysis.
- Never sacrifice correctness, testing, or scientific validity solely to reduce usage.
- Do not assume access to account usage information or automatically change model settings.

### 5. Reminder Format

When a model change is recommended, briefly display:

**Recommended Codex settings:** `[Model]` — `[Reasoning Level]`

**Reason:** One concise sentence explaining why these settings fit the current task.

Then proceed according to the user's instructions.

### 6. Research Project Priority

For the SE-adLIF research repository, prioritize correctness, reproducibility, experiment isolation, validation integrity, and checkpoint safety over speed or token savings.

Use High reasoning when reviewing changes that could affect experimental results, training dynamics, evaluation protocols, or scientific conclusions.
