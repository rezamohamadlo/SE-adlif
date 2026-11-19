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
- Keep the stage-status table in `supervisor-guideline-roadmap-fa.md` current
  whenever work changes the supervisor study's implementation, evidence,
  completion status or next step. Update its review date and evidence links in
  the same task; distinguish implemented, executed, verified and pending work,
  and pilot subsets from full-validation results. Reconcile the three-PC plan
  when relevant. Never mark a stage complete from plans or stale summaries alone.

## Project map and configuration authority

- `run.py`: one classification training job, using Hydra and Lightning.
- `runner.py`: shared controlled comparisons, preflight, resume and aggregation.
- `ssc_validation_study.py`: explicit `runner.py --ssc-study-stage` workflow for
  PC3/PC4 SSC validation tuning and frozen confirmation. Its versioned study
  config is separate from `extension_v1`; never mix its tuning results into
  the default baseline grid or launch training as an agent.
- `scripts/adaptation_ssc_suite/run.py`: user-launched one-PC SSC mechanism
  queue for saved-checkpoint validation evaluations and exploratory statistics.
  Launch it from the repository root as
  `.\.venv\Scripts\python.exe -m scripts.adaptation_ssc_suite.run`, so its local
  `statistics.py` does not shadow Python's standard-library module. Preserve
  the recorded `--output-root` when resuming completed prediction chunks/jobs.
  For the existing v2 run's Windows atomic-JSON-write failures, the user resume
  entrypoint is `.\.venv\Scripts\python.exe -m scripts.resume_ssc_suite` with
  the same output root. Its scoped I/O retry layer records separate compatibility
  provenance and leaves frozen source/manifests unchanged; its `--check-only`
  uses disposable filesystem probes plus ordinary preflight, without tests or
  scientific execution.
  `--check-only` performs preflight without tests, inference or fitting. The
  execution command runs selected contract tests, preserves the exact completed
  AMT/MT evidence with a path index, then evaluates/analyzes the other pairs.
  Suite results remain in `results/experiments/`; it performs no training or
  official-test evaluation. ECG and stage 7 gradients are separate workflows.
- `scripts/adaptation_gradient_pilot/run.py`: isolated user-launched stage-7
  SHD* MT/AMT seed-42 input-surrogate-gradient pilot. The 40-record class-balanced
  cohort is chosen without outcomes. Original-batch full-validation anchors
  remain separate from declared single-record gradient/control batching; record
  source-batch drift, and require same-mode autograd forward equivalence.
  Differentiate only the per-record saved supervised CE with respect to inputs
  through `MLPSNN.forward`, under enabled autograd and frozen eval-mode weights.
  Never use no_grad/inference-mode histories for backward, update weights or
  interpret SLAYER sensitivities as true hard-spike finite differences/causal
  attention. Include burn-in inputs in physical sensitivity; report padding
  gradients separately. MT-guided exact-budget deletions and three matched
  random controls must use the same inputs for both models. Zero event-gradient
  guidance is explicitly ineligible; it cannot be replaced by a fabricated map.
  Tests include toy backward passes and are USER-RUN ONLY. `--check-only` performs
  artifact/config/input checks, never forward/backward or deletion generation.
  Binary gradient caches bind metadata/checksums and use `allow_pickle=False`;
  missing published sidecars may be recovered from authenticated embedded data,
  but a missing array for an existing sidecar must fail rather than be replaced.
  No optimized diagnostic inputs or full stage-7 completion are claimed here.
- `scripts/adaptation_batch_precision/run.py`: isolated user-launched SHD*
  forward-only batching/precision diagnostic. Preserve the executed gradient
  pilot, including recording 3896; use a separate output root. Compare original
  batching, natural singletons and original-padded singletons at fixed inputs,
  labels and block masks. Record actual backend flags; stricter precision modes
  are diagnostic deviations, never replacement anchors or an automatic winner.
  Source-high original anchors retain their original tolerances. Only
  `--check-only` and `--verify-only` are agent-safe; tests and every model forward
  are user-launched. No gradients, training, official test or full stage-7
  completion are authorized through this diagnostic.
- `scripts/adaptation_gradient_strict/run.py`: separate user-launched SHD*
  fixed-precision input-gradient repeat diagnostic. Use `highest_matmul` and
  the authenticated same-mode validation references from the numerical study;
  never substitute them for source-high training/validation evidence. Two
  full-validation references must pass before backward. Keep repeated raw
  gradients and explicit forward/repeat/padding criteria. Report zero maps and
  failed stability checks; completed execution does not imply stability passed.
  Cross-precision legacy comparisons are descriptive, not repeat gates. No
  automatic deletion experiments, input optimization or full stage-7 completion.
  Agent use is limited to `--check-only`/`--verify-only`; all tests (including
  toy backward) and experiment execution remain user-launched.
- `scripts/adaptation_gradient_deletion/run.py`: separate matched event-deletion
  pilot reusing the fixed natural repeat-0 strict gradients. All 80 fresh clean
  reference checks precede any perturbations. MT-guided and three exact random
  event-deletion controls share budgets and inputs across both models; padding,
  labels and block masks stay fixed. No new backward, training or input fitting.
- `scripts/adaptation_stage7/run.py`: registered three-dataset stage-7 diagnostic
  suite for all four model pairs and three saved training seeds. Preserve all
  source-high baselines. The common supervised objective excludes auxiliary JS;
  only input leaves receive gradients and all model weights/state remain frozen.
  Input optimization relocates at most 1% of native events within channels,
  preserves length/targets/masks and per-channel totals (both ECG polarities),
  and retains failed/rejected proposals. Compare identical optimized inputs and
  budget-matched random relocations on both parent and variant. Scheduling flags
  select jobs, never scientific cohorts or artifact identities. Only the verified
  54-job, nine-transfer-bundle closeout can mark its registered stage-7 scope
  complete; limited gradient cohorts are not exhaustive full-validation gradients.
  Default execution policies above/below still apply; explicit task authorization
  belongs in per-run provenance, not a permanent execution-policy exception.
- `scripts/adaptation_stage8/run.py`: user-launched final supervisor-study
  synthesis from authenticated saved evidence only. It verifies all 54 study
  checkpoint/cost/history identities, source completion ledgers and saved
  stage-6/7 summaries. It performs no inference, feature extraction, fitting,
  bootstrap, training or official-test access. Preserve source-high ordinary
  results separately from highest-precision gradient diagnostics, both input
  optimization origins, dataset-specific units/losses and original interval
  families. Hidden activity includes saved minimum padding and burn-in;
  output integrators do not spike. Historical ECG input-projection accounting
  is retained rather than silently recounted. Timing is descriptive across
  recorded sessions, not a controlled speed/energy claim. `--check-only` checks
  artifacts/configs without tests, aggregation or figures; `--verify-only`
  authenticates a completed report. Agents may run these two modes only. The
  ordinary command runs user contract tests and generates the final report,
  figures and CSVs in a separate PC-independent results root. Keep stage 8
  pending until actual execution and review; no new training is needed.
- `scripts/adaptation_ssc_windows/run.py`: isolated user-launched clean SSC
  MT/MT-TC output-window diagnostic. It uses all 9,981 official validation records
  and six saved checkpoints, with 234 resumable original-batch chunks. Ordinary
  predictions must match the completed clean caches exactly, and ordinary
  accuracy/CE must match saved validation tolerances. Preserve the ordinary loss
  mask, including its saved minimum padding; descriptor windows separately
  exclude burn-in and all synthetic padding. Nonoverlapping 5/10/20-frame
  windows are descriptive; frame-softmax probability metrics and mean-membrane
  window CE/accuracy are distinct. Never count windows as independent seeds or
  automatically regress output confidence/stability against loss benefit.
  `--check-only` runs no forward, descriptor metrics or tests; agents may run
  preflight/verification only. Keep frozen SSC and SHD sources unchanged.
- `scripts/adaptation_ecg_extension/run.py`: isolated user-launched ECG feature,
  aligned-time-resampling and descriptive-analysis extension. Preserve/snapshot
  the completed v2 evidence without rewriting manifests; reuse its 72 prediction
  chunks and add only 36 mapped chunks. Signed inputs, labels and masks share
  one nearest-bin-center time map; rebuild target-aligned blocks and retain
  the saved zero-vector-to-class-0 policy. Record resampling event-count changes.
  Feature time units are model steps, not established milliseconds. No patient
  IDs are inferred: associations, per-training-seed sample SD and record influence
  are descriptive, without bootstrap, p-values or population confidence intervals.
  Constant controls, undefined/censored features and non-estimable designs must
  be reported explicitly. `--check-only` performs no feature extraction, tests,
  model forward or regression. Execution remains user-launched; do not declare
  stage 6 complete automatically from queue completion.
- `scripts/adaptation_ecg_suite/run.py`: completed user-launched four-pair ECG
  clean/deletion evaluation queue. Use module mode and the v2 output root.
  It matches training's `torch.set_float32_matmul_precision('high')`, records
  backend settings and preserves the failed v1 evidence. All 18 unique clean
  anchors must pass unchanged tolerances before deletion. Parent predictions
  are shared across pairs; outputs remain PC-independent. `--check-only` runs
  no tests or scientific execution. Features, time scaling and inferential
  statistics remain separate pending work. Never reuse failed v1 predictions
  or relax anchor tolerances to bypass a numerical mismatch.
- `scripts/adaptation_ecg/run.py`: historical v1 ECG validation alignment
  and signed-event-deletion pilot. Use module mode from the repository root.
  `--check-only` checks data, 18 checkpoint identities/configs and input-label
  alignment without tests or model forward. Normal execution runs user contract
  tests, six AMT/MT clean anchors, then deletion only after all anchors pass.
  The first user run failed its MT seed-42 clean anchor; its source and results
  are frozen. Use the suite above for the corrected numerical contract.
  Open only the training MAT; preserve the saved argmax treatment of all-zero
  class vectors, temporal blocks and masks. No training or official-test access.
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
preflight-only checks remain allowed. Standalone checkpoint evaluation and
statistical analysis are also user-launched under the execution rule below.

## Execution and checkpoint safety

- All experiment execution is user-launched, including checkpoint evaluation,
  neural inference, statistical analysis, bootstrap/resampling and experiment
  queues, even when short or local. Agents may prepare/edit implementations,
  inspect existing files and run preflight-only checks that do not execute the
  experiment or produce scientific results. Return the exact command for the
  user to launch. For experiment implementation tasks, give test commands to
  the user rather than running tests; agent-run checks are limited to preflight.
  Do not use tests or smoke checks to execute an experiment. The local-default
  rule below concerns scheduling, not permission to execute experiments.
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
- Ask how many PCs will be used only when preparing a long-running experiment
  or training workload where multiple PCs could materially help, unless the
  user has already specified the count for that workload. Assess expected
  runtime and workload first; do not ask merely because work is an experiment.
  Short evaluations, analyses and other lightweight processes default to local
  execution without a PC-count question. For long workloads, tailor assignments
  and commands to the agreed count (for example, three labeled commands for
  three PCs). All training remains user-launched.
- Keep experiment results independent of PC assignments. Organize outputs by
  study, dataset, analysis/condition, model and seed; record hardware only as
  provenance. PC counts affect scheduling and commands, not artifact identity.
  Keep shared result packages self-contained with portable paths and verifiable
  hashes. Preserve historical identities when relocating evidence and provide
  an explicit path index rather than rewriting completed manifests.
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
