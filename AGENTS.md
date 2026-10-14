# Project instructions for agents

These instructions apply throughout this repository. Keep durable policies
here; put changing scores, run progress, timing estimates and findings in
`reports/` or the relevant results directory.

## Project map and configuration authority

- `run.py`: one classification training job, using Hydra and Lightning.
- `runner.py`: shared controlled comparisons, preflight, resume and aggregation.
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
- `tests/`: unittest checks. `reports/`: maintained guides and comparisons;
  `reports/archive/`: historical notes; `reports/exports/`: document exports;
  `reports/audio-compression/`: audio measurement CSVs.

Paper reproduction uses `config/experiment/SHD_SE_adLIF_paper.yaml` separately
from the controlled protocol. Its documented SHD settings are batch 256 and
300 epochs without the locally added early-stopping rule. Read the actual
configs for current controlled settings; do not silently change them to match
paper reproduction. Passing `evaluation_file=evaluation.yaml` applies the
controlled settings over job overrides, so do not use it for paper reproduction.

Reference records are under `results/valid/reference/`; matched proposed-model
records are under `results/valid/protocol_v2/`. Tuning and reproduction use
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
- Existing runner entry points do not automatically enforce this new file.
  Verify config inheritance/overrides before launching future comparisons;
  do not assume invoking `runner.py` alone applies the paper baseline.

## Preserve fair comparisons

The user requires the main runners to remain unchanged for later fair
experiments. Use separate experiment configs or launchers for tuning and paper
reproduction. Do not change `run.py`, `runner.py`, shared model implementations
or `evaluation.yaml` to implement a new search unless explicitly requested.
Keep new results separate from existing runs and preserve their provenance.

Do not launch training merely because you are providing a command. When the
user asks to run it themselves, return the command and let them start it.

## Execution and checkpoint safety

- Use the project's `.ven` Python explicitly in commands. Do not accidentally
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

Preflight before training where supported:

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF --datasets SHD --seeds 42 --dry-run
.\.ven\Scripts\python.exe search_shd.py --dry-run
.\.ven\Scripts\python.exe queue_shd_dropout.py --dry-run
.\.ven\Scripts\python.exe run.py experiment=SHD_SE_adLIF_paper --cfg job --resolve
```

Run tests relevant to code/config changes. For example:

```powershell
.\.ven\Scripts\python.exe -m unittest tests.test_search_shd tests.test_queue_shd_dropout tests.test_paper_config
git diff --check
```

For wider implementation changes, use unittest discovery. Do not start full
training as a routine test. Documentation-only changes usually need command,
path/link and formatting checks rather than model training. Report what was
actually verified and any material limitations.

## Artifacts, cleanup and Git

- Preserve summaries, manifests, test-result JSONs, aggregate reports and
  underlying metrics. Keep best checkpoints needed for evaluation and last
  checkpoints needed for resume. Large weights and raw metrics may remain local
  or in external storage; retain enough provenance to audit reported results.
- Keep `.ven`, datasets, preprocessing caches, verbose logs, weights, locks and
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
