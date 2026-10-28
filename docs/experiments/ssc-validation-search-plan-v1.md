# SSC validation-based improvement plan v1

Status: implemented as an explicit opt-in `runner.py` workflow for PC3 and PC4;
no training launched by the agent. This is follow-up development after observing historical SSC test
results, including exploratory TTA. Those observations must remain disclosed.

## Objective

Find a configuration with reliable SSC accuracy under ordinary inference.
SA-MT is a priority candidate, but select the model using official validation
results. Include MT and AMT rather than requiring SA-MT to win.

A useful practical target is at least 81.00% mean test accuracy, with a positive
paired difference against a freshly trained reference. This is an aspiration,
not a promised outcome or a threshold for suppressing results. Report all final
seeds even if the target is missed. The paper mean is 80.44%; the historical
local reference mean is about 80.47%. Neither is a validation selection target.

## Starting evidence

Read-only inspection of saved `metrics.csv` records on 28 October 2026:

| Model | Best-validation mean (%) | Per-seed best-validation accuracy, 42 / 123 / 456 (%) | Selected epochs, zero based |
|---|---:|---|---|
| Reference | 82.01 | 82.22 / 81.82 / 82.00 | 18 / 13 / 27 |
| MT | 82.15 | 82.00 / 82.00 / 82.47 | 23 / 36 / 17 |
| SA-MT | 81.88 | 81.82 / 81.95 / 81.89 | 13 / 26 / 27 |
| AMT | 82.05 | 82.25 / 81.95 / 81.95 | 35 / 27 / 13 |

Sources, each with `SSC/seed_<seed>/logs/mlp_snn/version_0/metrics.csv`:

- [Reference](../../results/valid/reference/SSC/)
- [MT](../../results/valid/MT_SE_adLIF/SSC/)
- [SA-MT](../../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/)
- [AMT](../../results/not_improved/AMT_SE_adLIF/SSC/)

These are means of each run's validation-selected checkpoint accuracy, not
test results. MT currently leads validation; AMT's observed TTA test gain does
not establish that more seeds or input augmentation will improve ordinary
inference. Peaks occur before the budget ends in all inspected runs; audit
learning curves before considering more epochs. This alone does not diagnose
overfitting or prove a particular intervention will work.

## Fixed protocol

- Inherit the SSC section of [paper_baseline.yaml](../../config/paper_baseline.yaml).
- Keep official train/valid/test partitions, mean-membrane loss under
  `ssc_mean_loss_v1`, preprocessing, two 720-neuron layers, batch 256, Adam,
  40 epochs, no early stopping, and current scheduler behavior.
- Baseline LR is 0.006 and dropout is 0.15. Record every deliberate override as
  tuning, with distinct protocol metadata and output directories.
- Screening uses validation only: no test loader iteration, final test,
  test-derived ranking, TTA, or test-set cost measurement.
- Every new training job starts from random weights and a fresh optimizer.
  Resume only that job's own `last.ckpt`; select its best validation checkpoint.
- Only the user launches training. Preparation and dry runs perform no training.

## Stages and maximum budget

| Stage | Work | New training runs |
|---|---|---:|
| 0 | Audit existing configs, validation curves, checkpoints and run histories; freeze manifest and selection rules | 0 |
| 1 | MT, SA-MT and AMT; LR 0.003 / 0.006 / 0.009; seeds 42 and 123 | 18 |
| 2 | Two best model/LR candidates; dropout 0.10 / 0.20 in addition to existing 0.15; seeds 42 and 123 | 8 |
| 3 | Best configuration from each finalist model on seed 456 | Up to 2 |
| 4 | Freeze one winner; train it and the fixed SE-adLIF reference on five fresh matching seeds | 10 |

Maximum budget is 38 new runs. All screening baselines are trained afresh:
historical runs do not have sufficient source provenance for automatic reuse.
Keep run timing evidence and estimates beside the actual study results. Stop
after a stage if evidence does not justify the next expense; do not extend this
grid after looking at test scores. Never copy completion markers to manufacture
new runs.

Stage 1 ranks each model's best LR by mean best-validation accuracy over the
two matching seeds and advances the best two distinct models. Stage 2 chooses
one dropout per finalist by the same criterion. Stage 3 ranks those two frozen
finalists over seeds 42, 123 and 456. All these seeds are
development evidence; previous inspection means seed 456 is not independent
confirmation. Break exact mean ties by lower sample SD, then lexicographic
configuration ID. Report individual seeds as well as means;
two seeds provide only a screening signal.

If neither finalist has a positive two-seed screening mean validation difference
from its own fresh LR-0.006/dropout-0.15 baseline, end this grid and review learning curves before committing
the confirmation budget. A positive difference permits confirmation; it does
not guarantee test improvement or statistical significance.

Proposed confirmation seeds: 1009, 2027, 3037, 4049, 5051. Verify they have not
already been used for SSC development, and freeze them before starting. Do not
change the existing ECG/SHD* extension cohort or its active output directories.
Train all confirmation jobs without testing. After completion, freeze the
checkpoint list and evaluate all ten checkpoints once using ordinary inference.
Do not switch winners after seeing those test results. The official test set
has already been observed: fresh training seeds test seed robustness, not
generalization to a previously untouched test dataset.

## Workflow and machine assignments

Use `runner.py` for the research workflow, including preflight, resume,
aggregation, frozen final evaluation and per-seed cost measurement. Do not
substitute the SSC diagnostic launchers. Keep outputs under a new root such as
`results/experiments/ssc_validation_search_v1/`, with distinct candidate and
confirmation subdirectories; never overwrite `results/valid/` or existing runs.
Save resolved configs, baseline version/section, deviations, seeds, checkpoint
identities and implementation provenance with the results.

The explicit `--ssc-study-stage` route uses
[ssc_validation_study.py](../../ssc_validation_study.py) and the frozen
[study definition](../../config/studies/ssc_validation_search_v1.yaml).
Ordinary runner behavior, `extension_v1`, the baseline, model implementations
and `run.py` are unchanged. Generic overrides cannot be combined with this
route; only the declared LR/dropout grid is accepted. Every training stage
uses the official runner preflight, validation-only training and resume checks.

| Stage | PC3 | PC4 | Coordinator after merging records |
|---|---|---|---|
| LR | Nine candidates, seed 42 | Nine candidates, seed 123 | `select-lr` |
| Dropout | Four new candidates, seed 42 | Four new candidates, seed 123 | `select-dropout` |
| Check | First finalist, seed 456 | Second finalist, seed 456 | `select-winner` |
| Confirm | Winner + reference, seeds 1009 / 3037 / 5051: six jobs | Winner + reference, seeds 2027 / 4049: four jobs | `freeze-test` |
| Test | Its six frozen checkpoints | Its four frozen checkpoints | `report` |

Commands are in [command.txt](../../command.txt). Start only the LR stage now:

```powershell
# PC3 preflight, then user-launched training:
.\.venv\Scripts\python.exe runner.py --ssc-study-stage lr --study-pc pc3 --dry-run
.\.venv\Scripts\python.exe runner.py --ssc-study-stage lr --study-pc pc3

# PC4 preflight, then user-launched training:
.\.venv\Scripts\python.exe runner.py --ssc-study-stage lr --study-pc pc4 --dry-run
.\.venv\Scripts\python.exe runner.py --ssc-study-stage lr --study-pc pc4
```

Each PC needs a prepared `.venv` and the exact SSC HDF5 files in
`data/SSCWrapper/`. Git does not transfer datasets or ignored weights. Keep the
same source commit and clean source state throughout the study. On PCs already
running PC1/PC2 jobs, do not pull this change into their active checkouts; let
those jobs finish or use a separate checkout for the new work.
Each final model/reference seed pair stays on the same PC to keep hardware
assignment matched within the paired comparison. Record environment differences
across PCs; this allocation deliberately favors pairing over equal job counts.

Rerun the same stage command to restore each interrupted job from its own
`last.ckpt` and skip verified completions. Source revisions, config drift,
unknown output directories, missing exact checkpoints and mixed CSV histories
are rejected. A failure before any checkpoint is not continuation; preserve
the evidence and investigate before choosing a separately versioned restart.
Each PC has its own OS-managed launcher lock. Do not remove a lock to bypass
an active launcher. Training is visible in the invoking terminal, with each
seed's output retained in `out.log` and validation CSV.

## Transfer between stages

Use one coordinator checkout of the same commit. After each training stage,
copy that stage's small records from each PC to the coordinator, preserving
the exact repository-relative layout:

```text
results/experiments/ssc_validation_search_v1/pc3/<stage>/<candidate>/<model>/SSC/seed_<seed>/
results/experiments/ssc_validation_search_v1/pc4/<stage>/<candidate>/<model>/SSC/seed_<seed>/
```

Include `FULL_RUN_CONFIG.json`, `LAUNCH_ENVIRONMENTS.json`,
`VALIDATION_RESULTS.json`, `validation_completed.txt`, `VALIDATION_SUMMARY.md`,
`TRAINING_SUMMARY.md`, and `logs/mlp_snn/version_0/metrics.csv`. These small
records permit selection without transferring all checkpoints. Keep best and
last checkpoints on the training PCs; checkpoint hashes are sealed in the
completion records. Selections verify validation evidence and source provenance;
each PC verifies its actual checkpoint metadata and hashes before final testing.
Do not overwrite an existing conflicting record during transfer.
Also copy each PC's `pc3/DATA_PROVENANCE.json` or `pc4/DATA_PROVENANCE.json`.
The workflow hashes the actual train/valid HDF5 files before user-launched
training and refuses changed files on restart or mismatched data across PCs.
Final tests record the test-file hash and evaluation environment; aggregation
rejects mixed test files or source revisions.

Run the corresponding coordinator selection command from `command.txt` only
after both PCs finish. It refuses missing evidence and writes an immutable,
checksummed JSON under the study root's `selection/`. Copy all selection JSONs
back to both PCs before the next stage. Keep the coordinator's earlier-stage
records for later selection. Selection dry runs are read-only; training dry runs
do not start training, test inference, locks or result writes.

Final test inference is forbidden until all ten confirmation jobs finish and
`freeze-test` seals their checkpoint identities. The test stage never calls
training. It writes `TEST_RESULTS.json` and automatically measures new test
completions beside each seed. Existing tests are verified and skipped. To retry
pending costs without repeating test evaluation or training, rerun the test
command with `--measure-existing-costs`.

After testing, merge the new `TEST_RESULTS.json`, `COST_RESULTS.json/.md` and
any `COST_ERROR.json` into the coordinator's matching seed paths. Run `report`
to write `CONFIRMATION_RESULTS.md` in the study root. Do not count validation
completion as final-test completion; this workflow never writes `completed.txt`.

## Reporting and subsequent decisions

Report the frozen winner and fixed reference over all five new matching seeds:
per-seed test accuracy, mean, sample SD, paired accuracy differences and their
95% confidence interval. Treat small-sample intervals cautiously. Preserve
validation and test results separately. Include ordinary-inference cost evidence
beside the seed directory and use signed paper deltas in current comparisons.
Do not combine historical development seeds with confirmation as if all were
independent confirmation.

The winner is tuned while the reference is fixed. This supports a comparison of
these configurations, not a claim that the neuron mechanism alone caused an
improvement. An architecture claim needs matched tuning opportunities for the
reference and relevant controls. A five-seed result can still be inconclusive.

If this grid fails, propose a separately versioned follow-up based on validation
diagnostics: model timescales, training-only input-gain augmentation, or a changed
LR schedule. Do not combine these changes in the first grid. Training-time
augmentation is a hypothesis requiring implementation and a matched control;
it is not justified solely by the historical TTA test improvement. Longer budgets,
larger architectures and objective changes likewise require separate protocols.
