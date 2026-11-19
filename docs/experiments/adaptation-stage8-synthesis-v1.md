# Stage 8: final saved-evidence synthesis

**Executed by the user and reviewed 17 November 2026; registered exploratory
synthesis complete.** All ten contract tests passed, all 16 report outputs and
16,814 source bindings verified, and both generated figures were inspected.
An independent arithmetic audit checked all 318 scalar seed aggregates and
the exported feature summaries, counts and original measurement records.
Read the [final synthesis](../../results/experiments/adaptation_stage8_synthesis_v1/SUMMARY.md)
and [completion review](../../results/experiments/adaptation_stage8_synthesis_v1/POST_RUN_REVIEW.md).
No rerun is needed. Stage 7 remains complete within its registered diagnostic
scope. This workflow needs no new
training, checkpoint evaluation, model forward, backward, feature extraction,
regression, bootstrap or official-test data access.

The [Persian report](../../supervisor-guideline-report-fa.md) also has a
[PDF edition](../../supervisor-guideline-report-fa.pdf), exported on 17 November
2026 with RTL layout, embedded regular/bold B Nazanin fonts, existing figures,
page numbers and landscape pages for wide tables. Export provenance is separate
from the frozen scientific ledgers in
`results/experiments/adaptation_stage8_synthesis_v1/persian_pdf_export_v1/`.
This is a presentation export only; the original Markdown and scientific
results remain unchanged. The local export helper uses Markdown, playwright,
PyMuPDF 1.26.7 and an installed Chrome/Edge browser. To recreate the PDF:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_supervisor_report_pdf
```

It discovers B Nazanin in installed system/user font directories. On another
machine, `--font-regular`, `--font-bold` and `--browser` accept explicit local
paths. The self-contained HTML uses embedded font/image data; no font files
need to be installed by readers of the PDF.

Agent preflight passed for all 54 cost/checkpoint/history identities, 25
completion ledgers, the archived SSC path index and changed-cwd/simulated
checkout relocation. [Recorded checks](../../results/experiments/adaptation_mechanism_study_v1/STAGE8_PREFLIGHT.json)
bind the exact prepared source files. Source syntax and Markdown links also
passed checks. The subsequent report/tests were user-run; the agent verified
saved artifacts and arithmetic without executing new scientific work.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_stage8.run --output-root results/experiments/adaptation_stage8_synthesis_v1
```

This single local command authenticates saved sources, runs the report contract
tests, produces the report/tables/figures and verifies its completion ledger.
Source hashing and portable-path verification may take several minutes. There is no new
long neural workload to distribute across PCs. Outputs identify the study,
dataset/model/seed and retain hardware only as historical provenance.

The exact input selection is pinned in
[the versioned specification](../../config/studies/adaptation_stage8_synthesis_v1.json).
It inherits the identity of `config/paper_baseline.yaml`; each saved resolved
configuration is checked against the corresponding authenticated stage-7 run.
The collector verifies 54 checkpoint identities, matching cost records and
metrics histories, including one final-test accuracy per history reproduced by
the saved ordinary-inference cost evaluation. Original ledgers and source files
remain unchanged. All classification paths use `project_paths.py` and
repository-relative artifact names, resolved against the checkout root.

## Outputs

The output directory contains:

- `SUMMARY.md`: the four requested sections—ordinary performance, input-feature
  distributions/patterns, feature–benefit associations and controlled
  experiments—followed by activity, compute and timing evidence.
- `figures/input_feature_distributions.png`: clean saved descriptors on separate
  native-unit axes, explicit finite/censored/missing counts; no feature re-extraction.
- `figures/saved_accuracy_and_timing.png`: individual training seeds using
  existing ordinary-test accuracy and saved timing sessions.
- `VALIDATION_PER_SEED.csv`, `COST_AND_TEST_PER_SEED.csv` and `ACTIVITY_EXTENTS.csv`:
  per-seed evidence, including stored versus batch-padded counts and environments.
- `ASSOCIATIONS.csv`, `INTERVENTIONS.csv`, `FEATURE_DISTRIBUTIONS.csv`,
  `DIAGNOSTIC_ORIGINS.csv` and `PAIRED_COST_DIFFERENCES.csv`.
- `COST_RECORDS.json` and `AGGREGATES.json`: original scopes/denominators,
  environment/repeat timings and mean/sample SD across matching seeds.
- `MANIFEST.json`, `CONTRACT_TESTS.txt`, `STAGE8_SCOPE_AUDIT.json` and
  `COMPLETE.json`: exact inputs/source hashes, execution checks and completion.

The report links the independently verified stage-7 plots with common color
limits and both predeclared optimization origins. It does not rewrite the
frozen generated stage-7 report, historical comparisons or prior summaries.
The four main comparison reports stay in `reports/`; this study's detailed
evidence and presentation stay together in its results directory.

## Interpretation rules

Validation, historical official-test and constrained diagnostic results are
separate. Source-high ordinary evidence remains distinct from stage-7
highest-precision sensitivities. No TTA or additional training seeds are mixed
in. No numeric paper-test baseline is compared to incompatible validation
metrics; existing paper comparison tables retain their own provenance.

The report copies SHD*/SSC intervals from their original separate feature and
intervention families; no new multiplicity guarantee or joint cross-dataset
inference is invented. ECG remains descriptive, with no patient intervals or
physical-millisecond inference. Shared parent predictions, speaker/window
replicates and optimization attempts are not independent training seeds.

Hidden activity counts L1/L2 binary spikes over stored sequence extents,
including saved minimum padding and burn-in. The output layer is a nonspiking
integrator. Feature support and gradient padding diagnostics use different
declared masks. Historical ECG first-projection accounting remains dense MACs
under its saved convention despite the independently verified signed-event
validation representation; no cost record is silently recounted.

Timing is forward-only and excludes data loading, transfer, supervised loss and
temporal aggregation. Recorded hardware is RTX 4090, batch 256 for SHD*/SSC and
64 for ECG, high matmul precision, two warm-up batches and three repeats.
Sessions/evaluator snapshots differ; figures and paired differences are
descriptive, not a controlled speed comparison. Workflow duration is not
inference latency. Estimated SynOps/logical operations are not measured energy.

All ineffective stage-7 input attempts and both origins remain visible. These
model-selected surrogate perturbations do not establish causal attention,
natural robustness or general architectural superiority. Additional CV-ISI,
temporal permutations, independent data, new learned-trace mediation or a
controlled hardware benchmark remain possible follow-ups, not executed claims.

## Preflight, verification and resume

Agent-safe commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_stage8.run --check-only
.\.venv\Scripts\python.exe -m scripts.adaptation_stage8.run --verify-only --output-root results/experiments/adaptation_stage8_synthesis_v1
```

`--check-only` authenticates existing artifacts/configurations and parses saved
inputs; it writes no results and runs no tests, aggregation or figure creation.
`--verify-only` requires a completed report, authenticating its outputs and all
source bindings without scientific execution. The ordinary command runs the
ten user-only report tests automatically; they can also be launched separately:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_stage8
```

The ordinary command resumes an interrupted compatible report. An OS-held
`.report.lock` prevents two writers; a released lock file may remain. Do not
delete it to bypass an active writer. A completed compatible report is verified
without regeneration. Changed sources/configuration fail the manifest guard:
preserve the old evidence and choose a new `--output-root`.

The completion audit, four report sections and both new figures have now
passed post-run review, and the roadmap records executed/verified completion.
For future versions, update it only after repeating those checks.
`stage8_scoped_complete` denotes the registered exploratory
reporting scope; it is not independent scientific confirmation or exact
upstream analytical-figure parity.
