# SSC AMT/MT validation pilot v1

Prepared, user-executed and reviewed on 2026-11-12; **all six full clean anchors
passed, and the 256-record pilot is complete**. Unit-test execution remains
unverified. See the [saved-evidence review](../../results/experiments/adaptation_ssc_amt_pilot_v1/POST_RUN_REVIEW.md).
This is the first bounded SSC extension of the
mechanism study (roadmap stage 6). Stage 7 gradient analysis has not started.
The [readiness audit](../audits/ssc-mechanism-readiness-2026-11-12.md) and
[study definition](../../config/studies/adaptation_ssc_amt_pilot_v1.yaml)
record the frozen dataset and checkpoint identities.

## Scope and compatibility gates

- Compare existing AMT/MT checkpoints at training seeds 42, 123 and 456, using
  ordinary frozen inference. No training, checkpoint selection or TTA occurs.
- First reproduce clean loss and accuracy on **all 9,981 official validation
  records for each of the six checkpoints**, in the original batch order.
  Every anchor must pass before feature or pilot diagnostics proceed.
  Accuracy tolerance is 1e-6 (fraction); loss tolerance is .001. A failure stops
  the workflow; inspect preprocessing, masks and aggregation rather than
  replacing a checkpoint or relaxing tolerances to obtain a preferred result.
- Then evaluate a fixed **256-record pilot subset**: clean input, count-preserving
  time scaling .9 and 1.1, and 10% per-event deletion with three deterministic
  realizations. Selection is class-balanced round robin with seed 20261112,
  independent of predictions, and retains official file order. This is a
  diagnostic subset, not a prevalence-weighted full-validation estimate.
- Shared feature rows and input hashes enforce identical inputs between the
  models and training seeds. Replicates stay distinct and are averaged within
  seed before descriptive summaries across training seeds. Report means and
  sample SD; no significance test, bootstrap or causal conclusion is produced.
- Report headers and metadata explicitly separate the pilot from the full
  clean anchors. Do not pool SHD and SSC losses or claim final-test evidence.
  Wider SSC diagnostics, other model pairs, ECG and gradient work remain pending.

The loader opens only `data/SSCWrapper/ssc_valid.h5`, verified against the
study's SHA-256. It calls neither a Tonic dataset constructor nor `SSCLDM.setup`;
it does not download data or open train/test H5 files. Raw float16 seconds widen
to float64 **before** multiplication by 1e6, matching the current SSC wrapper.
Downsampling and framing match saved configs, including the unused sensor y
dimension used by Tonic's one-dimensional framing. Input-only preflight compares
the tensors, targets and block masks against the current wrapper on IDs
0, 4990 and 9980, without constructing a dataset or forwarding a model.

The study inherits the **SSC section** of `config/paper_baseline.yaml`:
LR .006, batch 256, 40 completed training epochs, 720 neurons per hidden layer,
140 input channels, 35 classes, four-ms bins, minimum 300 frames, ten-frame
burn-in, `mean_membrane_potentials` and `ssc_mean_loss_v1`. SHD holdout settings
are rejected. Feature definitions alone are reused from the earlier pilot:
20/200-ms windows on physical count-frame support before minimum/batch padding.
Physical support, raw event duration and stored padded length remain separate.

## User commands

From the repository root, with the recorded data and six checkpoints present:

```powershell
# Preflight only; agents may run this. No model forward or result writes.
.\.venv\Scripts\python.exe scripts/adaptation_ssc/evaluate.py --check-only

# User launches the complete bounded pilot on one local GPU.
.\.venv\Scripts\python.exe scripts/adaptation_ssc/evaluate.py --device cuda:0

# Verify saved hashes after completion, without rerunning inference.
.\.venv\Scripts\python.exe scripts/adaptation_ssc/evaluate.py --verify-only

# User-launched contract tests; these have not been run by the agent.
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ssc
```

Based on saved RTX 4090 forward measurements for these six checkpoints
(1.56–2.15 ms/sample), 69,102 sample-forward evaluations imply about 128.5 seconds
of forward work at the measured rates. Those measurements used saved test cost
records and exclude data loading, preprocessing, aggregation, features, reports
and startup. This does not access the test dataset or establish pilot runtime.
**Budget 10–20 minutes, allowing up to 30 on a comparable idle RTX 4090**;
this is an unverified planning estimate, not a performance result. Slower GPUs,
CPU execution, disk speed and concurrent workloads can change it substantially.
The completed pilot will record actual elapsed time. A single local PC is
reasonable for this bounded workload; no fixed multi-PC scheduling is assumed.

**Measured completion:** the user's RTX 4090 run recorded 412.344 seconds
(6 minutes 52 seconds) in `COMPLETE.json`, covering the launcher execution after
startup preflight. The 63 saved output hashes and all six clean anchors passed
review. The timing estimate above is retained as the historical planning estimate.

`--seeds 42` (or another unique subset) limits checkpoints, retaining the same
256 pilot samples and full clean anchors for those selected checkpoints. Default
partial-cohort outputs use `seed_cohorts/42/` (or joined sorted seed IDs), never
PC names. Reports explicitly mark missing members of the three-seed cohort.
Do not present a partial cohort as the completed study or merge summaries by
copying files. `--verify-only --seeds 42` selects that same cohort. Choose cohort
scheduling before launch; an existing parent study and nested cohorts must not
be run simultaneously. Consolidation of separate cohorts is not implemented.

Changing `--samples` (bounded at 512) requires a separate explicit
`--output-root results/experiments/<new-study-name>`. Keep default samples for
the current study. A manifest/config/source/device mismatch is a preservation
guard: retain existing evidence and choose a new root instead of deleting the
manifest. Use the same command to resume an interrupted unchanged run; verified
complete chunks are reused, while an interrupted chunk restarts. One launcher
owns an OS-managed lock for each destination; stale lock-file presence alone is
not a reason to remove it.

## Outputs, portability and verification

The default root is `results/experiments/adaptation_ssc_amt_pilot_v1/`:

- `MANIFEST.json`: resolved SSC study, fixed sample IDs, all validation IDs,
  validation-file identity, saved configs/metrics/checkpoint hashes, source
  hashes (CRLF normalized to LF), environment and three-record loader preflight.
- `VALIDATION_ANCHORS.json` and six `*_full_validation.json` chunks: full clean
  anchor results and differences against saved selected-epoch validation metrics.
- `diagnostic_rows/`: six shared feature chunks, 36 pilot prediction chunks,
  and 4,608 paired CSV records when all three training seeds finish.
- `analyses/{clean,time_scaling,spike_deletion}/`: subset summaries in JSON/MD
  and descriptive PNG/SVG figures; no PC assignment enters result identities.
- `PROGRESS.json` and `COMPLETE.json`: live phase/elapsed time, completion scope,
  seed-cohort completeness and checksums for the 63 saved evidence outputs.

All persisted project paths resolve through `project_paths.py` against the
checkout root, independently of the current working directory. Transfer the
entire results root, including diagnostic rows, for hash verification; Git does
not transfer ignored datasets or checkpoints. Inference on another checkout
requires those exact recorded artifacts under the same relative paths. Hardware
is provenance; a different environment requires its own output root under the
strict manifest guard. Previous SHD results and historical manifests are retained.

Agent preparation covered real data/checkpoint/config preflight, the three-record
input contract, syntax and path checks. After the user completed execution,
read-only review verified output/source/cache/checkpoint identities, saved anchor
agreement, paired records and summary consistency. The agent did not run inference,
features, statistical fits, resampling or tests. This completes the bounded pilot;
all-condition full-validation diagnostics and wider stage 6 remain pending.
