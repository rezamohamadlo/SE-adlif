# SSC MT/MT-TC direct output-window diagnostics

**Completed by the user and verified — 16 November 2026:** the
[saved-evidence review](../../results/experiments/adaptation_ssc_windows_v1/POST_RUN_REVIEW.md)
confirmed 256 output hashes, 79 source hashes, 12 passing user tests, six passing
clean anchors and zero changed ordinary predictions. All 234 chunks, 179658
descriptor rows and 89829 paired rows match their recorded coverage. The
recorded execution attempt was 380.953 seconds, excluding initial preflight.
Findings reverse at seed 456 relative to 42/123; no consistent superiority is
claimed. The later [read-only scope audit](../audits/supervisor-stage6-closeout-2026-11-16.md)
closes stage 6 within its registered exploratory scope and records stage 7 as
ready for preparation, not started. Frozen experiment-written completion flags
are preserved. Preparation notes below describe the completed workflow.

Prepared 16 November 2026. This is a separate stage-6 diagnostic, not training
or stage 7. All tests, forward passes and diagnostic/report calculations are
user-launched. Agents run preflight and read-only verification only.

Preflight passed for the authenticated source subset, all six local checkpoint
files/configs, the validation H5 and the three-record loader contract. The
readiness record marks tests, forward and descriptor calculations as unexecuted;
no new diagnostic result or anchor reproduction is claimed yet.

## User command

From the repository root, on the same local PC used for the SSC study:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ssc_windows.run --device cuda:0 --output-root results/experiments/adaptation_ssc_windows_v1
```

The command performs preflight, 12 user contract tests, source-evidence
preservation, six clean checkpoint passes, anchor checks and paired reports.
Use the exact same command/output root to resume. The original batch size and
order are retained: 39 batches per checkpoint, **234 resumable chunks** overall.
A model is loaded only if a missing chunk needs inference. Completed chunks
are checked without repeating their forward passes. An OS-owned lock prevents
two launchers writing the same output directory.

No training, hyperparameter tuning, official-test access, TTA, download,
bootstrap or scientific execution by the agent is involved. This workload is
one additional full-clean pass through each of six selected checkpoints; the
completed per-record loss caches do not contain temporal outputs and cannot
substitute for these forward passes.

The [versioned specification](../../config/studies/adaptation_ssc_windows_v1.yaml)
inherits the **SSC** paper baseline through the saved configs. It preserves
all 9,981 official validation records, split membership, seeds 42/123/456,
720-neuron layers, preprocessing, float32 matrix precision `high`, disabled
cuDNN benchmark and the supervised mean-membrane objective. MT-TC training-time
auxiliary JS is not added to any prediction loss here.

## Anchor and mask contracts

The completed [SSC source](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md)
and all frozen source code remain unchanged. Preflight verifies the six exact
checkpoint hashes, configs, selected epochs, full 40-epoch validation histories,
matched parent/variant settings, official-validation H5 hash and the existing
three-record wrapper/collator input contract, without calling a model forward.

New ordinary predictions are compared **record by record** with the completed
clean MT/MT-TC caches: ID, label, speaker, input hash, physical support, stored
length and predicted class must match exactly. Mean CE drift from the completed
cache must stay within the original .001 tolerance. Ordinary aggregate accuracy
and CE must independently reproduce the saved validation-selected metrics within
1e-6 accuracy fraction and .001 CE. Maximum per-record loss drift is recorded.
Any failure blocks report publication; it does not widen tolerances or replace
a checkpoint. Raw failed chunks remain evidence.

**Ordinary prediction loss** retains the saved block mask, including the
original minimum padding to 300 frames. Its burn-in and batch padding treatment
are unchanged. The **diagnostic mask** is deliberately different: outputs after
the first ten burn-in frames and before the end of each original framed input.
Thus descriptor windows exclude both minimum and batch padding. Burn-in inputs
still affect the forward state; no input is truncated for these diagnostics.

## Three exploratory window widths

Use nonoverlapping windows of **5, 10 and 20 frames**, stride equal to width.
The saved four-ms bin gives **20, 40 and 80 ms**. For every record/width,
drop and record an incomplete tail; require two complete windows. Retain all
records, with explicit undefined reasons when insufficient support or zero
true-class probability mass prevents a metric. Do not choose a width based on
the best outcome. These nonoverlapping diagnostic windows differ from the
overlapping stride-one training JS objective.

The [existing SHD metric implementation](../../scripts/adaptation_tc_windows/metrics.py)
is reused unchanged for probability descriptors. Frame softmax is evaluated
in float32; window means and diagnostic reductions use float64:

- Adjacent-window JS in nats, with its per-adjacent-window values.
- Mean true-class probability, per-window true-class probabilities and the
  normalized HHI concentration (zero for uniform temporal mass, one for mass
  in a single window). Zero total true-class mass gives undefined concentration.
- Accuracy of the **mean frame-softmax probability** in each window, its
  predicted classes, and adjacent top-class switch fraction.
- Mean supervised window CE: first average **raw membrane outputs** inside
  each window, then apply stable log-sum-exp CE. Record its per-window values
  and the corresponding mean-membrane argmax accuracy/classes separately.

Mean-probability argmax and mean-membrane argmax can differ. The two accuracy
columns are distinct, and neither replaces ordinary full-record accuracy.
Lower JS or fewer class switches may reflect stable incorrect/low-confidence
outputs. Interpret them jointly with true-class confidence, accuracy and CE.

## Coverage and summaries

There are 59,886 ordinary prediction records, **179,658 model/seed/record/width
descriptor rows**, and **89,829 paired record/seed/width rows**. These counts are
not independent training replicates. Summaries give each record equal weight
within model/seed/width, then report per-seed MT-TC-minus-MT differences and
mean/sample SD across the three training seeds. Each metric reports its common
finite paired-record count. Undefined metrics are never replaced with zero.

No p-values, confidence intervals, speaker bootstrap or automatic regression of
output-derived confidence/stability against loss benefit are produced. Such a
regression could be mathematically coupled to the prediction loss. This is an
exploratory clean-validation diagnostic after earlier result inspection, not
preregistration, causal proof or independent-test confirmation.

## Portable evidence and outputs

The queue preserves **12 authenticated source files** byte-for-byte: the suite
manifest/completion, the MT-TC evaluation manifest/completion/anchors, shared
clean input features and six clean reference prediction caches. A path index
under `shared/ssc_clean_evidence/` records that this is a **selected subset**, not
a full copy/reverification of all 230 historical SSC outputs. Completed source
manifests are never rewritten. Resumes can use the snapshot when the original
source directory is unavailable; the exact checkpoints and validation H5 must
still be transferred separately in their repository-relative layout.

Outputs are independent of PC assignments:

- `MANIFEST.json`, `CONTRACT_TESTS.json`/`.txt`, `PROGRESS.json`;
- authenticated `shared/ssc_clean_evidence/` subset and path index;
- `chunks/<model>/seed_<seed>/batch_<number>.json`, including selected per-window
  arrays; full frame logits and hidden-state histories are not retained;
- `CHUNK_INDEX.json`, six `ANCHORS.json` records;
- `SAMPLE_WINDOWS.csv` (179,658 rows) and `PAIRED_WINDOWS.csv` (89,829 rows);
- `SUMMARY.json`/`.md` and hash-registered `COMPLETE.json`.

Raw window arrays can make this package substantially larger than the old
per-record prediction caches. CSVs are streamed rather than retaining all raw
window arrays in memory. Atomic JSON writes use the existing bounded Windows
retry writer; temporary derived CSVs are published only after all anchors pass.
Record timing/progress stays with the result package; there is no measured
duration for this new diagnostic yet.

Agent-safe preflight:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ssc_windows.run --check-only --device cuda:0
```

Optional user contract command (already included by the main command):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ssc_windows
```

Read-only verification after successful execution:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ssc_windows.run --verify-only --output-root results/experiments/adaptation_ssc_windows_v1
```

The [preflight record](../../results/experiments/adaptation_mechanism_study_v1/SSC_WINDOWS_PREFLIGHT.json)
distinguishes data/config checks from unexecuted tests/forward/descriptors.
After user execution, verify coverage, anchors and descriptor quality, then
complete the outstanding study/code scope audit before deciding to close
stage 6. The queue does not automatically mark stage 6 complete or start stage 7.
