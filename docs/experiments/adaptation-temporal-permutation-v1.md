# Temporal-window permutation v1 — pilot completed, full pending

The user executed the SHD* pilot on 19 November 2026. Read-only verification
authenticated all 1,267 registered outputs, including 1,260 score chunks; the
11 contract tests and 18 clean anchors passed. See the
[pilot review](../../results/experiments/adaptation_temporal_permutation_pilot_v1/POST_RUN_REVIEW.md).
Scientific pilot review and explicit approval for full execution remain pending.
ECG remains blocked. The preparation checks and original protocol below are
retained; preparation-only statements describe the earlier preparation session.

Prepared 19 November 2026 for one PC. No experiment, neural forward, training,
statistical analysis or test-suite command was run by the preparation agent.
This is an exploratory validation extension; the existing final Persian report,
stage-8 outputs and completed-study manifests remain unchanged.

Preparation checks passed: static parsing of all five new Python files; read-only
pilot preflight (40 SHD* records); full prerequisite preflight (1,631 SHD* and
9,981 SSC records, ECG explicitly blocked); and pilot preflight after changing
the process working directory to the system temporary directory. No permutation
results directory was created. The contract tests and all scientific execution
remain pending user launch. These checks do not establish model anchor agreement.

## Scientific source and anchors

The integrated checkout contains six shared scientific files that differ from
the completed study. The extension therefore uses Git-backed imports from the
explicit full source commit `f0bf2e16e56d956408b06b3e862e4701469f7451`, rather than
pretending that the integrated implementation has the historical identity.
The loader authenticates LF-normalized historical source hashes against the
registered manifests, records every integrated-source mismatch, and records the
hashes of the frozen modules actually loaded. It refuses pre-imported scientific
modules: always start a fresh module-mode process. It never rewrites source files,
patches model implementations, changes frozen manifests or disables their guards.
Its module `__file__` points into the live repository for portable artifact lookup;
the executed code bytes come from the authenticated Git commit.

The extension itself has separate raw-byte source/config/test hashes and an
integrated Git revision in its new manifest. Preserve these files unchanged when
resuming. `.gitattributes` protects their bytes across Windows checkouts. A clone
must contain the source commit and the exact ignored datasets, checkpoints and
metrics required by the registered manifests; Git alone does not transfer them.
See the earlier [publication audit](../transfers/input-statistics-study-git-2026-11-19.md).

Validation membership and selected checkpoints come from the SHA-256-pinned
stage-7 manifest and authenticated stage-6 source packages. Each checkpoint's
hash, selected epoch, resolved saved configuration, dataset-specific paper baseline
and metrics hash must match. Only the official training HDF5 for SHD*, official
validation HDF5 for SSC, and saved validation identities are used. No official-test
file or data-module setup is invoked.

All 18 unique model/seed clean anchors for every selected executable dataset are
evaluated on its **complete validation set** before any permutation is evaluated.
Original batch membership, order, batch size, minimum padding, loss mask, and
dataset-specific supervised aggregation are retained. The same ordinary
`MLPSNN.forward`/temporal aggregation is reused; MT-TC's auxiliary JS training
penalty is excluded. Float32 matmul precision is `high`, cuDNN TF32 is enabled,
and cuDNN benchmarking is disabled. Actual flags and software environment are
saved and bound on resume. Accuracy tolerance is `1e-6` (fraction), CE tolerance
`0.001`, preserving the existing contracts. Failures and measured discrepancies
are saved in `ANCHORS.json`; failures prevent all perturbed forward calls.

## Registered transformation

SHD* and SSC use native physical count frames before minimum/batch padding.
For each record, leave its first ten physical burn-in frames unchanged. Partition
the remainder into nonoverlapping windows of 5, 10 or 20 frames: 20, 40 or 80 ms
at the saved 4 ms/frame. Independently permute complete windows with one shared
order across every channel; leave any incomplete final window unchanged.

Length, dtype, the complete within-window content, per-channel totals, total event
count, class target and every block/mask entry stay fixed. No synthetic padding
is permuted. A nonidentity order is sampled uniformly by rejection whenever at
least two windows exist. An order change can nevertheless produce identical data
(zero/repeated windows); these are explicitly `ineffective`, while insufficient
windows are `ineligible`. Neither is discarded. Audit records include eligible
regions, actual source-window order, trailing support, seed, input hashes and
per-channel event counts. Zero, short and empty inputs have explicit contracts.

Three replicates are registered per width. The seed is the first eight SHA-256
bytes, little-endian, of JSON `[20261119, dataset, original_sample_id, width,
replicate]`. Training seed, model, pair, PC and output root are absent. Parent and
variant receive identical transformed inputs, verified through matching complete
transformation records. Different training seeds also use the same transformation.

The pilot selects two records per each of 20 SHD* classes using existing
outcome-independent selection seed `20261116`, retaining validation order. Its
exact 40 IDs are printed/read-only checked and saved in the new manifest. To
preserve original batch numerical structure, score selected pilot records in
their **original full validation batches**, leaving other batch inputs clean.
Thus the pilot is scientifically small but not necessarily proportionally cheap.
Full cohorts are 1,631 SHD* and 9,981 SSC validation records.

Registered pilot IDs (original training-file row IDs, in validation order):
`1361, 3896, 6851, 2299, 1947, 5437, 3602, 6449, 1023, 6884, 1313, 6456,
2972, 759, 4158, 8070, 1717, 873, 4001, 7281, 2381, 7123, 2932, 3272,
5822, 3181, 4182, 1814, 7128, 3947, 6621, 7601, 5, 7533, 6170, 7916,
3693, 4203, 2267, 4581`.

### ECG is explicitly blocked

ECG's 30 validation records have signed inputs, saved per-timestep labels and
per-timestep scoring block IDs. Those IDs do not establish independent,
scientifically reviewed temporal-label regions for this new intervention.
All-zero source class vectors also map to class 0 under the original policy.
The extension does not infer that every constant argmax run is an interchangeable
clinical region. ECG execution is therefore **blocked pending a defensible
independent region/boundary contract**, rather than moving labels or silently
omitting the dataset. Full defaults include ECG and write this blocked status;
`registered_scope_complete` stays false when ECG is requested. Explicit
`--datasets ECG` records a blocked scope without ECG inference.

The pure signed-transform implementation and user-run tests support supplied
constant-label runs, preserve positive/negative counts per channel and leave
labels unchanged. Passing those numerical tests will not remove the scientific
ECG blocker. ECG widths would be model steps, not authenticated milliseconds.

## Comparisons and interpretation

Four pairs: AMT/MT, MT-TC/MT, SA-MT/MT and LT-DTH/DTH; saved training seeds
42, 123 and 456. Store clean/permuted predictions, accuracy, supervised CE,
input event totals, channel counts and actual transformation per record/seed.
`benefit = CE(parent) - CE(variant)`;
`delta_benefit = benefit(permuted) - benefit(clean)`.
Positive delta means improved *relative* variant loss benefit; it does not imply
that either model's absolute performance improved.

Average the three permutation replicates within a sample/seed, average records
within that seed, then report individual training seeds, mean and sample SD
across the three seeds. The summary includes clean/permuted CE and accuracy for
both sides. Shared MT checkpoints are reused across pairs, never extra seeds.
No bootstrap, p-values or population intervals are introduced. Negative, zero,
ineffective, failed and inconclusive results are retained.

Permutation changes long-range order, window transitions, autocorrelation and
possibly class meaning while preserving numerical class labels. Counts are
controlled; temporal features are not manipulated one at a time. Burn-in and
trailing support remain unperturbed. Effects depend on width, particular windows,
training seed and dataset, and cannot prove a causal adaptive mechanism or
independent-test improvement.

## Resume and completion

Only direct children named `results/experiments/adaptation_temporal_permutation_*`
may be written. Pilot/full roots must differ. An OS-held lock prevents concurrent
writers; lock-file presence alone is not permission to delete it. Every resumable
original-batch chunk binds the complete new manifest, checkpoint, original batch
IDs, selected IDs and condition, plus a value checksum. Corrupt/incompatible
chunks fail; they are never silently recomputed. All clean anchors must pass on
each resume before any cached or fresh perturbed results are used. Weights/state
are frozen and authenticated before/after model use.

Normal execution first runs the user contract tests. `FAILED.json` retains
execution failures; chunks remain resumable. `COMPLETE.json` authenticates the
manifest, runtime, test log, anchors, all score chunks, paired results, summary
and loaded-source audit. `--verify-only` authenticates these saved files and source
identities without model execution or fresh statistics. It does not require an
experiment rerun and reports blocked scope explicitly.

Full execution requires a human pilot review. With the pilot complete, inspect its
anchors, transformations, ineffective counts and paired results before writing
`PILOT_REVIEW.json`. The review must name a reviewer and bind the exact completion
checksum; it is saved in the full manifest. An approval does not imply a positive
scientific result. Resume uses exactly the original phase, dataset list, device
and output root. For the review gate, retain the standard pilot root below.

## One-PC user commands (repository root, Windows PowerShell)

Read-only pilot preflight, no models/tests/output generation:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase pilot --datasets SHD --device cuda:0 --check-only
```

User-run contract tests, then pilot execution (execution also repeats these tests):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_temporal_permutation
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase pilot --datasets SHD --device cuda:0 --output-root results/experiments/adaptation_temporal_permutation_pilot_v1
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase pilot --output-root results/experiments/adaptation_temporal_permutation_pilot_v1 --verify-only
```

After actual pilot review, explicitly record your decision (replace the name):

```powershell
$pilotRoot = 'results/experiments/adaptation_temporal_permutation_pilot_v1'
$review = @{ approved_for_full = $true; reviewer = 'YOUR NAME'; pilot_complete_sha256 = (Get-FileHash -LiteralPath "$pilotRoot/COMPLETE.json" -Algorithm SHA256).Hash.ToLowerInvariant() }
$review | ConvertTo-Json | Set-Content -LiteralPath "$pilotRoot/PILOT_REVIEW.json" -Encoding UTF8
```

Full preflight and execution, sequentially on the same PC; ECG remains explicitly
blocked and SHD*/SSC form the executable scope. This can require hours with input
loading/JSON storage; no wall-clock promise is made from historical model timing.

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase full --datasets SHD SSC ECG --device cuda:0 --check-only
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase full --datasets SHD SSC ECG --device cuda:0 --output-root results/experiments/adaptation_temporal_permutation_full_v1
.\.venv\Scripts\python.exe -m scripts.adaptation_temporal_permutation.run --phase full --output-root results/experiments/adaptation_temporal_permutation_full_v1 --verify-only
```

Keep the same execution command to resume. Neither an implementation nor a
successful preflight constitutes executed pilot/full scientific evidence.
