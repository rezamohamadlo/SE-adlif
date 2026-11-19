# ECG validation alignment and signed-event deletion pilot

**Historical v1:** the user passed all 11 contracts, but the first clean anchor
failed and deletion did not start. Preserve this source/results directory.
The current command is the [four-pair v2 queue](adaptation-ecg-four-pair-queue-v2.md),
which matches training matrix precision and retains the strict clean gate.
The preparation notes below describe the original v1 implementation.

Prepared 14 November 2026. Implementation is ready; scientific execution and
contract tests are pending the user's launch. This is the next bounded task in
stage 6, before extending the four ECG mechanism studies. It requires no training.

The [study specification](../../config/studies/adaptation_ecg_alignment_pilot_v1.yaml)
inherits the ECG section of `config/paper_baseline.yaml`. It freezes all 18
selected checkpoints for MT, AMT, DTH, LT-DTH, SA-MT and MT-TC, across seeds
42/123/456. Preflight checks their actual SHA-256, checkpoint state/config/epoch,
complete 400-epoch validation histories and validation selection. All four pairs
must match outside their explicit neuron/objective differences. No existing
runner, model implementation, checkpoint or completed speech study is changed.

The next execution evaluates **only AMT/MT**, six checkpoints, on all 30 records
of the saved ECG validation split. It first checks every clean anchor against
the saved validation accuracy (tolerance 1e-6 fraction) and CE (tolerance .001).
All six must pass before deletion starts. A deterministic 10% event deletion
has three realizations, shared across both models and all training seeds.
The full queue has 24 prediction chunks and 360 paired record rows. These are
pilot diagnostics, not four completed ECG mechanism studies.

## Data and target contract

- Open only `data/QTDB/QTDB_train.mat`; do not instantiate `ECGLDM`, whose setup
  also opens the official test file and whose constructor can download data.
- The MAT contains 618 stored records, each 1301 steps by four channels.
  Inputs are signed events in {-1, 0, +1}. Delete nonzero events without changing
  the polarity of retained events. Do not use count thinning for these inputs.
- Reproduce the original split exactly: fresh Torch generator with split seed
  42, `randperm(618)`, train first 588 indices and validation last 30 in that order.
  All records fit in one saved batch of 64. No inference-based sample selection.
- Preserve the original temporal labels, `argmax(y, axis=-1)`, and one block
  per model step. All-zero class vectors occur in the MAT; the original loader
  maps them to class 0. They remain scored. This compatibility decision does not
  establish the physiological meaning of those vectors. No new exclusion mask
  is inferred from zero labels, quiet inputs, `max_i`, or test data.
- Saved burn-in is zero. The collator prepends an ignored dummy target (-1);
  any collator padding has ignored targets and block zero. A future nonzero
  burn-in requires an explicit protocol review; preflight fails closed.
- Shared supervised loss is CE after `summed_membrane_potentials` aggregation
  into temporal blocks. Compute per-record mean CE and accuracy, plus pooled
  step-weighted metrics separately. With equal 1301-step records they agree;
  neither time steps nor deletion realizations count as independent seeds.
- The MAT supplies no verified patient identities. Do not treat the 30 rows
  as independent patient clusters. Time is reported in model steps; physical
  milliseconds are not established from the numeric time grid alone.

The loader's input-only preflight compares three validation records with the
existing `QTDB_ECGWrapper` and `PadTensors`, without DataModule setup or model
forward. The readiness record also counts all-zero target positions explicitly.

## User command

From the repository root, in the project environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg.run --device cuda:0 --output-root results/experiments/adaptation_ecg_alignment_pilot_v1
```

This runs preflight, the targeted contract tests, six clean anchors, and then
the deletion pilot. Its size (30 records, 36-neuron layers) supports local
execution; there is no measured ECG-pilot duration yet. Start only one launcher
for this output root. The OS-owned `.analysis.lock` prevents concurrent writers.
Use the **same command and output root** to resume compatible completed chunks.
Manifest mismatches or damaged caches stop execution; no evidence is overwritten
to bypass a mismatch. Atomic JSON publication uses the existing scoped Windows
retry writer, with unique temporary files and no delete/in-place fallback.

Agent-safe preflight, without tests, inference, feature extraction or statistics:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg.run --check-only --device cuda:0
```

Optional separate user test command (the main command already runs these tests):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_ecg
```

Read-only completion verification:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_ecg.run --verify-only --output-root results/experiments/adaptation_ecg_alignment_pilot_v1
```

Outputs remain PC-independent and use repository-relative paths:
`MANIFEST.json`, `CONTRACT_TESTS.txt`, resumable `raw/*.json`,
`VALIDATION_ANCHORS.json`, `paired_records.csv`, `SUMMARY.json`/`.md`,
`PROGRESS.json` and hash-registered `COMPLETE.json`.

## After this pilot

Review all six anchors, target/block/input pairing, clean and transformed losses,
and the user test log. Then define ECG-specific input features and label-aligned
time scaling before extending the other three pairs. Regression, resampling,
patient-level uncertainty, direct SSC window diagnostics and stage 7 gradients
are separate pending work. No official-test evaluation or improvement claim
follows from this pilot.
