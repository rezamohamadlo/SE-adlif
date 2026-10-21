# TTA inference costs

`measure_tta_costs.py` measures existing DTH/MT official-test checkpoints without
training. The training runners and ordinary `COST_RESULTS` remain unchanged.
Run from the repository root with the local environment (this checkout uses
`.venv`; use `.ven` instead where that is the environment directory).

```powershell
.\.venv\Scripts\python.exe measure_tta_costs.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --dry-run
.\.venv\Scripts\python.exe measure_tta_costs.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456
```

SHD is reported as SHD*. The preflight verifies selected full-budget checkpoints,
validation selection, local datasets and matching saved TTA test evidence. It
does not run inference or write results. At measurement time the launcher checks
test-batch fingerprints and reproduced identity/TTA accuracy before publishing
a completion record. It neither downloads datasets nor starts training.

Each seed receives `TTA_COST_RESULTS.json` and `TTA_COST_RESULTS.md` beside its
training summary. JSON includes checkpoint/config identity, source hashes,
environment, data fingerprint, per-view activity, raw timing repeats and scopes.
The console prints per-dataset/model means and sample SD across unique seeds.
Existing matching measurements are reused. Changes to sources, environment,
evidence or timing settings require a new basename, e.g.
`--record-name TTA_COST_RESULTS_v2`; old records are preserved. One evaluator
holds the seed's `.evaluation.lock`; run commands sequentially and avoid other
GPU work when measuring timing.

## Interpretation

- `single_forward` and `tta_forwards` time uninstrumented model forwards only;
  TTA inputs are scaled before this timing. These match ordinary forward timing
  scope. Hooks used to count activity are removed before any timing.
- `single_pipeline` and `tta_pipeline` include input scaling, temporal output
  reduction, class softmax and probability averaging through the same evaluator
  function. The single-view pipeline uses its identity-only path. Neither time
  includes data loading/transfer, loss or final decision/metric bookkeeping.
- Warm-up defaults to two passes of each path, followed by three full-test
  timing repeats, synchronized on CUDA. Timing order alternates. Mean per-input
  milliseconds and raw repeats are saved. Across-seed SD includes session noise;
  this is not an energy benchmark or a hardware-controlled significance test.
- Dense GFLOPs count matrix MACs as two operations, summed across all three
  forwards with executed batch padding. Arithmetic GFLOPs also count the shared
  evaluator's scalar neuron arithmetic. Both exclude temporal/class probability
  aggregation, nonlinearities and comparisons. Gain scaling is recorded
  separately; these are explicitly scoped estimates, not total pipeline FLOPs.
- SynOps are ideal event-driven dense fan-out additions over stored sequence
  lengths, including minimum padding/burn-in and excluding extra batch padding.
  Input gain changes amplitudes, not original event multiplicity; the input
  event count is retained, while each view's hidden spikes are actually measured.
  SynOps/spikes sum all views per original input sequence, not per view. ECG
  analog input projection is excluded from SynOps and saved as dense MACs.
- Parameters are shared across views. TTA remains exploratory evaluation of
  previously observed test sets. Existing ECG reference stopping/budget limits
  remain; measuring TTA costs does not repair those comparison differences.

Test with `python -m unittest tests.test_measure_tta_costs` using the local
environment interpreter; these tests use tiny CPU models, never training.
