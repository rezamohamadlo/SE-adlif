# Official-test reference forward-cost measurements

The SSC and ECG reference models are evaluated on training seeds 42, 123 and
456, using their existing best validation-selected checkpoints. No weights,
training settings, saved training summaries or accuracy histories are changed.
The retained ECG runs remain unmatched in training budget/stopping; measuring
their test-set costs does not correct that mismatch.

```powershell
.\.venv\Scripts\python.exe measure_reference_costs.py --dry-run
.\.venv\Scripts\python.exe measure_reference_costs.py --reports-only
.\.venv\Scripts\python.exe -m unittest tests.test_measure_reference_costs tests.test_measure_shd_costs
```

Each `<dataset>/seed_<seed>.json` saves the exact checkpoint hash and epoch,
resolved configuration, evaluator/model-source hashes, ordered test-batch hash,
software/device details, raw counts, operation profiles and three timing repeats.
`AGGREGATE.json` combines these with existing SHD* reference measurements.
`PER_SEED.csv` exports the six new per-seed measurements. `EVALUATOR_SOURCE.py.txt`
preserves the exact evaluator used for inference; after measurement, only report
wording/links and a reports-only refresh option were corrected in the root script.
The raw measurement hashes and values are preserved. The original benchmark
command was `measure_reference_costs.py --update-reports`; to perform another
benchmark with the current script, choose a fresh `--output` directory.
Matching measurements are reused; changed code/checkpoints/data require a new
`--output`. An OS lock prevents concurrent writers. Checkpoints and datasets
must already be available locally; no automatic downloads or training occur.

| Quantity | Counting scope |
|---|---|
| Dense matrix GFLOPs/sample | Two FLOPs per feed-forward/recurrent dense matrix MAC, all executed padded test timesteps |
| Matrix + scalar arithmetic GFLOPs/sample | Dense matrix count plus cell-forward add/subtract/multiply/divide, negation, reciprocal, bias additions and sum/mean reduction arithmetic; nonlinear functions/comparisons excluded |
| Matrix + all logical scalar GOp/sample | Dense matrix count plus all counted cell-forward scalar operations; each nonlinear function/comparison counts as one logical operation, not its hardware FLOP expansion |
| Uncounted work | Tensor allocation, indexing, copies, memory traffic, loss, temporal prediction aggregation, backward pass and host scheduling; these are not exact end-to-end FLOP totals |
| Hidden spike counts | Binary L1/L2 outputs before dropout; transformed sequence extent before batch padding, including preprocessing minimum padding and burn-in |
| Firing fraction | Hidden spikes / neuron-timestep opportunities; not spikes per second |
| SSC input events | Framed event multiplicity retained when estimating dense-fan-out event-driven SynOps |
| ECG analog input | Input amplitudes are not counted as spikes. Its first projection is dense MACs, stored separately in `analog_input_dense_macs_per_sample` and already included in dense GFLOPs |
| ECG event-driven SynOps | Only hidden-spike-driven projections/recurrence; exclude the analog-input projection. Do not compare this directly with an all-event-input SynOps total without that qualification |
| Recurrent boundaries | Previous spikes at each destination timestep; no recurrence for final spikes lacking a subsequent destination timestep |
| Padding | Separate pre-batch sequence and executed batch-padded counts are retained; compute/runtime includes all executed padding |
| Runtime | Uninstrumented `MLPSNN.forward`, synchronized device wall time, two first-batch warm-ups then three full-test repeats; excludes data loading/transfer, loss and temporal aggregation |
| Accuracy check | Reproduced saved final test accuracy within 1e-6; ECG accuracy is per target timestep, compute is per input sequence |
| Aggregation | Mean and sample SD across three distinct training seeds; dataset-specific batch sizes/preprocessing retained |

The arithmetic and logical totals provide broader cell-forward accounting than
matrix-only GFLOPs, but nonlinear costs have no universal hardware-independent
FLOP equivalent. No energy measurement is performed. SynOps estimate ideal
event-driven additions; the current GPU implementation still runs dense matrices.
Keep the separate column labels when reporting these results.

The paper-versus-reference report retains historical **validation-shape** dense
estimates separately from these **test-shape** measurements. Published paper
results are unchanged. The original SHD* evaluator and its measurement records
are preserved.
