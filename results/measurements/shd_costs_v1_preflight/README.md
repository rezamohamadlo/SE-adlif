# SHD* checkpoint activity and cost measurements

These records evaluate the canonical seven models on training seeds 42, 123
and 456. They perform no training or checkpoint selection. Each run loads its
existing validation-selected checkpoint and checks that its official-test accuracy
reproduces the saved accuracy to within 1e-6 before accepting the measurements.

```powershell
.\.venv\Scripts\python.exe measure_shd_costs.py --dry-run
.\.venv\Scripts\python.exe measure_shd_costs.py --update-reports
.\.venv\Scripts\python.exe -m unittest tests.test_measure_shd_costs
```

Run these from the repository root. The script resolves artifact paths against
that root even if the caller's working directory differs. A held OS lock prevents
two evaluators writing the same output. Matching completed records are reused;
different checkpoints, code, device settings or data require a new `--output`.
The `.evaluation.lock` file may remain after completion; its existence does not
mean the OS lock is held. No checkpoints, training summaries or accuracy logs
are overwritten. Git does not transfer ignored checkpoints or datasets.

Each `<model>/seed_<seed>.json` contains the checkpoint path and SHA-256, its
resolved configuration, implementation/model-source hashes, test-batch hash,
environment, timing repeats, raw counts and normalized metrics.
`AGGREGATE.json` collects the 21 records after successful completion. Accuracy
is a consistency check, not a new criterion for selecting checkpoints or models.

| Measurement | Definition |
|---|---|
| Parameters | Total and trainable PyTorch parameter elements, excluding buffers |
| Hidden spikes | Binary outputs of L1/L2 before dropout, counted at every stored sequence timestep |
| Sequence scope | Transformed sample length **before batch padding**, including configured minimum padding to 250 frames and the initial ignored classification frames; this is not the original event duration |
| Padded scope | Every timestep actually executed after samples are padded to their batch maximum; includes additional padding-driven neuron activity |
| Spikes/sample | Total counted spikes divided by 2,264 official test samples; both sequence and padded raw totals are retained |
| Firing fraction | Sequence-scope spikes / (sequence-scope sample timesteps x layer neurons), expressed as a percentage; not spikes per second |
| Synaptic operations | Ideal event-driven addition estimate for full fan-out: framed input event multiplicity and hidden spikes times receiving neurons, including actual previous-step recurrence; no pruning of zero-valued weights |
| Recurrent boundary | Count previous spikes at each destination timestep; final outputs with no next timestep cause no recurrent operations |
| Dense GFLOPs | Two operations per dense matrix MAC, evaluated over batch-padded test lengths; state operations excluded. Existing report estimates over validation lengths remain separately labeled |
| Non-matrix operations | Logical scalar operations in actual cell forward code, on all padded timesteps, including parameter transforms, bias additions, state/threshold updates and surrogate forward arithmetic |
| Logical scalar convention | Each arithmetic, comparison or nonlinear scalar operation counts as one; reduction sum costs N-1 additions, mean costs N-1 additions plus one division; softmax costs 5N-2 per row (max, subtract, exp, sum, divide). These are **not hardware FLOPs** |
| Exclusions from scalar count | Matrix multiplies, tensor allocation, indexing, copying, memory traffic, loss, output temporal aggregation and backward pass |
| Runtime | Synchronized wall time for uninstrumented `MLPSNN.forward`, after two first-batch warm-ups; three full-test timing passes; batch size 256 with final partial batch; inputs transferred before timed regions |
| Aggregation | Arithmetic mean and sample SD across the three training seeds; not across three timing repeats |

Neuron-operation profiles account for the initial singleton membrane state,
batch-shared parameter transforms and multi-rate update phase. They are checked
for affine batch scaling and periodicity, and tested against an independently
instrumented full forward for all seven cell types. Spike hooks are tested to
preserve outputs and agree with manually collected spike tensors.

The current implementation executes dense matrices even when firing is sparse.
Event-driven SynOps and dense GPU runtime describe different execution models.
Do not add logical operations to SynOps and call the sum FLOPs or energy. No
power measurement is performed. Runtime is a local software/hardware benchmark,
including Python/kernel-launch overhead; it can vary with desktop activity and
must not be interpreted as neuromorphic latency or evidence of energy savings.

Results update `reports/reference-vs-models.md`, the SHD* local-reference activity
row in `reports/paper-vs-reference.md`, and the LG-DTH aggregate result report.
Published paper results remain unchanged. SSC and ECG activity remain pending.
