# Independent voltage sensitivities in MT: SSC experiment

`VS_MT_SE_adLIF` tests one change to MT's fast and slow adaptation memories:

```text
fast_drive = q * (a      * post_reset_membrane + b * current_spike)
slow_drive = q * (a_slow * post_reset_membrane + b * current_spike)
```

The new coefficient is a trainable vector with one entry per hidden neuron.
It starts as an independent clone of the freshly initialized `a`, without
additional random draws or trained weights. Equal coefficients recover MT's
forward/state dynamics. Constraints use the inherited `a_range`; resetting
parameters copies the new `a` into `a_slow` again. There is no separate spike
coefficient. The fast/slow decays, static mixture, reset, surrogate gradient,
shared `b` and `q`, recurrence and update order remain inherited from MT.
Two 720-neuron hidden layers add 1,440 trainable parameters and no new states.

This is a voltage-sensitivity hypothesis, not a change to optimizer learning
rate or memory decay. Previous test results have already been observed; do
not select architectures or hyperparameters using these test accuracies.
No improvement, biological specialization or novelty is guaranteed.

## Configuration and lifecycle

[SSC_VS_MT_SE_adLIF.yaml](../../config/experiment/SSC_VS_MT_SE_adLIF.yaml)
inherits MT's architecture and the SSC comparison group, which inherits
[paper_baseline.yaml](../../config/paper_baseline.yaml). Training seeds are
42, 123 and 456. The resolved current baseline uses LR 0.006, batch 256,
40 epochs, no early stopping, Adam, plateau scheduler with patience 9999 and
factor 0.9, official train/validation/test partitions and mean-membrane loss
under `ssc_mean_loss_v1`. Authoritative configs determine future values;
the launcher accepts no tuning overrides.

[runner_vs_mt_ssc.py](../../runner_vs_mt_ssc.py) restricts execution to SSC and
delegates to the official `runner.py` for preflight, training, manifest checks,
resume, validation checkpoint selection, final testing, aggregation and costs.
Existing MT and ordinary runner defaults are unchanged. A lock serializes
instances of this wrapper using its fixed output root. Never bypass a held
lock or launch a separate generic runner into the same root concurrently.

New jobs start from epoch zero with fresh random weights and optimizer state.
The same command resumes each interrupted job only from its own `last.ckpt`
and skips completed jobs. A failed startup without a checkpoint cannot be
called continuation. Changed configs or unknown existing artifacts are rejected.
Fully resolved manifests include model identity, baseline version/section,
training and split seeds and the model-variant deviation. The test set is
evaluated using the best validation-selected checkpoint, not necessarily last.

## Commands

Run from the main repository root, not the historical PC3 resume checkout.
Only the user launches training. Prefer waiting for the active PC3 GPU queue
to finish before starting this experiment.

```powershell
.\.venv\Scripts\python.exe runner_vs_mt_ssc.py --datasets SSC --seeds 42 123 456 --dry-run
.\.venv\Scripts\python.exe runner_vs_mt_ssc.py --datasets SSC --seeds 42 123 456
```

Selecting a subset such as `--seeds 42` is supported. Other datasets and seeds
are rejected. The wrapper checks local SSC HDF5 files before training; Git
does not transfer datasets or checkpoints. Data, output and artifact paths
resolve against the repository root rather than Hydra's working directory.

Read saved results and parameter arrays without training or test inference:

```powershell
.\.venv\Scripts\python.exe runner_vs_mt_ssc.py --report-only
```

New completions automatically receive activity/compute/runtime measurement.
Cost failures remain separate from training completion and never require
retraining. To retry missing costs on completed runs, rerun the normal command
with `--measure-existing-costs`; completed training jobs remain skipped.
`--no-measure-costs` explicitly disables new measurement. Report-only mode
does not measure costs. The evaluator's dynamic scalar-operation profiler and
spike hooks support this registered cell without a special counting formula.
Costs do not measure energy or establish a hardware-independent speedup.

## Evidence and comparison

Outputs stay under `results/experiments/vs_mt_ssc_v1/`. Per-seed manifests,
metrics, summaries, checkpoints and `COST_RESULTS.json`/`.md` stay inside
`VS_MT_SE_adLIF/SSC/seed_<seed>/`.

`COMPARISON.md`/`.json` compare ordinary-inference test results against compatible
existing `results/valid/MT_SE_adLIF/SSC/seed_<seed>/` runs. The wrapper validates
effective configs, reports each requested seed, completed counts, means,
sample SDs, paired differences over matching completed seeds, and available
costs. It never retrains the baseline. Pending scores remain pending. Test JSON
and CSV disagreements or mismatched cost checkpoint identities are errors.
Different timing sessions/environments limit runtime comparisons.

MT is inspected as read-only evidence, not as a training/resume job. Compatible
resolved configs and verified `RESULT.json` records can supply the baseline
comparison without local MT checkpoints, completion flags or CSV logs. Where
CSV/test JSONs are local they must agree with the result record. Costs must
match its checkpoint hash and test score. Missing evidence is labeled
unavailable and mismatched configs are excluded; neither schedules baseline
training. Missing MT weights only prevent baseline parameter export.

Each completed VS-MT seed also receives `NEURON_PARAMETERS.json` from its
validation-selected checkpoint. Aligned neuron arrays include `a_fast`,
`a_slow`, shared `b`, fast/slow mixture weights, mixture-weighted voltage
coefficients, membrane/fast/slow time constants, `q` and `dt`. The comparison
JSON includes both models' parameter records and exact checkpoint hashes.
Inspect these jointly; raw coefficient differences alone do not establish a
causal explanation. These arrays are read from trusted local checkpoints and
do not require inference.

## Non-training verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vs_mt_adlif tests.test_vs_mt_ssc_runner
git diff --check
```

Focused checks cover equal-coefficient MT equivalence and summed-gradient
equivalence, independent storage and unchanged RNG consumption, coefficient
constraints/reset, step/sequence consistency and causality, saved-config
checkpoint reconstruction, parameter counts, scalar-cost profiles on CPU and
CUDA when available, and a synthetic SSC cost-evaluator call without fitting.
Tests also cover baseline inheritance, seed/dataset rejection, read-only dry
runs, guarded resume/config mismatch, report-only dispatch, and relocation to
another checkout/Hydra working directory. The runner uses the inherited
float32 state initialization; arbitrary float64 execution is not validated.
No `Trainer.fit`, optimizer steps or training smoke tests are used.
