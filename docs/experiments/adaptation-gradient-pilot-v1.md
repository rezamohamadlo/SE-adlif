# Stage-7 SHD* frozen input-gradient pilot

**Reviewed 16 November 2026: user execution complete and saved outputs verified,
with a numerical batching follow-up required. Stage 7 remains incomplete.**
The [post-run review](../../results/experiments/adaptation_gradient_pilot_v1/POST_RUN_REVIEW.md)
authenticates 544 output hashes, 50 source hashes, 18 passed user tests, two
full-validation anchors and 80 finite nonzero physical input gradients, with
unchanged model state and exact same-mode autograd forward agreement.

MT recording **3896** changes prediction from 1 to 9 when moving from original
batching to the single-record diagnostic mode; CE increases by **0.3386995792**.
It also dominates MT's raw gradient magnitude. Keep the record and frozen
outputs; investigate batching/precision in a separate user-launched workflow
before expanding or interpreting architectural differences. The cause is not
established. Original-batch anchors passed; autograd itself causes zero recorded
same-mode forward drift. Normalized time profiles give each record equal weight.

The separate [batching/precision follow-up](adaptation-batch-precision-v1.md)
has now been executed by the user and
[reviewed](../../results/experiments/adaptation_batch_precision_v1/POST_RUN_REVIEW.md).
Strict matmul precision removes pilot layout prediction mismatches while
changing some full-validation predictions; preserve those results separately.
It generates no replacement gradients. A separately versioned strict-precision
[gradient repeat follow-up](adaptation-gradient-strict-v1.md) is implemented and
has now been executed by the user and
[reviewed](../../results/experiments/adaptation_gradient_strict_v1/POST_RUN_REVIEW.md).
All 86 required repeat/padding checks passed. This original pilot remains
preserved; matched strict-precision deletion controls are the next preparation.

Prepared 16 November 2026, after the
[scoped stage-6 closeout](../audits/supervisor-stage6-closeout-2026-11-16.md).
Scope: **MT versus AMT, seed 42, 40 validation records, two per class**.
The fixed class-round-robin selection uses labels and original validation order,
without predictions/losses/gradients. This is a tooling pilot, not all
pairs/seeds/datasets, a full-validation gradient study, or full stage 7.

## User command

The pilot is complete. For read-only verification of its saved output ledger:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_pilot.run --verify-only --output-root results/experiments/adaptation_gradient_pilot_v1
```

The original execution/resume command is retained below for provenance.
From the repository root, on the local PC:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_pilot.run --device cuda:0 --output-root results/experiments/adaptation_gradient_pilot_v1
```

The command runs **18 user contract tests**, two original-batch full-validation
anchors, 80 input backward passes, matched event-deletion controls and reports/
figures. Resume using the same command/output root; an OS-owned lock excludes
concurrent writers. No agent tests, forward/backward, perturbations or plots
were executed during preparation. This bounded workload defaults to local
execution. The completed execute attempt took **244.985 seconds**, about
**4 minutes 5 seconds**, excluding initial preflight; this is workflow time,
not pure inference latency. The agent only verified saved evidence after the run.

No weights are trained, no optimizer is created, and no official test, TTA,
fine-tuning or input optimization is performed. Loading saved checkpoints is
evaluation, not training continuation/resume. Future optimized diagnostic inputs
need a separate constrained protocol.

## Frozen identities and numerical gates

The [specification](../../config/studies/adaptation_gradient_pilot_v1.yaml)
inherits the SHD section of `config/paper_baseline.yaml` and authenticates a
clean subset of the completed AMT/MT study. Preflight checks checkpoint hashes,
epochs/configs, full validation histories, split seed 42, training-H5 identity,
state-tensor hashes and three reconstructed inputs. It opens only the training
H5 and retains its existing 20% validation split.

MT's historical `shd_costs_v1` measurement has no status/dataset fields. Its
schema is recognized explicitly using the legacy identity, saved SHD config,
model/seed/checkpoint/epoch and precision. No status is invented and no record
is rewritten. Modern AMT records must retain their actual complete status;
unknown schemas fail. Checkpoint/config/validation checks apply independently.

The saved training/model configs remain unchanged. The launcher declares a
**single-record diagnostic batching deviation** to bound backward memory:

1. Before gradients, both checkpoints must reproduce all **1,631 validation
   records** using the original batch size/order. Input identities and class
   predictions must match the frozen clean cache. Aggregate accuracy/CE must
   meet the original 1e-6 fraction/.001 CE validation tolerances.
2. Each pilot record gets a clean single-record `no_grad` reference and then
   the same forward with autograd enabled. Class prediction must match, with
   CE difference at most 1e-6. Raw parameter/buffer hashes must remain unchanged;
   every parameter stays frozen, in eval mode and without accumulated gradients.
3. Single-record versus original-batch CE/class differences are **recorded**.
   GEMM/GEMV shape changes can affect numerics. These diagnostic-mode results
   are not silently relabeled as original-batch validation accuracy. Gradient
   comparisons and deletion effects use the same single-record mode.
4. Float32 matrix precision remains `high`, model/input dtype float32, and
   cuDNN benchmark disabled. Nonfinite forward/backward results fail.

The [preflight record](../../results/experiments/adaptation_mechanism_study_v1/GRADIENT_PILOT_PREFLIGHT.json)
explicitly marks tests, forward/backward and gradient verification unexecuted.
A preflight pass is not a successful gradient or model-anchor result.

## Gradient objective, domain and masks

Differentiate **one record's saved supervised CE** through ordinary
`MLPSNN.forward`, under `torch.enable_grad`, with a cloned floating input leaf
requiring gradients and `torch.autograd.grad`. SHD* retains
`sum_softmax_over_time`; there is no new margin/mean-membrane objective or
batch-mean factor of 1/256 in this per-record derivative. Do not use the
`no_grad` state-history path or an enclosing inference-mode context.

The model supplies SLAYER surrogate derivatives. These are not true hard-spike
finite differences, derivatives of raw spike timing, causal attribution or
proof of attention. Finite-change agreement is not required for completion.

Raw signed gradients cover the full stored input. Primary physical-input
summaries include **burn-in inputs**, since they affect later states even when
early outputs are ignored. Record synthetic-padding gradient mass separately;
padding is never perturbed. Save event-cell/empty-cell mass, L1/L2 magnitude,
burn-in mass and normalized channel/20-bin relative-time profiles. Display bins
align physical support proportionally; model inputs are not time-resampled.

Zero physical mass yields undefined normalized profiles, without an epsilon or
fabricated map. Zero gradient on existing-event cells makes guided deletion
ineligible for both models on that record; keep clean/random observations.
An all-zero-gradient quality flag does not establish useful model sensitivity.

## Shared exact-budget deletion controls

Delete exactly `max(1, floor(total_events*0.01))` events per record, or zero for
an event-empty input. Retain original length and nonnegative integer counts.
Physical burn-in events are eligible; their removed count is recorded.

- Guided deletion ranks existing-event cells by **absolute MT gradient**, with
  ascending flattened-index ties, and greedily consumes the fixed budget.
  Absolute ranking does not promise increased/decreased CE.
- Three fixed random realizations sample event identities uniformly without
  replacement and map them to count cells. Budgets are exact, not Bernoulli
  thinning with different realized counts.
- **Both models receive the same MT-guided and random inputs.** AMT gradients
  predict its own local linear change but do not choose a different input for
  paired model comparison.
- Record each model's `sum(gradient * delta_counts)`, actual CE/class change
  and paired benefit. Direction disagreement is a result, not a failed gate.

Guided-versus-random scalar contrasts use only the same eligible records;
average the three random scalar effects within each record. Other clean/random
cases remain visible. Repeated realizations are not extra training seeds or
TTA. Coverage is 80 gradients and at most 400 single-record evaluations/200
paired condition rows, separate from full anchors.

## Portable caches, figures and interpretation

Six authenticated clean source files and a path index are copied atomically to
`shared/shd_clean_source/`; the scoped stage-6 decision/document is preserved
under `shared/stage6_closeout/`. Completed manifests/checkpoints remain unchanged.
Transfer exact ignored datasets, weights and arrays separately when needed;
Git pull does not supply them.

Result layout under `results/experiments/adaptation_gradient_pilot_v1/`:

- Manifest, user test log/status, progress, original-batch anchors;
- `gradients/<model>/seed_42/sample_<id>.npz` with JSON sidecars;
- `shared_inputs/sample_<id>/<condition>/rep_<rep>.npz` with sidecars;
- Per-record evaluations, gradient index/summaries and frozen-state checks;
- Paired CSV, summary JSON/Markdown and three figures: normalized relative-time
  sensitivity, declared batching drift, and matched guided/random effects;
- Hash-registered completion evidence.

Array caches use `allow_pickle=False`, embedded metadata and hashes, closed
unique temporary files and bounded Windows sharing retries. A published NPZ
with a missing JSON sidecar can recover the sidecar without recomputing a
gradient; an existing sidecar with a missing array fails and requires transfer
of the exact artifact.

All summaries are descriptive for one pair/seed/cohort. Gradient magnitude can
depend on confidence/loss; MT-guided selection is model-dependent. There are no
population confidence intervals, significance tests, independent-test claims,
automatic stage-7 completion or upstream figure-reproduction claims.

## Check and verify

Agent-safe preflight:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_pilot.run --check-only --device cuda:0
```

Optional **user-only** tests (include toy forward/backward, no training):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_gradient_pilot
```

Read-only completion verification after execution:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_gradient_pilot.run --verify-only --output-root results/experiments/adaptation_gradient_pilot_v1
```

After execution, review anchors, autograd forward equivalence, batch drift,
zero-gradient flags, frozen state and matched-control eligibility before
expanding seeds/pairs/data or preparing diagnostic-input optimization.
