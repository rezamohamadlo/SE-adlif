# Registered stage-7 diagnostic suite

**Registered scope completed on 16 November 2026 and independently reviewed
17 November 2026.** All 54 checkpoint jobs, 432 repeat checks, 216 constrained
optimization attempts and nine shared-input transfer bundles completed. There
were no repeat/autograd failures. Of the 216 attempts, 171 produced accepted
nonzero changes; all 45 ineffective attempts remain in the evidence.

The [independent review](../../results/experiments/adaptation_stage7_suite_v1/POST_RUN_REVIEW.md),
[closure checklist](../../results/experiments/adaptation_stage7_suite_v1/STAGE7_CLOSEOUT_AUDIT.json),
[current presentation](../../results/experiments/adaptation_stage7_suite_v1/presentation_v1/SUMMARY.md)
and [both input origins](../../results/experiments/adaptation_stage7_suite_v1/origin_controls_v1/SUMMARY.md)
are the current result references. The main package has 15,439 registered
outputs. Independent audits verified 2,952 saved gradients, all 5,184 transfer
evaluations and the seed-matched arithmetic. Exact source/cohort archive ledgers
and all checkpoint/history hashes also passed post-run checks.

This closes the **registered exploratory diagnostic scope**, with the limited
cohorts below; it does not claim exhaustive SHD*/SSC validation gradients,
independent confirmation or natural-accuracy superiority. Stage 8 synthesis and
cost/activity reporting remain separate. Scope was fixed before suite execution
on 16 November 2026, after the prior gradient/numerical pilots.

The user explicitly authorized agent commands and experiments to finish stage 7
on 16 November 2026, overriding the prior launch-only restriction for this task.
This authorization is recorded in per-run provenance. No model-weight training,
fine-tuning, optimizer update or official-test evaluation is performed.

## Scope fixed before execution

The [specification](../../config/studies/adaptation_stage7_suite_v1.yaml) inherits
`config/paper_baseline.yaml`. It authenticates the completed stage-6 sources and
all **54 unique saved checkpoints**: six model types, three training seeds
(42/123/456), and SHD*/SSC/ECG. The four matched pairs are AMT/MT, LT-DTH/DTH,
SA-MT/MT and MT-TC/MT; shared MT parents are not new independent observations.

| Coverage | SHD* | SSC | ECG |
|---|---:|---:|---:|
| Strict-precision reference records per checkpoint | 1,631 | 9,981 | 30 |
| Fixed gradient cohort | 40 (2/class) | 70 (2/class) | All 30 validation records |
| Repeat-check records per checkpoint | 8 | 8 | 8 |
| Diagnostic-input records per checkpoint | 4 | 4 | 4 |

Cohorts use labels, fixed validation membership/order and a fixed selection seed,
not model outcomes. The four diagnostic records are the first four selected
records; all are shown, including ineffective attempts. Full reference coverage
does not make the SHD*/SSC gradient cohorts full-validation. This is a registered
exploratory diagnostic scope, not an independent confirmation study.

The queue produces **2,520 primary gradient records**, **432 repeat checks**,
**216 constrained input-optimization attempts** and **nine transfer bundles**
containing **5,184 shared-input target evaluations**. The final pair tables have
288 record/seed/pair/origin-role rows. Random replicates average within record;
means and sample SDs aggregate the three matching training seeds.

## Common numerical and loss contract

All new diagnostics use float32 `highest` matmul precision, CUDA matmul TF32 off,
cuDNN TF32 retained and cuDNN benchmarking off. This is a separate numerical
protocol; original source-high scores and checkpoint selection remain unchanged.
Software and hardware are recorded separately from scheduling assignments.

`MLPSNN.forward` is used directly. SHD* retains sum-of-frame-softmax aggregation,
SSC retains mean-membrane aggregation, and ECG retains target-aligned summed
membrane aggregation with mean CE over valid targets per record. Dummy targets
remain ignored; ECG's saved all-zero-label argmax-to-class-0 treatment is
preserved. MT-TC auxiliary JS is excluded from the shared supervised objective.
Targets remain int64. The common-engine reevaluation of existing SHD* MT/AMT
seed-42 records is intentional engine validation, not additional training seeds.

Input gradients include burn-in inputs and report synthetic padding separately.
Profiles normalize within each nonzero physical record before averaging; raw
magnitudes remain available. Zero maps keep undefined normalized values. Repeat
criteria are relative physical L2 <= 0.01 and cosine >= 0.999, with explicit zero
cases. Autograd forward checks require matching predictions and CE difference
within `2e-5 + 1e-5 × abs(reference CE)`. These are engineering criteria, not
statistical significance or proof that surrogate derivatives are true hard-spike
finite differences.

## Constrained diagnostic-input optimization

For each fixed diagnostic record and model/seed, projected surrogate-gradient
ascent attempts to increase its supervised CE, for at most six steps. Only input
leaves are differentiated. Each proposal is projected from the **original**
record, preventing budget accumulation across iterations. Acceptance requires a
measured CE increase exceeding the same numerical bound, not a predicted linear
gain alone. Rejected proposals and their gradients remain in the trace.
This is a bounded greedy projection heuristic, not a global optimizer. A failed
attempt does not establish that other admissible diagnostic inputs cannot exist.

The native constraints are checked on every proposal, accepted input, control
and transferred input:

- Fixed physical length, channel count, target labels and block masks.
- At most floor(1% of native events), with a minimum one when nonempty, relocated
  from original occupied cells to original zero cells within the same channel.
- Nonnegative integer count inputs for SHD*/SSC; original per-channel count
  totals and maximum cell counts remain unchanged/bounded.
- ECG stays signed ternary, with positive and negative counts preserved
  separately in each channel; no polarity flips or event collisions.

Each accepted diagnostic input gets three random temporal relocations with the
same **actual** per-channel/polarity movement allocation. A zero/ineffective
attempt keeps the original input and zero-budget controls; it is not silently
dropped. Held diagnostic labels do not establish that relocated inputs retain
their original semantic class. Model-selected patterns are not direct causal
explanations of natural data.

All six models at the same seed receive every optimized input and matched
control. Pair tables distinguish parent-optimized and variant-optimized origins.
The primary parent-origin contrast is explicitly subject to model/loss selection
coupling and must not be presented as a natural-accuracy improvement.

## Commands and artifacts

The completed package can be inspected with read-only verification:

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_stage7.run --verify-only --output-root results/experiments/adaptation_stage7_suite_v1
```

The original complete local queue command is retained for reproducibility:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.adaptation_stage7.run --device cuda:0 --output-root results/experiments/adaptation_stage7_suite_v1
```

For multiple compatible checkouts, scheduling flags such as `--seeds 42`,
`--seeds 123` and `--seeds 456` assign disjoint jobs without changing the global
manifest/cohort. Outputs remain under dataset/model/seed paths. Exact frozen
data, weights and source packages must be available locally; Git does not move
ignored checkpoints/datasets. An OS-owned queue lock prevents concurrent writers
on one shared root. Do not delete lock files to bypass active locks.

```powershell
.\.venv\Scripts\python.exe -m scripts.adaptation_stage7.run --check-only
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_stage7
.\.venv\Scripts\python.exe -m scripts.adaptation_stage7.run --finalize-only
.\.venv\Scripts\python.exe -m scripts.adaptation_stage7.run --verify-only
```

Preflight checks sources, checkpoints/configs, baseline metadata, native loaders,
validation identities and portable paths without executing the study. Execution
uses 14 common objective/domain/cache contracts, including finite frozen-weight
input gradients on CPU and CUDA. The final two commands require corresponding
completed job data; finalization cannot claim closure from missing jobs.

Outputs live in `results/experiments/adaptation_stage7_suite_v1/`:

- `jobs/<dataset>/<model>/seed_<seed>/`: strict full references, primary/repeat
  gradients, optimizer traces (accepted and rejected), matched inputs, frozen
  state, metrics and completion ledger.
- `transfers/<dataset>/seed_<seed>/`: clean reference gates and all shared-input
  cross-model evaluations, with a separate completion ledger.
- `figures/`: paired normalized profiles, SHD*/SSC class profiles and all fixed
  diagnostic examples for all seeds, plus intervention effect comparisons.
- `RAW_GRADIENT_MAGNITUDES.csv`, `PAIRED_DIAGNOSTIC_EFFECTS.csv` and
  `SEED_AGGREGATES.json`: raw sensitivity and seed-matched CE contrasts.
- `STAGE7_CLOSEOUT_AUDIT.json`, `SUMMARY.md`, `COMPLETE.json`: the registered
  coverage/validity decision. Only verified closure marks this scope complete.
- `run_provenance/`: authorization, runtime/environment and test evidence.
- `source_snapshot/`: the exact 112 scientific files with an independently
  sealed raw/normalized checksum ledger and portable source-path index.
- `cohort_snapshot/`: all 140 selected physical/native inputs, natural stored
  inputs, true labels and block masks, with its own portable checksum ledger.
- `presentation_v1/`: current SHD* labels, explicit legends/color bars and
  shared diagnostic scales across all model/seed panels; frozen output values
  remain unchanged.
- `origin_controls_v1/`: both predeclared optimization origins shown together;
  model-conditioned selection substantially changes the contrast's direction.
- `INDEPENDENT_FINAL_VERIFICATION.json`, `INDEPENDENT_TRANSFER_VERIFICATION.json`
  and `POST_RUN_REVIEW.md`/`.json`: final independent verification and scope.

An additional read-only audit, `scripts.review_stage7_suite`, independently
checks embedded array identities, physical-gradient summaries, native input
constraints, accepted/rejected optimizer traces and exact matched control budgets.
It never constructs a neural model or reruns inference/backward. Its source and
intermediate audit evidence are archived separately in run provenance.

Frozen prior result packages, raw data, model implementations, main runners and
baseline configs remain unchanged. Stage 8 final synthesis/cost reporting,
exhaustive validation gradients and independent official-test confirmation are
separate; none is implied by this diagnostic scope.
