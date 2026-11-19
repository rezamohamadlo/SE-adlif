# Supervisor study: stage-6 scope and available-code closeout

Review date: 16 November 2026. This is a read-only audit of source, specifications
and existing user-produced evidence. No tests, inference, gradients, feature
extraction, regressions or resampling experiments were executed by the agent.

**Decision:** stage 6 is complete **within the registered exploratory
evaluation/analysis scope**. Stage 7 is ready for preparation; no stage-7
implementation or scientific execution has started. The whole supervisor study,
exact upstream analytical-figure reproduction and independent confirmation are
not complete.

The [machine-readable evidence check](../../results/experiments/adaptation_mechanism_study_v1/STAGE6_EVIDENCE_CHECK.json)
records fresh hash checks for the completed SHD* evaluations/statistics/direct
window diagnostic, the full SSC suite, the ECG extension and the new SSC window
diagnostic. Nested and reused artifacts appear in multiple completion ledgers;
their hash registrations are not independent runs or unique observations.

## Executed coverage and exit checks

| Check | Evidence reviewed | Outcome |
|---|---|---|
| Four model pairs | AMT/MT, LT-DTH/DTH, SA-MT/MT, MT-TC/MT | All four evaluated/analyzed on each of SHD*, SSC and ECG; 12 packages within their recorded protocols |
| Saved checkpoint and protocol identities | Per-study manifests, saved configs and preflight/post-run records | Selected checkpoints, seeds 42/123/456, baseline sections and intentional objective/neuron deviations preserved |
| SHD* evaluation and associations | [SHD source registry](../../results/experiments/adaptation_mechanism_study_v1/SHD_SOURCES.json) | Nine completion ledgers, including four evaluations, four statistical packages and MT/MT-TC windows, currently match their recorded hashes |
| SSC evaluation and associations | [SSC review](../../results/experiments/adaptation_ssc_suite_v2/POST_RUN_REVIEW.md) | 230 registered outputs remain intact; four pair evaluations and speaker-clustered exploratory analyses |
| ECG features and mapped labels | [ECG extension review](../../results/experiments/adaptation_ecg_extension_v1/POST_RUN_REVIEW.md) | 163 registered outputs intact; 180 feature rows, 108 effective chunks and 2160 paired rows |
| ECG descriptive design | Saved `ASSOCIATIONS.json` records | All four designs have 30 complete records, no feature exclusions, rank 9 and 21 residual degrees of freedom; no population inference |
| Direct SSC temporal evidence | [SSC window review](../../results/experiments/adaptation_ssc_windows_v1/POST_RUN_REVIEW.md) | 256 registered outputs, 12 passing user tests, six anchors, 234 batches, 179658 descriptors and 89829 paired rows verified |
| Ordinary SSC prediction equivalence | Six saved window anchors and per-record references | All 59886 new ordinary predictions match their reference classes; unchanged validation accuracy/loss tolerances pass |
| Mask separation | Source loader/model item, supervised aggregation and window code | Ordinary loss preserves its recorded minimum-padding mask; diagnostic windows exclude burn-in and synthetic minimum/batch padding |
| Negative or mixed outcomes | Saved per-seed summaries | Retained as findings; neither a positive result nor independent-test improvement is a completion condition |

This audit closes the **registered extension work**, not every possible future
experiment mentioned in the planning document. Additional window offsets,
continuous corruption curves, extra input descriptors and independent data are
follow-ups or later-stage scope, not silently claimed as executed here.

## Available code and adaptation map

The supervisor asks for an explicit account of what analytical code is actually
available. Local inventory found no `.ipynb`, `.m` or `.jl` analytical notebooks
outside excluded environment/data/result directories. The Python checkout has
the classification model, dataset wrappers and newly added versioned analytical
workflows. No separate upstream figure/saliency package was identified from
that inventory. This is a **local availability finding**, not proof that upstream
analysis code does not exist elsewhere, and not an external-paper parity audit.

| Available component | Reuse/adaptation decision |
|---|---|
| [MLPSNN.forward](../../models/pl_module.py) | Reuse the saved classification forward and temporal-output aggregation; keep checkpoint configs and weights unchanged |
| [SE-adLIF layers](../../models/alif.py) and registered variants under `models/ours/` | Reuse model implementations selected by saved configs; architecture/objective differences remain explicit |
| [Surrogate derivative helpers](../../models/helpers.py) | SLAYER gradient injection is available for stage-7 surrogate input sensitivity; its existence does not verify gradient results |
| `layer_forward_with_states` in the layer code | It is decorated with `torch.no_grad`; do not use this history path for input-gradient analysis |
| [Versioned input descriptors](../../scripts/adaptation_analysis/features.py) and [count interventions](../../scripts/adaptation_analysis/perturbations.py) | These are local operational definitions for the study, not established reproductions of an upstream figure implementation |
| [SSC validation loader](../../scripts/adaptation_ssc/data.py) | Reuse validation-only authenticated framing, physical-support metadata and collator equivalence; avoid dataset constructors that may download/open test data |
| [ECG signed inputs/maps](../../scripts/adaptation_ecg_extension/inputs.py) | Adapt explicitly for signed events and temporal labels; one shared map, class-0 argmax semantics, recorded event-count changes and native-step units |
| [Temporal consistency objective](../../functional/temporal_consistency.py) | Its overlapping same-label training windows and auxiliary penalty remain separate from ordinary evaluation loss and nonoverlapping diagnostic windows |
| [Direct window descriptors](../../scripts/adaptation_tc_windows/metrics.py) and [SSC extension](../../scripts/adaptation_ssc_windows/metrics.py) | Reuse the frozen probability descriptors; keep mean-membrane window CE/accuracy distinct from mean-probability accuracy |

No main runner, model implementation, training objective, frozen baseline,
source manifest or checkpoint was changed by this closeout. Exact upstream
analytical-figure equivalence remains **unverified** and must not appear in a
reproduction claim.

## Scientific limits carried forward

- All evidence is exploratory and conditional on validation-selected trained
  checkpoints. Official tests have historical development exposure; neither
  fresh seeds nor a new stage name restores independent-test status.
- SHD*/SSC cluster-based exploratory uncertainty and ECG descriptive analyses
  are different protocols. ECG's 30 rows have no verified independent patient
  identities; no patient intervals, bootstrap or significance claims are made.
- ECG feature time is in model steps; SHD*/SSC use documented physical bins.
  Their numeric feature values cannot be pooled as physically matched measures.
- ECG nearest-bin time resampling changes event counts and class proportions
  jointly. It does not isolate duration or prove a causal mechanism.
- Shared parent predictions, repeated corruptions and multiple windows are
  shared observations, not extra independent training seeds.
- At every SSC window width, seeds 42/123 favor MT-TC smoothing and window CE,
  whereas seed 456 reverses those directions. A lower across-seed mean JS is
  not consistent superiority or necessarily useful class evidence.
- CV-ISI, additional transformations and comprehensive final visual synthesis
  are not claimed as executed; keep them explicitly scoped if added later.

## Readiness for stage 7

The roadmap's stage-7 prerequisite is a validated feature/loss/intervention
path. The registered packages and ordinary-equivalence gates now satisfy that
bounded prerequisite, including direct SSC windows. There is no need to rerun
completed stage-6 experiments to begin **preparing** stage 7.

The next implementation should use `MLPSNN.forward` under enabled autograd,
frozen model weights and an input requiring gradients. It must record that the
sensitivity uses the model's surrogate derivative, not a physical or causal
gradient. Any later input-only diagnostic optimization needs explicit domain,
length and event-budget constraints. Prepare a small validation pilot first;
all scientific execution and contract tests remain user-launched.

The experiment-written `COMPLETE.json` records intentionally retain
`stage6_complete: false`: execution alone did not close the study stage. This
later, separately recorded audit supplies the scoped completion decision
without rewriting those frozen execution/provenance records.
