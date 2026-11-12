# Complementary mechanisms study v1

**Discovery plan superseded by [v2 speaker-disjoint development](complementary-mechanisms-study-v2.md)
on 2026-11-11. Preserve the v1 canonical results; do not use their checkpoints
to assess unseen training speakers or launch the superseded150epoch queue.**

User-approved study, 2026-11-11. Version: `complementary_mechanisms_v1`.
Outputs: `results/experiments/complementary_mechanisms_v1/`.

## Objective and authorization

Discover two complementary ideas through Baseline -> Baseline+A -> Baseline+A+B.
The final matched ablation includes Baseline, A, B, and A+B on SHD*, SSC and ECG.
Improvement is a hypothesis; negative findings remain valid outcomes. There is
no mandatory architecture family and no fixed wall-clock deadline.

### User clarification: unseen-data generalization, 2026-11-11

The primary final success criterion is **ordinary-inference test accuracy on
unseen data**, compared with the matched SE-adLIF control. Validation accuracy
and loss are development/selection evidence only. A validation gain never
establishes better test generalization or paper superiority. All four arms must
eventually report matched train/validation/test metrics, generalization gaps,
per-seed differences and sample SD; n=1 has no across-seed SD.

The original instruction reserves official tests until A/B/configs/evaluation
are frozen. The user subsequently emphasized test rather than validation
accuracy. Clarification has been requested on whether the current baseline/A001
should receive an explicitly exploratory test evaluation now. Until clarified,
no early test-driven idea selection or unsupported unseen-data claim is made.
If an early evaluation is requested, preserve checkpoint selection based on
validation, label test exposure exploratory/reused, and never present later
reuse of that same test set as independent confirmation.

Use repository-local `data/` as explicitly confirmed by the user. Reinspect
actual complete files before new acquisition; incomplete range-download parts
are not a usable SSC dataset. All source/checksum and data-access records belong
under the study results root.

For this study the user explicitly authorizes launching, resuming and safely
stopping our own training, overriding the repository's user-launch-only rule.
Preserve unrelated jobs and historical evidence. Do not commit or push.
Use existing `runner.py`/`run.py`; no new runner or orchestration script.
Only isolated opt-in configs/components, reusable callbacks and logging hooks
may extend the existing workflow. Execute experiments, not merely commands.

## Native Goal and budget

The existing native Goal is active and references the supplied study request.
Do not recreate it or reset its usage. The actual native remaining-token budget
is null: it has no enforced budget. Available Goal tools cannot change the
objective/budget of an active Goal. The requested native allowance is 50,000
tokens, but it could not be applied; do not claim it is enforced or invent a
remaining allowance. Initial reported thread usage was 151,776. Report actual
native accounting and preserve it. If a native bound becomes available, stop
launching near that bound with recovery/reporting reserve. Do not change
unrelated global settings.

Short reference for a future new Goal, only if no Goal is active:
`Execute docs/experiments/complementary-mechanisms-study-v1.md with a 50000-token budget.`

## Initial inspection and evidence

Read relevant audits, supervisor guidance and unsuccessful studies once, then
inspect only files needed for the next decision. Verify actual data encoding,
Python/dependencies, GPU/memory/processes, compatible control curves, checkpoint
identities, output paths and locks before spending compute.

Initial local inspection found `.ven` Python 3.11, torch 2.6.0+cu124,
Lightning 2.2.1, an idle RTX 4090 with 24 GiB total/about 22.5 GiB free, and
raw SHD events. SSC and ECG raw data were absent initially. Never substitute
speaker-holdout SHD for SHD*. Preserve historical test exposure as a limitation.

The November 10 cross-dataset audit records failed or inconsistent DTH/MT
extensions and a negative MT-DTH combination. Existing SHD baseline controls
may be reused only after matching seed, split, batch, precision, preprocessing,
optimizer, schedule, budget and source dynamics. Historical ECG stopping and
budget differences require new matched controls when incompatible.

## First idea and prior-art distinction

**A001:** zero-initialized causal per-input-channel residual FIR, before
synaptic projection, initially using past lags 1, 2, 4 and 8 frames. Hypothesis:
feature-specific short-term temporal alignment complements SE-adLIF adaptation.
At initialization it must reproduce the reference exactly with usable filter
gradients. Inputs retain their original length/resolution and classification
targets/masks. Neuron dynamics and the reference implementation stay selectable.

The failed DA variant mixed delays of already projected/summed current using
fixed 1/2/4-frame delays. A001 learns signed channel-specific temporal filtering
before projection, with a wider horizon. It is not a repeat of that hypothesis.
Temporal filtering and learned delays have established prior art:
[ICLR 2024 DCLS delay learning](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4df1cc5a7528b7197ad8ae76ff30107a-Abstract-Conference.html).
Do not claim invention of delay learning or transfer that paper's scores to this
architecture. B remains unselected until an observed limitation of A motivates it.

## Throughput and numerical controls

Before expensive SSC/ECG runs, calibrate with disposable model/optimizer states
and real training batches, including representative long sequences. SHD may be
calibrated too. Candidates: SHD/SSC 256,512,1024; ECG 64,128,256, higher only with
evidence. Include warm-up, CUDA synchronization, step time, samples/second,
peak allocated/reserved memory, optimizer updates/epoch and estimated epoch time.
Keep about 15% memory headroom and stop increasing batch when unhelpful.

Freeze the fastest stable batch for every arm of a matched dataset protocol.
Do not automatically scale LR. Record deviations and verify the matched control
learns adequately. A changed batch requires a new matched comparison, not an
architecture-only claim against an old smaller-batch run. Calibration cannot
establish convergence; its weights never initialize main training.

Benchmark validation batch separately and verify masks, aggregation, predictions
and numerical consistency. Change data loading only on measured bottlenecks.
Cache deterministic processing; keep any training augmentation fresh. Retain
established precision initially. AMP/compile changes require independent
surrogate-gradient/state correctness and performance tests for all arms.
Never truncate sequences/BPTT, change resolution, or alter the task for speed.

## Stages and bounded decisions

Training seed 42 is for discovery; split seed 42 remains fixed where applicable.
All main runs start with fresh random weights and optimizer state. Staged
extensions resume their own last checkpoint; they are not new independent runs.
Keep full-budget schedule metadata intact during stopping boundaries.

| Stage | Required experiment | Checks / limits |
|---|---|---|
| 0 | Recover compatible baseline configs and validation curves | Reuse only exact matched controls; otherwise train one |
| 1 | Discover A on SHD*: Baseline versus A | 30 -> 75 -> 150 when promising/inconclusive; full budget 300 |
| 2 | A transfer: matched Baseline versus A on SSC | 10 -> 40; investigate failed transfer before B |
| 3 | A transfer on ECG | 50 -> 100, further only with reason; full budget 400 |
| 4 | Identify a remaining A limitation; compare A versus A+B on SHD* | Reuse compatible A; reject bad combinations and preserve A |
| 5 | Combined transfer: A+B versus A, SSC then ECG | Matched staged budgets, initially seed 42 |
| 6 | Complete Baseline+B and four-arm ablation | Identical dataset-specific conditions |
| 7 | Freeze architecture/settings; add seeds 123,456 | Only after promising transfer; add seeds if effects unstable |

For development screens only, use validation accuracy for checkpoint/idea
selection while jointly interpreting loss, stability and curves. The primary
final success endpoint is matched ordinary-inference test accuracy on unseen
data. SHD's near-ceiling validation accuracy makes loss useful preliminary
evidence, never proof of test improvement. Do not reject solely on weak early epochs or
repeatedly extend a flat run without a recorded reason.

Before each experiment record hypothesis/mechanism, prior-art distinction,
matched control, primary metric, advancement rule, practical effect threshold,
epoch ceiling and estimated compute cost. Initial practical thresholds:
SHD >=0.20 pp; SSC >=0.30 pp; ECG >=0.20 pp validation improvement, interpreted
with loss/stability. SHD accuracy within 0.20 pp with >=10% lower loss or a still
improving curve is inconclusive and may extend. At 30 epochs extend a technically
sound run unless sustained plateau/substantial deficit warrants rejection.
Thresholds are screening rules, not significance guarantees.

Before final test access, freeze the practical test-effect thresholds and the
four-arm/seed/dataset evaluation plan. Report A-Baseline and A+B-A on test data,
not just validation. An unsuccessful test result remains unsuccessful even if
validation improves. Do not select the next architecture or hyperparameter from
official test scores. Any early test evaluation explicitly requested by the
user is exploratory/reused evidence and is labelled as such.

Decisions distinguish technical failure (fix), inconclusive (justified extension),
unpromising (retain evidence, different idea), and promising (allocate transfer
compute). Do not force A+B into the paper narrative. Aim at all three datasets;
report regressions and never omit/replace an unsuccessful dataset.

## Splits, configs and official tests

Inherit `config/paper_baseline.yaml`, using its dataset-specific section, and
record every deviation in resolved configs/manifests. SHD*: official training
data with random 20% validation and split seed 42. SSC: official validation.
ECG: established split with verified actual record separation, input encoding,
timestep labels and masks. Do not assume legacy reference comparability.

Keep official tests outside the entire idea-selection loop, including opt-in
train/validation-only data access. Freeze A, B, configs and final evaluation
plan before four-arm ordinary-inference testing; no TTA or test-driven choices.
Fresh seeds do not erase earlier test inspection. If considering existing
augmentation, first resolve the padding/time-warp confound: preserve physical
support separately and never infer duration solely from the last nonzero frame.

## Logging, monitoring and artifacts

Print flushed stage-start purpose/idea/hypothesis, dataset/arm/training and split
seeds, batch, precision, LR, ceiling, output and fresh/resume mode. Every epoch
prints completed count versus ceiling (separate from zero-based checkpoint
index), available train/val loss/accuracy, best value/epoch, LR, batch, updates,
measured seconds, peak VRAM and measured-runtime ETA labelled as an estimate.
Do not manufacture missing metrics. Print actual errors and diagnostic action.

At decisions print `[DECISION] ADVANCE / EXTEND / REVISE / REJECT / BLOCKED`,
evidence and next action. Keep compact `STATUS.json`, experiment registry,
per-epoch metrics, fully resolved configs, best/last checkpoints and cumulative
`RESULTS.md` under the study root, using portable repository-relative paths.
Keep stage snapshots so extensions cannot erase earlier checkpoint evidence.

One training job per GPU. Check real handles/processes before launching or
claiming activity. Monitor compact summaries roughly every ten minutes, at
completion/error, using efficient waits. Do not alter active-run hyperparameters.
Enforce stage ceilings and checkpoint safety. Near the operational token limit,
stop launching, preserve state, report any live bounded job and exact recovery
commands; do not reset the native Goal. No fixed wall-clock deadline.

## Completion audit

Report all attempted ideas, failures included; per-seed and mean/sample SD;
A-Baseline, B-Baseline, A+B-A, A+B-B; interaction
`(A+B-A)-(B-Baseline)`; uncertainty/practical effect sizes; appropriate paper
deltas; parameters, measured inference/training costs and protocol deviations.
Combined superiority does not establish synergy, three seeds do not establish
significance, and single-seed validation cannot establish paper superiority.
Short screens are preliminary, not completed full-budget comparisons.

Stop with verified findings when complete, a justified next step lacks resources,
or a genuine blocker persists. Honest negative findings are acceptable; never
fabricate improvement or promise publication. Complete the native Goal only
after requirement-by-requirement evidence audit. Preserve accounting throughout.
