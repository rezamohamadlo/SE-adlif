# Finding one model that improves at least two datasets

Review date: 10 November 2026. Scope: the reference paper, current baseline
authority, classification model implementations/configs, accepted results,
retained unsuccessful candidates, expanded seed cohorts, speaker studies,
tuning/continuation history and unfinished experiments. Audio-compression and
waveform-generation workflows answer different questions and are outside this
classification accuracy assessment.

The central finding is that **no examined candidate currently has robust,
protocol-matched ordinary-inference evidence of beating the local reference on
two datasets**. DTH and MT genuinely improve SHD* in the original matched
cohort. SSC is the most useful second target. ECG needs a matched reference
before a strong architecture comparison, and the initial SA-MT ECG advantage
does not survive the expanded seed evidence.

This does not establish that DTH or MT is intrinsically incapable of transfer.
It establishes that the examined architectures and training settings have not
met the two-dataset objective. No training, checkpoint inference, baseline
replacement, source-model changes, commit or push was performed for this review.

## Verified ordinary-inference evidence

The review script cross-checked **78 original-seed cost/CSV pairs**, plus one
VS-MT saved cost reproduction without its local raw training artifacts. It also
read **28 additional-seed test records**. Cost reproductions and saved tests
agree within floating-point tolerance (1e-6 in fractional accuracy). Each
available CSV has a unique validation epoch history and the recorded selected
epoch agrees with its best validation accuracy. Reference SSC seed 42 retains
its documented standalone-test provenance rather than an injected CSV test row.

The following original cohort uses seeds 42, 123 and 456 (n=3). Values are
test accuracy (%) as mean +/- sample SD. Parentheses give the paired mean
change from the **local reference**, in percentage points, before rounding.
There is no numeric published-paper comparison in this table. ECG differences
are descriptive because reference stopping/budgets differ.

| Model | SHD* test (delta reference) | SSC test (delta reference) | ECG test (delta reference; unmatched) |
|---|---:|---:|---:|
| Reference SE-adLIF | 93.15 +/- 0.42 | 80.47 +/- 0.20 | 88.32 +/- 0.35 |
| DTH | 94.49 +/- 0.52 (+1.34 pp) | 79.95 +/- 0.46 (-0.51 pp) | 87.63 +/- 0.31 (-0.69 pp) |
| MT | 94.61 +/- 0.23 (+1.46 pp) | 80.07 +/- 0.14 (-0.40 pp) | 87.81 +/- 0.49 (-0.51 pp) |
| AMT | 92.71 +/- 1.94 (-0.44 pp) | 80.26 +/- 0.18 (-0.21 pp) | 87.83 +/- 0.18 (-0.49 pp) |
| MT-TC | 93.80 +/- 0.95 (+0.65 pp) | 80.14 +/- 0.11 (-0.32 pp) | 88.13 +/- 0.44 (-0.19 pp) |
| LT-DTH | 93.98 +/- 0.63 (+0.82 pp) | 80.17 +/- 0.19 (-0.30 pp) | 87.67 +/- 0.61 (-0.65 pp) |
| SA-MT | 93.73 +/- 0.52 (+0.57 pp) | 80.10 +/- 0.25 (-0.37 pp) | 88.39 +/- 0.22 (+0.07 pp) |

SA-MT superficially clears two original means. Its ECG margin is only 0.070 pp,
the comparison has unmatched stopping, and expanded seeds reverse that outcome.
It should not be presented as the requested two-dataset success.

| Model / dataset | Original 3 seeds | All 10 seeds |
|---|---:|---:|
| DTH / SHD* | 94.49 +/- 0.52 | 94.10 +/- 0.56 |
| MT / SHD* | 94.61 +/- 0.23 | 93.89 +/- 1.13 |
| MT / ECG | 87.81 +/- 0.49 | 87.69 +/- 0.45 |
| SA-MT / ECG | 88.39 +/- 0.22 | 87.94 +/- 0.71 |

The ten-seed cohort contains the original three plus
1009/2027/3037/4049/5051/6067/7079. These are overlapping descriptive cohorts,
not independent samples. The reference has only three seeds, so the ten-seed
means do not establish ten matched reference/model comparisons. DTH is a more
stable SHD starting point than the original three-seed MT ranking suggests.

## What has already been tried

| Variant | Examined mechanism | Outcome relevant to the next decision |
|---|---|---|
| DTH | Layer-activity-controlled dual-timescale thresholds | SHD strength; SSC/ECG below reference |
| MT | Learned mixture of fast and slow adaptation currents | SHD strength; SSC/ECG below reference |
| LG-DTH | Additional per-neuron local activity corrections | SHD mean 93.74%; below DTH/MT |
| MT-DTH | Combined MT adaptation and DTH threshold | SHD mean 93.74%; worse than both parents in every original matching seed |
| DA | Gated mixture of fixed 1/2/4-frame delays of projected current | SHD mean 93.33%; legacy SSC evidence is not current mean-loss confirmation |
| FP | Frequency-derived coupling parameterization | SHD mean 93.35%; historical SSC result has verification problems |
| MR, K=2 | Less frequent adaptation updates | SHD mean 93.89%; no current completed multi-dataset accuracy success |
| AMT | Activity-dependent fast/slow mixture | Best completed proposed SSC mean, still below reference |
| LT-DTH | Learned activity-trace timescales | SHD mean below DTH; other datasets below reference |
| MT-TC | Training-only temporal JS consistency | Some recovery over MT on SSC/ECG, still below reference |
| SA-MT | Independent slow spike coefficient | Initial ECG gain is not robust to seed expansion |
| VS-MT | Independent slow voltage coefficient | Only one recorded SSC test, 80.43%; remaining evidence incomplete |

The `not_improved` location classifies SHD performance relative to DTH/MT; it
does not mean all results there are invalid or negative on every dataset.
Historical summed-loss SSC, unverified/mixed histories, partial runs, speaker
holdout, TTA and weights-only continuation are separate evidence categories.
They must not be pooled into a fresh matched multi-dataset claim.

The earlier favorable SA-MT continuation settings also failed to transfer to
fresh training: its retained three-seed fresh trial averages about 88.74% on
SHD*. This is a warning against treating continuation improvements as evidence
for a fresh architecture/configuration.

## Why another feedback mechanism is not the first priority

Source inspection shows that SE-adLIF already has membrane-dependent and
spike-dependent adaptation. The reference paper emphasizes oscillatory dynamics
and the update order: membrane update, spike/reset, then adaptation driven by
the current post-reset membrane (local PDF page 6, Eq. 10).

Several variants add further ways to regulate activity, adaptation strength or
adaptation timescales. Their negative outcomes make simply stacking these
mechanisms a weak next hypothesis. The direct MT-DTH negative result is
especially relevant. This is an interpretation of the tested evidence, not
proof that such mechanisms can never work.

The learned mechanisms are active. Read-only selected-checkpoint inspection
found MT fast-mixture averages near .44-.48 on speech but only .14-.26 on ECG;
ECG strongly favors the slow branch. DTH learns substantially different
homeostatic/transient gains across datasets. These coefficients suggest
dataset-dependent dynamics; they do not identify the cause of accuracy changes.
Learning DTH's target rate alone is not a compelling distinct mechanism:
its constant offset can be absorbed by the existing free threshold logits.

SSC is not exhibiting the SHD near-100% training pattern. At selected epochs,
reference/MT training accuracy is approximately 87.18%/87.49%, with validation
82.01%/82.15%. MT's slightly better validation did not translate to better test
accuracy. Therefore a tiny validation peak improvement is only screening
evidence, and a universal 'increase dropout' or 'train longer' answer is not
supported. These training values also use training mode, unlike validation.

## Resolve unfinished evidence first

The existing [SSC study](../experiments/ssc-validation-search-plan-v1.md) has
**18 completed validation-only LR-screening runs**, without a completed final
confirmation cohort. Its frozen selection advances:

| Candidate | Mean best validation, seeds 42/123 | Improvement over own fresh LR .006 control |
|---|---:|---:|
| SA-MT, LR .009, dropout .15 | 82.02585% | +0.19036 pp |
| AMT, LR .003, dropout .15 | 82.01583% | +0.04508 pp |

The finalists differ by only 0.01002 pp. The dropout/check/confirmation stages
must not be described as completed or as producing test gains.

VS-MT SSC seed 123 has only config/summary files locally; its summary claims
30/40 epochs but its metrics and checkpoint are absent. A dropout-study
candidate similarly has a 15/40-epoch summary without raw local artifacts.
Recover the original metrics and own `last.ckpt` before discussing resume;
do not create a fresh run under the interrupted run's identity. No such recovery
or training was launched in this review.

## Ranked next experiments

### 1. Smallest useful new study: locate the helpful layer

All examined two-hidden-layer saved configs use the same cell in both hidden
layers. No SE/MT, MT/SE, SE/DTH or DTH/SE placement ablation was found.
Before designing another cell, test whether applying the existing change in
both layers causes the cross-dataset regression.

The first minimal screen should compare:

- **SE-adLIF -> MT -> readout**;
- **MT -> SE-adLIF -> readout**;
- the existing matched **SE -> SE** and **MT -> MT** controls.

Here arrows describe the two hidden layers, not separate dataset choices.
Use the same chosen architecture on SHD* and SSC. Screen SSC with validation
only, then require acceptable SHD* validation behavior before freezing a final
candidate. Use matching seeds and original inherited budgets first. Do not
claim that whichever layer wins SSC will necessarily transfer to SHD*.

This is an architecture-placement ablation with direct local motivation. It
could yield a useful single hybrid model, but placement alone is not an
established novel neuron contribution. A small first pilot is appropriate;
do not immediately launch every layer/model combination across all datasets.

### 2. Cheap optimization control: preserve reference dynamics at initialization

Current MT starts its learned convex mixture at 50% fast / 50% slow. Compare
that with a fast-dominant or exactly reference-preserving initialization.
For example,

```
w_used = w_fast + g * (w_slow - w_fast),  0 <= g <= 1
```

At g=0, adaptation reduces to the reference fast branch. A numerical limit test
must copy shared parameters and verify the original SE update order and outputs;
fresh training still starts from random weights, not a pretrained reference.
The gate must have a usable gradient; placing a sigmoid at hard saturation can
silence learning of the added branch.

This expression is algebraically MT's existing convex mixture with
g=1-mix. **It is an initialization/optimization ablation, not a new expressive
model family or a novelty claim.** Reference-preserving initialization also
does not guarantee that optimization will preserve reference accuracy.
Zero-initialized residual gates have established prior art, for example
[ReZero](https://proceedings.mlr.press/v161/bachlechner21a.html); that work does
not demonstrate that this particular spiking-cell change will improve accuracy.

### 3. If a substantially different mechanism is needed: feature-specific timing

The paper identifies synaptic delays and dendritic processing as complementary
to somatic adaptation (local PDF pages 10-11). There is direct primary research
on learnable delays evaluated on both SHD and SSC:
[DCLS delay learning, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4df1cc5a7528b7197ad8ae76ff30107a-Abstract-Conference.html).

This proposal must be distinguished from the existing negative DA experiment.
DA delays a neuron's **already summed/projected current** using fixed short
delays; it cannot independently align different incoming features. A genuinely
different capability is input-specific or synapse-specific learnable timing:

```
I_i(t) = sum_j W_ij * x_j(t - d_j)       # input-specific
I_i(t) = sum_j W_ij * x_j(t - d_ij)      # synapse-specific
```

Consider a parameter-efficient structured version first, with causal padding,
an explicit maximum timing horizon and a reference/no-delay limit. A wider
learnable horizon differs from DA's fixed 1/2/4 frames. Interpret the horizon
using each dataset's actual frame duration rather than the neuron's `dt`.
Use controls for fixed random delays and comparable added parameter/compute
cost. Per-synapse timing can be expensive, and merely combining known adaptation
and delay learning is not by itself a novelty claim.

The cited delay work has different architecture, preprocessing, normalization
and optimization. Its success motivates a hypothesis; its reported accuracy
must not be treated as a guaranteed outcome in this project.

## The two-dataset target and confirmation contract

Prioritize **SHD* + SSC**: both use event-based cochlear inputs and have current
matched reference protocols. ECG is a different per-timestep streaming-labeling
task. The reference paper describes level-crossing-encoded ECG inputs; the local
loader's conversion to floating point does not establish that they are analog.
The actual MAT values and timestamp convention must be checked before selecting
spike statistics or perturbations. The ECG MAT files were not present in the
local data tree when this representation issue was reviewed.
Its accepted reference completed 383/283/185 epochs, with early stopping for
seeds 123/456, whereas proposed runs completed 400. Preserve that historical
reference, and use a separate matched reference/model cohort if an ECG claim is
later pursued.

A single model means one frozen architecture/cell definition across datasets.
Dataset-specific widths, LR, surrogate scale and readout loss are legitimate
when predefined and inherited consistently; the paper itself tunes such values
(local PDF pages 28-29 and Table 7 on page 35). It does not mean one checkpoint
or one numeric hyperparameter vector must solve all datasets.

For a defensible confirmation:

1. Freeze the candidate, selection rule, permitted dataset-specific settings
   and a practically meaningful target before final testing. Do not select
   architectures by counting tiny positive test means across many candidates.
2. Give the reference a comparable optimization-tuning opportunity when tuning
   the candidate, or explicitly label the comparison as a tuning study.
3. Use validation-only development, including the existing SSC study. Keep
   SHD split seed 42 and its 20% validation reserve; do not substitute the
   speaker-holdout study as another successful dataset.
4. Confirm the frozen winner and reference on five fresh matching training
   seeds per target dataset: **20 fresh runs for two models x two datasets x
   five seeds**. This is a budget example, not a launched queue. Preserve all
   seeds, including negative outcomes, and distinguish means, paired differences,
   sample SD and uncertainty. More seeds are warranted if the estimated effect
   remains small or unstable; five does not ensure significance.
5. Evaluate validation-selected checkpoints with ordinary inference after
   freezing selection. TTA is a separate experiment requiring reference TTA
   and an equal inference budget. Previously inspected official tests remain
   reused development data; fresh initialization seeds do not erase that history.
6. Use `runner.py`, opt-in configs, new result roots, resolved baseline/deviation
   metadata and normal cost measurement. Do not alter completed runs or the
   baseline to manufacture a win. All training is launched by the user.

The immediate recommendation is **a targeted SSC-first layer-placement
ablation, supported by the existing validation study**, while retaining DTH as
the strongest current SHD control. The larger research direction is genuinely
feature-specific temporal processing if the small ablations do not resolve the
SSC deficit. Another ad-hoc DTH/MT combination is a lower-priority use of compute.

## Evidence files

- [Recomputed per-run scores, means, paired differences and study state](cross-dataset-review-2026-11-10/EVIDENCE.json).
- [Accuracy and seed-expansion figure](cross-dataset-review-2026-11-10/cross-dataset-evidence.png)
  and [SVG](cross-dataset-review-2026-11-10/cross-dataset-evidence.svg).
- [Read-only aggregation/plot script](cross-dataset-review-2026-11-10.py).
- [Reference paper](../../reference_paper.pdf), [baseline authority](../../config/paper_baseline.yaml),
  [current broad comparison](../../reports/comparison-summary.md),
  [reference/proposed comparison](../../reports/reference-vs-models.md).
- [SSC LR selection](../../results/experiments/ssc_validation_search_v1/selection/select-lr.json)
  and [frozen study definition](../../config/studies/ssc_validation_search_v1.yaml).
- [DA implementation](../../models/ours/da_adlif.py), [MT](../../models/ours/mt_adlif.py),
  [DTH](../../models/ours/dth_adlif.py), [SA-MT](../../models/ours/sa_mt_adlif.py),
  [VS-MT](../../models/ours/vs_mt_adlif.py).

Some main reports contain older snapshots, historical text or stale partial
status. This review gives priority to seed-level JSON/config/CSV evidence;
it does not rewrite those historical records. The derived script can be rerun
with the local project Python:

```powershell
.\.ven\Scripts\python.exe docs/audits/cross-dataset-review-2026-11-10.py
```
