# DTH with learnable activity-trace timescales

The third candidate, `LT_DTH_SE_adLIF`, learns how quickly DTH's layer-level
activity estimates forget past spikes. It is independent of AMT and MT temporal
consistency; it does not combine MT and DTH.

Original DTH fixes `rho_fast=0.9` and `rho_slow=0.99`. This variant starts
at those rates and learns two scalar logits per hidden layer:

```
rho_fast = 0.5   + (0.98   - 0.5)   * sigmoid(fast_logit)
rho_slow = 0.985 + (0.9995 - 0.985) * sigmoid(slow_logit)
activity[t] = mean(spikes[t-1])
fast[t] = rho_fast * fast[t-1] + (1-rho_fast) * activity[t]
slow[t] = rho_slow * slow[t-1] + (1-rho_slow) * activity[t]
```

The nonoverlapping ranges guarantee `0 < rho_fast < rho_slow < 1`, even at
sigmoid saturation. The same learned rates apply across samples in one layer,
but activity states are separate for each sample and reset for each sequence.
The model adds four trainable parameters in a two-layer network and no extra
recurrent state tensors relative to DTH. Extra sigmoid/arithmetic operations
still have a runtime cost; parameter count does not establish latency.

These are decay rates per input frame. An equivalent exponential timescale
is `-1/log(rho)` frames; the initial values are approximately 9.49 and 99.50
frames. Do not interpret those numbers as milliseconds or equate frame duration
with the neuron's configured `dt` across datasets.

The range bounds are explicit local architecture choices, not paper values.
Membrane/adaptation dynamics, threshold bounds, target rate, gains, recurrent
connections, reset, surrogate gradient and supervised loss stay inherited
from DTH. The new logits initially reproduce the fixed DTH rates up to floating
point rounding. DTH's gains start at zero, so trace-rate gradients initially
depend on those gains becoming active. This is expected; inspect learned rates
and gains before concluding the extra mechanism was used.

## Hypothesis and comparison

Fixed activity timescales may be unsuitable for all datasets or layers. The
hypothesis is that learning them improves DTH's activity-controlled thresholds
without adding a second adaptation-current mechanism. Improvement is not
guaranteed, and no novelty claim has been established. Learning neuronal time
constants is an existing direction, for example
[Fang et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf).
That work concerns membrane time constants and does not validate this specific
DTH trace extension.

This differs from the retained LG-DTH candidate: LG-DTH adds per-neuron local
activity corrections; LT-DTH keeps DTH's global activity signal and learns its
two layer-level trace rates. It also differs from the retained DA model's
mixture of discrete feedforward delays. Prior candidates remain in
`results/not_improved/`; they are not overwritten or relabeled.

Use DTH as the primary ablation control with matching training seeds. All
three configs inherit DTH architecture and the appropriate comparison group,
which inherits `config/paper_baseline.yaml`. LR, batch, preprocessing, loss,
budget, stopping, and split rules are unchanged. The saved model-variant
deviation and resolved hidden-layer configs identify the learned trace rates
and bounds. SHD* keeps its 20% training-derived validation split and seed 42.
SSC uses mean loss. The existing ECG reference budget/stopping caveat remains
if that reference is included in a broader table.

## Main-runner commands

Non-training preflight from the repository root:

```powershell
.\.venv\Scripts\python.exe runner.py --models LT_DTH_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/lt_dth_v1 --dry-run
```

The user launches the nine new jobs:

```powershell
.\.venv\Scripts\python.exe runner.py --models LT_DTH_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --logdir results/experiments/lt_dth_v1
```

Fresh training starts with random weights and a fresh optimizer at epoch zero;
it does not initialize from a trained DTH checkpoint. Repeating the same command
resumes interrupted jobs from their own `last.ckpt` and skips matching completed
jobs. Final tests use validation-selected best checkpoints. Automatic cost
measurement and aggregation follow the main runner. A cost failure does not
require retraining.

Outputs are under
`results/experiments/lt_dth_v1/LT_DTH_SE_adLIF/<dataset>/seed_<seed>/`.
No existing DTH, AMT or MT-consistency results are reused as output directories.
Use disjoint dataset/seed selections across devices and preserve repository-
relative layouts; Git does not transfer datasets or checkpoints.

Freeze this configuration before looking at its test results. Any tuning of
the rate ranges must use validation and a separate output/configuration.
Report per-seed results, mean, sample SD and actual cost measurements. For
paper comparisons use `(Δ paper)` headings and signed percentage-point deltas
from unrounded local metrics. Retain the follow-up disclosure: historical test
scores were already observed, and an added mechanism does not guarantee a
test improvement or establish statistical significance.
