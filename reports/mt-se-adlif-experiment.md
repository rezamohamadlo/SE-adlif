# MT-SE-adLIF experiment

This proposed neuron retains SE-adLIF's fixed threshold, recurrent connections,
and symplectic update order. Two adaptation currents share the original a, b,
and q parameters but have separately learned time constants:

```text
g = sigmoid(mix_logits)                       # per neuron, initially 0.5
w_effective = g * w_fast_previous + (1-g) * w_slow_previous
u = alpha * u_previous + (1-alpha) * (current - w_effective)
z = spike(u - threshold)
u = reset(u, z)
drive = q * (a * u + b * z)
w_fast = beta_fast * w_fast_previous + (1-beta_fast) * drive
w_slow = beta_slow * w_slow_previous + (1-beta_slow) * drive
```

The adaptation time-constant ranges are [60, 300] and [300, 600]. Both states
start at zero for every sample. The slow time constants and mixing weights add
1,440 trainable parameters for two 360-neuron hidden layers. Convex mixing
bounds the mixture coefficients, but does not by itself prove stability of
the coupled neuron dynamics; monitor training and state trajectories.

Implementation: `models/ours/mt_adlif.py`.
Configuration: `config/experiment/SHD_MT_SE_adLIF.yaml`.
The original SE-adLIF and DTH models remain separately selectable.

One-epoch integration check:

```powershell
.\.ven\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD --seeds 42 --epochs 1 --logdir results/model_checks --resume 0
```

Shared-protocol SHD run (seed 42, maximum 300 epochs, batch 512):

```powershell
.\.ven\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD --seeds 42 --resume 1
```

Results use `results/valid/protocol_v2/MT_SE_adLIF/<dataset>/seed_42/`.
The shared protocol uses 20% training-derived SHD validation, separate from
the official test set. No novelty claim has been established.

## Completed seed 42 results

| Dataset | Final test accuracy | Test loss |
|---|---:|---:|
| SHD | 92.76% | 0.5292 |
| SSC | 77.60% | 1.3197 |
| ECG | 87.23% | 0.3349 |

The SHD result was recovered by independently evaluating the validation-selected
epoch-104 checkpoint; its original training logs contain overlapping runs.
ECG selected epoch 258 and ended with final testing recorded at epoch 260.
See [seed 42 comparison](seed-42-model-comparison.md) for reference differences
and the ECG stopping-setting caveat.

A historical local training-progress snapshot is retained in
[the documentation archive](archive/mt-shd-local-progress.md).
