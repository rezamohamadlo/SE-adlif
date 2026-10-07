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
.\minimal_runner.ps1 -ModelVariant MT_SE_adLIF -Epochs 1 -LogDir results/model_checks
```

Full development run (default SHD, seed 42, 300 epochs, batch 512):

```powershell
.\minimal_runner.ps1 -ModelVariant MT_SE_adLIF -LogDir results/model_development
```

Results use `ours/MT_SE_adLIF/SHD/seed_42/` beneath the chosen output root.
No accuracy improvement or novelty claim has been established. This config
inherits the reference SHD test-as-validation setup. A separate validation
split is needed before tuning or selecting the final model.
