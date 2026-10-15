# DTH-SE-adLIF experiment

This experimental model adds a bounded dynamic threshold to the original
symplectic SE-adLIF membrane and adaptation-current equations. It is a research
hypothesis; accuracy improvement and novelty have not been established.

For each sample and layer, previous spikes update two activity traces:

```text
activity = mean(previous_spikes over neurons)
fast = 0.9 * previous_fast + 0.1 * activity
slow = 0.99 * previous_slow + 0.01 * activity
theta = 0.5 + sigmoid(v + gain_homeostasis * (slow - 0.05)
                       - gain_transient * abs(fast - slow))
```

The neuron-specific parameter `v` learns the baseline threshold. Two scalar
gains per layer control homeostasis and transient sensitivity. Gains start at
zero and are projected into [0, 5] after optimizer updates. Both traces start at
the target rate and reset for every sequence. Samples do not share activity.
The initial threshold is exactly 1, matching the original model. With two
360-neuron layers, this adds 724 trainable parameters.

The original `models/alif.py` and `SHD_SE_adLIF.yaml` are retained. The new class
is in `models/ours/dth_adlif.py`; future proposed models belong in `models/ours/`.
Its inherited experiment configuration is
`config/experiment/SHD_DTH_SE_adLIF.yaml`.

Run a development experiment from the project root:

```powershell
.\.ven\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets SHD --seeds 42 --resume 1
```

Outputs are saved under:

```text
results/valid/protocol_v2/DTH_SE_adLIF/SHD/seed_42/
```

The one-epoch integration check uses a separate output root:

```powershell
.\.ven\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets SHD --seeds 42 --epochs 1 --logdir results/model_checks --resume 0
```

The check verifies execution rather than final accuracy. Current shared
settings use 20% training-derived SHD validation, with official test data
evaluated after checkpoint selection.
