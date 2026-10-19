# SSC reproduction fidelity audit

Audit date: 18 October 2026. No training, test-set reevaluation, baseline changes or environment changes were performed.

The current SSC reference is 78.21 +/- 0.11% over seeds 42, 123 and 456, versus the paper's rounded 80.44 +/- 0.26% over five runs: a -2.23 percentage-point difference. The audit identifies a consequential difference from the released code, but does not establish the cause of that accuracy difference.

## Sources and scope

- Local [paper](../../reference_paper.pdf): preprocessing, optimizer, loss and reset details on PDF page 28; reparameterization and initialization on page 29; hyperparameters in Tables 6/7 on page 35; code availability on page 36.
- Authors' [repository](https://github.com/IGITUGraz/SE-adlif/tree/9a8eea4452a6df5bd25f51ca03b3b45000fb6de1), pinned to upstream commit `9a8eea4452a6df5bd25f51ca03b3b45000fb6de1`, retrieved on the audit date. This is released source available now, not proof of the precise revision used for the published scores.
- [Source comparison](ssc-reproduction-source-comparison.json): URLs, upstream/local SHA256 hashes and diffs for 12 relevant files. Equal text comparisons ignore line-ending differences.
- [Local evidence](ssc-reproduction-evidence.json): resolved reference configs, checkpoint metadata, optimizer settings, observed learning rates, dataset sample diagnostics and CPU/CUDA loss diagnostics.

## Confirmed loss difference: paper versus released code

The paper describes temporally summed SSC output, with the first ten timesteps discarded. The released [SSC config](https://github.com/IGITUGraz/SE-adlif/blob/9a8eea4452a6df5bd25f51ca03b3b45000fb6de1/config/experiment/SSC_SE_adLIF.yaml) instead selects `mean_membrane_potentials`. Its [loss implementation](https://github.com/IGITUGraz/SE-adlif/blob/9a8eea4452a6df5bd25f51ca03b3b45000fb6de1/models/pl_module.py) uses a temporal mean for every non-`softmax` aggregation setting.

Our [SSC config](../../config/experiment/SSC_SE_adLIF.yaml) and [loss implementation](../../models/pl_module.py) use an actual temporal sum. Local commit `aca2b5b4f0bf178759a5c9a020cb1cbd81e3657c` introduced both the explicit aggregation implementation and this SSC config change. All six original reference checkpoints (best and last for three seeds) record `summed_membrane_potentials`. Their metadata is consistent with the local change; checkpoints do not embed the full training source, so this is not independent execution tracing.

For N valid timesteps, summed logits equal N times averaged logits. Argmax predictions for a fixed output sequence agree, but cross-entropy and its gradients generally differ. Therefore changing aggregation only when testing an existing checkpoint cannot test the training hypothesis. A fresh run would be needed.

A synthetic diagnostic with 300 timesteps, ten ignored, and per-timestep logits [0.01, -0.01, 0] gave the following for target class 1:

| Aggregation | Valid logits | Cross-entropy | Output-gradient L2 norm |
|---|---|---:|---:|
| Sum | [2.9, -2.9, 0] | 5.856424 | 23.4132 |
| Mean | [0.01, -0.01, 0] | 1.108646 | 0.048186 |

Both modes had finite gradients on CPU and CUDA, with matching results within floating-point precision. These are illustrative synthetic values, not measured training gradients or evidence that mean pooling improves accuracy.

Conclusion: the current reference matches the paper's stated temporal-sum description, but is not an exact reproduction of the released SSC training loss. The loss discrepancy is the first candidate to isolate before a broad hyperparameter search. Do not silently change the baseline or relabel completed runs.

## Other fidelity checks

| Component | Finding |
|---|---|
| Hidden neurons and equations | `models/alif.py` is text-identical to pinned upstream, including SE update ordering, detached reset and recurrent input. Two layers of 720 neurons are saved in all reference configs. |
| Initialization | Same upstream implementation: feed-forward uniform bounds +/-sqrt(1/fan-in), zero bias, orthogonal recurrence with gain 1, uniform normalized a/b and time-constant parameters. Initial states are zero under saved defaults. |
| Surrogate gradient | `models/helpers.py` is text-identical upstream; SLAYER alpha=5, c=0.4 matches Table 7. |
| Time constants | `module/tau_trainers.py` is text-identical upstream. Interpolation and clipping match the paper; membrane range [5,25], adaptation range [60,300], q=120. |
| Output layer | `models/li.py` is text-identical upstream. Leaky integrator output, 35 classes and tau=15. |
| Dropout and constraints | Model forward/dropout and parameter-constraint hooks have no changes in the upstream/local model-module diff. Saved dropout is 0.15. |
| Optimizer | Last checkpoints confirm Adam LR=0.006, betas=[0.9,0.999], eps=1e-8, weight_decay=0. Paper values agree where specified. |
| Scheduler | Same plateau scheduler path as upstream. Patience=9999 exceeds the 40-epoch budget; every logged reference LR is 0.006. The paper does not independently specify this classification scheduler. |
| Gradient clipping | Upstream trainer hard-codes 1.5; local default is 1.5. Matches the paper. Historical checkpoints do not record every trainer option or every clipped gradient. |
| Budget and stopping | Saved configs disable early stopping. All three last checkpoints reach epoch 39, step 11760, with optimizer state. Forty validation epochs are present per seed. |
| Selection and testing | Both runners monitor configured validation accuracy and test `ckpt_path="best"`. SSC uses official train/valid/test datasets; no test-based checkpoint selection is evident in this path. |
| Framing and pooling | Shared transform pipeline matches upstream: spatial factor 0.2 (140 channels), time factor 0.001, 4-ms windows, minimum padding 300, ten ignored output timesteps. |
| Collation and padding | `datasets/utils/pad_tensors.py` and `datasets/utils/transforms.py` match upstream. Batch padding is masked; minimum-length padding remains part of the usable sequence, as upstream implements it. |
| Batch size and workers | Batch=256 matches paper/upstream. Saved num_workers=0 versus upstream dataset config 8. This is a local execution difference; equivalence of all random sequences is not established. |
| Numerical precision | Current local runner sets float32 matmul precision to `high`; pinned upstream does not explicitly set it. Historical per-run matmul/backend state is not saved, so its effect on these results cannot be established. |

The PyTorch cross-entropy call consumes aggregated logits directly and performs its own normalization; this audit does not recommend adding a second explicit softmax based on the paper's mathematical notation.

## Timestamp compatibility fix: retain it

Local `datasets/ssc.py` widens HDF5 timestamps to float64 before multiplying by 1e6. Upstream delegates to Tonic, whose installed SSC implementation multiplies first. The first sample in each train/valid/test file is float16: under the inspected environment, the original multiplication produces nonfinite values, while widening first produces finite values. Both local dataset locations were sampled; this is not a complete dataset integrity or duplicate-hash audit.

The official files contain 75,466 train, 9,981 validation and 20,382 test labels in each inspected location. The correction preserves the intended timestamp unit conversion and should not be reverted merely to make the wrapper identical upstream. It does not by itself establish the preprocessing behavior of the authors' original environment.

## Current environment discrepancy

The actual project environment available here is `.venv`; `.ven` is absent. No global Python was used. Installed versions are in the evidence JSON. In particular, NumPy is **2.4.6**, while `requirements.txt` pins **1.26.4** and Tonic 1.6.0 requires NumPy below 2.0. `pip check` fails on that dependency conflict.

The authors' environment specification also differs from this Windows environment (including PyTorch/CUDA declarations). Saved reference checkpoints record Lightning 2.2.1, but do not provide a complete historical dependency/backend snapshot. Today's environment discrepancy must not be assigned retrospectively as the cause of the saved accuracy gap.

Before any new reproduction run, establish a project environment consistent with the documented dependency pins and capture its package/backend versions. This audit did not install, downgrade or modify dependencies.

## Recommended next experiment, not launched

After environment validation, use a separate experiment config inheriting `config/paper_baseline.yaml`, with `loss_agg: mean_membrane_potentials` as an explicit deviation and a new output directory under `results/experiments/`. Keep all other settings matched. Compare against a fresh summed-loss control in the same environment; historical controls alone cannot isolate the loss if the environment changes. Both arms start from random weights with fresh optimizer state at epoch 0.

Use identical prespecified training seeds, the official validation split and the same full 40-epoch budget. Initially disable test evaluation, select using validation alone, and retain the fact that this diagnostic was proposed after inspecting existing test results. Fresh seeds do not make the previously observed official test set unseen again. Confirm a frozen choice before reporting further test results. No accuracy gain is guaranteed.

## Verification and limits

Nine existing loss-aggregation, classification-target-dtype and paper-config tests passed. The synthetic loss comparison ran on CPU and CUDA with finite gradients. Source hashes/diffs, six checkpoint metadata records, all reference LR histories and sample timestamp conversions were checked. No full dataset scan, full validation inference, training, environment repair or author-run artifact recovery was performed. The paper/released-code discrepancy remains unresolved without evidence of the exact published training revision.

Subsequent experiment preparation on the same date restored NumPy 1.26.4 in `.venv`, and `pip check` then passed. The evidence JSON retains the original audit-time environment snapshot. The [separate diagnostic guide](../experiments/ssc-loss-diagnostic-guide.md) provides user-run commands; training has not been launched.
