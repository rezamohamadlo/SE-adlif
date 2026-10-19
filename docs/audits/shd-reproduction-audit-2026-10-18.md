# SHD* paper-gap investigation

Snapshot: 18 October 2026. No training, inference on the official test set, process interruption, dependency change or training-config change was performed for this investigation.

**Finding: no confirmed implementation error explaining the 0.64 percentage-point gap was found.** Unlike SSC, SHD* has no temporal-loss mismatch against the released implementation. Several historical execution details and the authors' exact run seeds remain unverified; these are possible sources of variation, not demonstrated causes.

## Verified results

| Training seed | Best validation epoch (zero-based) | Best validation (%) | Final test (%) |
|---|---:|---:|---:|
| 42 | 261 | 99.14 | 92.93 |
| 123 | 204 | 99.39 | 92.89 |
| 456 | 243 | 99.08 | 93.64 |

Local mean/sample SD is 93.15 +/- 0.42%, n=3. Paper Table 2 reports 93.79 +/- 0.76%, n=20, for two-layer SHD*. Local minus published rounded mean is -0.6363 pp. A different number of runs does not itself establish why the means differ, and adding seeds does not guarantee an increased mean.

All three selected checkpoint SHA256 hashes match the [canonical aggregate](../../results/valid/reference/SHD/REFERENCE_RESULTS.json); saved callback best-validation scores agree with the aggregate and canonical metrics. Each canonical CSV contains 300 validation epochs. Each last checkpoint reaches epoch 299, global step 7500, with optimizer state. Nested historical Hydra logs were not pooled into these histories.

## Paper and released-source comparison

Paper sources: local [PDF](../../reference_paper.pdf), Table 2 on page 11, preprocessing/loss/optimization on page 28, initialization on page 29 and Table 7 on page 35. Released source is pinned to authors' commit `9a8eea4452a6df5bd25f51ca03b3b45000fb6de1`. This does not establish the exact source revision used for the published runs.

| Item | Evidence and conclusion |
|---|---|
| Correct target | Compare against SHD* 93.79%, not historical test-selected SHD 95.81%. |
| Validation protocol | Our records use a 20% training holdout, split seed 42, and no additional test-set validation. The paper describes this for SHD*. The [released default dataset config](https://github.com/IGITUGraz/SE-adlif/blob/9a8eea4452a6df5bd25f51ca03b3b45000fb6de1/config/dataset/shd.yaml) instead validates on `test`; that default corresponds to the other protocol, not the appropriate SHD* reproduction command. Do not revert to test selection to close the gap. |
| Loss | Upstream [experiment config](https://github.com/IGITUGraz/SE-adlif/blob/9a8eea4452a6df5bd25f51ca03b3b45000fb6de1/config/experiment/SHD_SE_adLIF.yaml) uses `softmax`; local uses `sum_softmax_over_time`. Both sum per-timestep softmax outputs and then apply cross-entropy. All six reference checkpoints record the local alias. A synthetic CPU comparison found exactly equal aggregated logits, cross-entropy and gradients. The SSC mean-versus-sum experiment does not apply here. |
| Architecture and neuron dynamics | Two recurrent 360-neuron hidden layers, 140 inputs, 20 outputs. Shared reference neuron, LI output, surrogate and time-constant files matched pinned upstream in the [shared source comparison](ssc-reproduction-source-comparison.json). |
| Initialization | Same upstream weight, recurrence and neuron-parameter initialization. No alternative initialization difference identified. |
| Hyperparameters | Saved LR=0.01, batch=256, epochs=300, dropout=0.15, alpha=5, c=0.4, q=120, tau_u=[5,25], tau_w=[60,300], output tau=15; matches paper Table 7. |
| Adam and scheduler | Checkpoints confirm betas=[0.9,0.999], eps=1e-8, weight_decay=0. Every logged LR is 0.01. Plateau patience=9999 prevents decay within this budget, matching the released scheduler/config path. |
| Clipping and stopping | Config/default gradient norm clipping=1.5; early stopping disabled; full budgets completed. Matches paper clipping and avoids the historical local stopping deviation. Not every historical runtime setting is embedded in checkpoints. |
| Preprocessing | Same framing/pooling pipeline as released code: 4-ms bins, five-channel groups, minimum 250 timesteps, first ten output timesteps ignored. Local wrapper widens timestamps before conversion; retain this compatibility correction. |
| Cached data | Local namespace includes split/preprocessing settings. Twelve sampled current train/validation cache entries exactly match fresh tensors, labels and masks. No sampled stale-cache mismatch was found. This is not a full-cache integrity proof. |

## Split and cache checks

Reconstructing the saved split produces 6,525 training and 1,631 validation samples, with no index overlap. All three saved reference configs/checkpoints specify split seed 42 independently of training seed. Index hashes and sample comparisons are in [the evidence JSON](shd-reproduction-evidence.json).

The current cache directory is `data/cache/SHD/controlled_v2_61555efccb7a8796`. Its name is a cache namespace, not proof that training used the retired controlled_v2 optimization protocol. Raw-versus-cache comparisons read existing files directly, without regenerating or deleting caches. The upstream splitter also defaults to seed 42, but exact authors' split indices and their historical PyTorch random-permutation behavior were not recovered. No split search is recommended: project comparisons must keep split seed 42 fixed.

The current validation/test accuracy separation is descriptive only. It does not establish leakage, a loader bug or a particular cause of the paper gap; the official test set is distinct from the training-derived validation set.

## Remaining uncertainties

1. **Author-run provenance:** exact 20 training seeds, exact split-index hashes, dependency/backend snapshot and the precise published training revision are unavailable in the inspected materials. The released default test-selection config does not supply a separate complete SHD* run manifest.
2. **Execution differences:** local saved runs use zero data-loader workers versus eight in the released dataset config. Current local `run.py` also explicitly requests float32 matmul precision `high`, whereas released `run.py` does not. Package/CUDA specifications differ. Random-number consumption or numerical behavior could differ, but neither an accuracy effect nor its direction has been established.
3. **Resume history:** last checkpoints establish completion, but not identical dropout/data-order sequences to uninterrupted author runs. Saved source/config metadata is not a full runtime trace.
4. **Sampling uncertainty:** three local initializations and 20 paper initializations do not provide matching-seed comparisons. The reported difference alone cannot distinguish random variation from a systematic reproduction difference.

These points are hypotheses or missing evidence, not confirmed explanations. The NumPy/Tonic conflict identified during the earlier SSC audit has since been corrected locally; that change cannot be retroactively attributed to the historical SHD* runs.

## Recommended next steps

- First recover author-run seeds/split indices and an environment or run manifest if publicly available or supplied by the authors; this would remove more ambiguity than changing the SHD loss.
- If a new experiment is wanted, prepare a separate validation-only execution-fidelity comparison, keeping architecture, loss, split seed 42 and paper hyperparameters fixed. Prespecify execution settings and assess validation before any further test evaluation. Do not change the main runners or current baseline to implement the diagnostic.
- Additional prespecified training seeds would improve estimation of the reference mean, not guarantee closing the gap. Preserve the existing three-seed reference and report the expanded sample honestly; never retain only favorable seeds.
- An LR/dropout/architecture search would answer an improvement question, not reproduce the original paper. Keep it separately labeled and validation-selected if requested later.

## Artifacts and verification

- [SHD source comparison](shd-reproduction-source-comparison.json): pinned SHD config, preprocessing and cache-loader sources, hashes and diffs.
- [SHD local evidence](shd-reproduction-evidence.json): resolved configs, checkpoints, optimizer settings, observed LRs, split fingerprints, cache samples and loss equivalence.
- [Shared reference-source comparison](ssc-reproduction-source-comparison.json): common model/initialization/output/surrogate/runner code comparison from the SSC audit.

Ten existing split/cache, temporal-loss and paper-config tests passed. No active training was stopped or changed, no new training was launched and no official test predictions were computed.
