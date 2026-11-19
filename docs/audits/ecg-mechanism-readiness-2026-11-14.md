# ECG mechanism-study readiness — 14 November 2026

**Update after user execution:** all 11 v1 tests passed, but the first clean
anchor failed. This readiness report did not verify predictions. See the
[precision-contract investigation](ecg-clean-anchor-precision-2026-11-14.md)
and the [current four-pair queue](../experiments/adaptation-ecg-four-pair-queue-v2.md).
The original preflight findings below remain historical metadata evidence.

**Status:** checkpoint/data preflight passed; pilot implementation prepared;
user contract tests and model evaluation remain pending. No training, inference,
feature extraction, regression, bootstrap or official-test access was performed.

The current machine-readable evidence is
[ECG_PREFLIGHT.json](../../results/experiments/adaptation_mechanism_study_v1/ECG_PREFLIGHT.json).
An earlier development preflight snapshot is retained in the study's
`preflight_history/` directory. It predates a change making a mocked user
contract test independent of local data files; it is not the launch authority.

| Readiness item | Verified from existing files |
|---|---|
| Checkpoints | 18/18, six registered models × seeds 42/123/456 |
| Checkpoint identity | Exact SHA-256, selected epoch, saved config and strict state loading; no forward |
| Selection evidence | One complete 400-epoch validation history per run; selected accuracy is a validation maximum |
| Baseline | Inherits `paper_classification_v1`, ECG section; LR .01, batch 64, 400 epochs, two 36-neuron layers |
| Pair fairness | Four parent/variant pairs match outside explicit neuron or MT-TC training-objective differences |
| Training MAT | `data/QTDB/QTDB_train.mat`; SHA-256 `72d1a10c96f1540880447e07256835d66da419c0b34f3dff495731a7928c5bc7` |
| Stored input | 618 × 1301 × 4; signed events −1/0/+1 |
| Validation | Fixed split seed 42, 588 training and 30 validation rows, original order preserved |
| Target alignment | Six class channels per time step; original argmax, block IDs and zero burn-in preserved |
| All-zero target vectors | 2,670 validation positions map to class 0 under the original loader; preserved and documented |
| Loader contract | Three validation records exactly match existing wrapper/collator tensors; input-only check |
| Official test | Not opened; DataModule constructor/setup not used |
| Scientific outputs | None from this task; clean anchors and deletion effects remain unexecuted |

The all-zero target vectors are not treated as padding or removed. Their
physiological meaning has not been independently established. Changing that
policy would define a different evaluation protocol and cannot repair or
reinterpret completed training retrospectively. Likewise, no patient IDs or
physical millisecond conversion are inferred from this MAT.

The [prepared pilot](../experiments/adaptation-ecg-alignment-pilot-v1.md) first
runs 11 targeted user contract tests, then six AMT/MT clean anchors on the full
30-row validation split. Every anchor must pass before three shared 10%
signed-event deletion realizations run. It writes 24 resumable prediction
chunks and 360 paired records. That remains a bounded diagnostic pilot;
features, aligned time scaling, the other pairs and statistical analysis
remain pending. Stage 6 stays open at 8/12 completed mechanism studies.
