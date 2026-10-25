# Tested candidates without accuracy improvement over DTH/MT

These completed SHD* experiments retain valid matched results. This category
means their observed three-seed mean accuracy did not exceed DTH/MT under the
tested configuration; it does not imply invalid training, statistical
significance, or failure on other datasets or efficiency measures.

| Model | SHD* test mean Â± sample SD (%) | Seeds |
|---|---:|---|
| DA | 93.33 Â± 0.93 | 42, 123, 456 |
| FP | 93.35 Â± 0.79 | 42, 123, 456 |
| LG-DTH | 93.74 Â± 0.71 | 42, 123, 456 |
| MR (K=2) | 93.89 Â± 0.18 | 42, 123, 456 |
| MT-DTH | 93.74 Â± 0.17 | 42, 123, 456 |
| [LT-DTH](LT_DTH_SE_adLIF/README.md) | 93.98 +/- 0.63 | 42, 123, 456 |
| [AMT](AMT_SE_adLIF/README.md) | 92.71 +/- 1.94 | 42, 123, 456 |

DTH: 94.49 Â± 0.52%; MT: 94.61 Â± 0.23%. See
[the full comparison](../../reports/reference-vs-models.md).

Each model retains all its dataset folders, seed folders, checkpoints, configs,
metrics, measurements and aggregate reports. MR retains its `K_2` hierarchy.
The model-level classification is based on SHD* accuracy; it does not assert
that every SSC/ECG experiment failed to improve. Saved run metadata retains
original paths as provenance. This folder is tracked by Git subject to normal
artifact exclusions; transfer ignored checkpoints separately.

The default runner refuses to recreate a retained experiment. Intentional new
experiments must use an explicit separate `--logdir`. Standalone measurement
readers locate these relocated seeds automatically.
