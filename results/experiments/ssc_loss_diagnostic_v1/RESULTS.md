# SSC loss diagnostic - validation only

Fresh initialization per arm/seed; full resume only for interrupted jobs. The canonical reference is unchanged. Training/selection used validation only; a separately authorized reference seed-42 test evaluation is recorded below.

| Loss | Seed | Status | Best validation (%) | Best epoch (zero-based) |
|---|---:|---|---:|---:|
| summed_membrane_potentials | 42 | complete | 80.16 | 29 |
| mean_membrane_potentials | 42 | complete | 82.22 | 18 |
| summed_membrane_potentials | 123 | pending/incomplete | pending | pending |
| mean_membrane_potentials | 123 | pending/incomplete | pending | pending |
| summed_membrane_potentials | 456 | pending/incomplete | pending | pending |
| mean_membrane_potentials | 456 | pending/incomplete | pending | pending |

| Loss | Completed seeds | Mean best validation (%) | Sample SD (pp) |
|---|---:|---:|---:|
| summed_membrane_potentials | 1 | 80.16 | unavailable |
| mean_membrane_potentials | 1 | 82.22 | unavailable |

These are validation-selection results, not final test accuracy. The experiment was designed after observing historical test results; do not describe a later evaluation as an untouched independent test.

## Separate official test evaluation - 19 October 2026

| Model / loss | Training seed | Best validation (%) | Selected epoch | Official test (%) | Test loss |
|---|---:|---:|---:|---:|---:|
| Reference / mean | 42 | 82.22 | 18 | 80.69 | 0.670582 |

Evaluated the validation-selected checkpoint without training. This is one seed, not a multi-seed confirmation or proof of the cause of the historical gap. The run was subsequently promoted to the canonical reference, retaining its diagnostic provenance. Original training configs retain run_test=false as historical training provenance.

## Reference promotion - 19 October 2026

The completed mean-loss reference seed 42 was moved to [the canonical reference](../../valid/reference/SSC/RESULTS.md), with all original files verified. Its 82.22% validation and 80.69% official test scores remain the same result, not additional independent runs. This diagnostic queue is retired; use `ssc_mean_comparison.py` for the two fresh DTH/MT jobs.
