# SSC loss diagnostic - validation only

Fresh initialization per arm/seed; full resume only for interrupted jobs. The canonical reference is unchanged. No official test evaluation.

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
