# SHD training from scratch

Ranked by completed-run validation accuracy, then validation loss. Test results are displayed only after explicit evaluation.
r06 is the fresh reference control: LR 0.01, batch 512. All runs use the same 20% training-derived validation split (split seed 42).

| Trial | Seed | LR | Batch | Status | Latest epoch | Best epoch | Val accuracy | Val loss | Test accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| r01 | 42 | 0.005 | 128 | complete | 102 | 95 | 98.896% | 0.06187 | pending |
| r04 | 42 | 0.01 | 128 | complete | 160 | 110 | 98.835% | 0.06523 | pending |
| r05 | 42 | 0.01 | 256 | complete | 117 | 105 | 98.835% | 0.07719 | pending |
| r02 | 42 | 0.005 | 256 | complete | 140 | 111 | 98.835% | 0.10078 | pending |
| r08 | 42 | 0.015 | 256 | complete | 105 | 55 | 98.712% | 0.05175 | pending |
| r03 | 42 | 0.005 | 512 | complete | 157 | 156 | 98.712% | 0.09278 | pending |
| r09 | 42 | 0.015 | 512 | complete | 73 | 23 | 98.529% | 0.07514 | pending |
| r06 | 42 | 0.01 | 512 | complete | 147 | 97 | 98.467% | 0.09371 | pending |
| r07 | 42 | 0.015 | 128 | complete | 71 | 21 | 98.345% | 0.06606 | pending |

## Validation across completed seeds

| Trial | Completed seeds | Mean val accuracy | Sample SD (pp) |
|---|---|---:|---:|
| r01 | 42 | 98.896% | pending |
| r02 | 42 | 98.835% | pending |
| r03 | 42 | 98.712% | pending |
| r04 | 42 | 98.835% | pending |
| r05 | 42 | 98.835% | pending |
| r06 | 42 | 98.467% | pending |
| r07 | 42 | 98.345% | pending |
| r08 | 42 | 98.712% | pending |
| r09 | 42 | 98.529% | pending |

Compare configurations on matching seed sets. Epoch numbers are zero based.
Repeat --report to refresh in-progress metrics. Training updates the report after each run.
