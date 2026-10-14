# SHD dropout search

Fresh training at LR 0.01, batch 512; only dropout differs from the reference control.
Validation-only selection. Test evaluation remains separate.

| Trial | Dropout | Status | Latest epoch | Best validation | Validation loss |
|---|---:|---|---:|---:|---:|
| d25 | 0.25 | complete | 260 | 99.203% | 0.04210 |
| d35 | 0.35 | queued | None | pending | pending |
