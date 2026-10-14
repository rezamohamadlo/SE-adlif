# Current paper/reference comparison

Final test accuracy (%), mean ± sample SD for local two-layer reference models.

| Dataset | Paper | Current reference | Difference (pp) | Paper/local runs |
|---|---:|---:|---:|---:|
| SHD* | 93.79 ± 0.76 | 93.15 ± 0.42 | -0.64 | 20/3 |
| SSC | 80.44 ± 0.26 | 78.21 ± 0.11 | -2.23 | 5/3 |
| ECG | 88.18 ± 0.36 | 88.32 ± 0.35 | +0.14 | 20/3 |

The canonical SHD* reference now uses batch 256 and 300 epochs without early
stopping. Per-seed test scores: 42 = 92.93%, 123 = 92.89%, 456 = 93.64%.
SSC and ECG reference records are unchanged. ECG retained inconsistent
stopping settings across seeds, so its comparison is descriptive.

[Full paper comparison](paper-vs-reference.md) provides sources and provenance.
[Seed-42 model comparison](seed-42-model-comparison.md) lists proposed-model
results; their controlled-v2 SHD settings do not match the new reference, so
no fair SHD improvement claim can be made from those old runs.

Historical test-selected SHD reproduction is documented only in the
[Phase 1 report](phase-1-se-adlif-reference.md), not in this current comparison.
