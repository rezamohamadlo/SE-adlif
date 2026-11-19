# SHD* batching/precision follow-up

User-launched forward-only diagnostic: MT/AMT, seed 42, 40 pilot records; original-batch checks cover all 1,631 validation records.
Source-high original anchors passed unchanged tolerances. Stricter precision results are diagnostic and do not replace saved evidence.
Three precision modes, original / natural singleton / original-padded singleton layouts, plus four controls for recording 3896.
All state hashes unchanged. No gradients, training, official test, inference statistics or automatic precision selection.

| Model | Mode | Layout | Prediction changes vs same-mode original | Max absolute CE drift |
|---|---|---|---:|---:|
| MT_SE_adLIF | source_high | singleton_natural | 1 / 40 | 0.338704586 |
| MT_SE_adLIF | source_high | singleton_original_padding | 1 / 40 | 0.338703513 |
| MT_SE_adLIF | highest_matmul | singleton_natural | 0 / 40 | 6.61611557e-06 |
| MT_SE_adLIF | highest_matmul | singleton_original_padding | 0 / 40 | 5.54323196e-06 |
| MT_SE_adLIF | highest_matmul_cudnn_disabled | singleton_natural | 0 / 40 | 5.54323196e-06 |
| MT_SE_adLIF | highest_matmul_cudnn_disabled | singleton_original_padding | 0 / 40 | 5.54323196e-06 |
| AMT_SE_adLIF | source_high | singleton_natural | 0 / 40 | 0.000444921257 |
| AMT_SE_adLIF | source_high | singleton_original_padding | 0 / 40 | 0.000444921257 |
| AMT_SE_adLIF | highest_matmul | singleton_natural | 0 / 40 | 0 |
| AMT_SE_adLIF | highest_matmul | singleton_original_padding | 0 / 40 | 0 |
| AMT_SE_adLIF | highest_matmul_cudnn_disabled | singleton_natural | 0 / 40 | 0 |
| AMT_SE_adLIF | highest_matmul_cudnn_disabled | singleton_original_padding | 0 / 40 | 0 |

Inspect SUMMARY.json focus comparisons and COMPARISONS.csv for fixed-layout precision and padding contrasts.
Repeat/order controls are limited to recording 3896. Highest precision is not exact arithmetic; this matrix alone does not prove a unique cause.
No gradient maps were regenerated. Stage 7 remains incomplete; review before expanding or changing the gradient protocol.
