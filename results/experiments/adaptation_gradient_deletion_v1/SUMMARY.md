# Strict-precision matched event-deletion pilot

SHD*, MT/AMT, seed 42, 40 fixed validation recordings; predetermined natural repeat 0 gradients reused.
All 80 fresh clean references passed before deletion. Weights/state unchanged; no training or new backward.
1% exact event budget, MT absolute-gradient guidance and three uniform exact random controls; shared inputs for both models.
Random effects average within record, not independent seeds. Zero guidance is skipped for both models.

| Model | Eligible records | Mean absolute CE change, guided | Mean absolute CE change, random | Mean paired difference |
|---|---:|---:|---:|---:|
| MT_SE_adLIF | 40 | 0.391981025 | 0.0209890873 | 0.370991937 |
| AMT_SE_adLIF | 40 | 0.219145495 | 0.000203500513 | 0.218941994 |

Paired condition rows: 200; model evaluation rows: 400; shared deletion inputs: 160.
Signed finite CE changes and own-gradient linear predictions remain in MODEL_EFFECTS.csv. Their agreement is not a success gate.
MT-guided selection, confidence/loss coupling and recording 3896 limit architectural/causal interpretation. No superiority or significance claim.
Original source_high scores and all prior studies remain unchanged. Stage 7 is incomplete; review before expansion or optimized inputs.
