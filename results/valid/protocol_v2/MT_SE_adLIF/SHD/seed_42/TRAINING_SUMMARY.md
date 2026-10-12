# MT_SE_adLIF SHD — Seed 42: verified checkpoint evaluation

Selected from the latest training checkpoint callback using validation accuracy.
No additional training was performed. The original folder had overlapping logs;
the original metrics CSV remains a mixed training history. The independent
evaluation and checkpoint provenance are preserved in
`RESULT.json` alongside this summary. The separate `mt_verified` folder
was removed after its verified summary was copied here.

- Best checkpoint: `ckpt/epoch=104-step=1260.ckpt`
- Validation accuracy: 98.84%
- Validation split: 20% of training, split seed 42
- Training ended at epoch: 126

## Final test evaluation

| Metric | Value |
|---|---:|
| Test accuracy | **92.76%** |
| Test loss | **0.5292** |
