# Comparison-grid reporting audit - 20 October 2026

Verified all 27 expected slots for reference/DTH/MT, datasets SHD*/SSC/ECG and training seeds 42/123/456. Twenty-one have local completed training/test evidence. Eighteen match the intended current configuration; the three reference ECG records are accepted historical evidence under the user agreement, retaining explicit unmatched stopping/budget labels. Six current mean-loss SSC seeds are absent locally.

Canonical metrics have unique validation epochs, recorded final test scores and present best/last checkpoints. Selected checkpoint epochs and callback validation scores agree with the metrics. Last checkpoints confirm 300/40/400 epochs for matched SHD*/SSC/ECG runs. Historical ECG reference last checkpoints confirm 383, 283 and 185 completed epochs; the seed-123/456 summaries were corrected from their prior one-epoch overcounts. No raw metrics, saved configs or checkpoints were modified.

Updated the per-seed summaries, model tables, reference ECG aggregate, active SSC comparison, project grid/README and the main paper/reference/model reports. Retained the old summed-loss results as explicitly labeled archived evidence. Paper-source accuracy values and unrelated SHD* model results were unchanged. No training, new inference, cost measurement or process modification was performed.

[Current per-seed evidence](../../results/valid/COMPARISON_STATUS.md) and [machine-readable grid](../../results/valid/COMPARISON_STATUS.json). Main reports remain in reports/; this audit is documentation. The agreement accepts historical evidence but does not claim matched configuration merely because an accuracy is close to the paper.
