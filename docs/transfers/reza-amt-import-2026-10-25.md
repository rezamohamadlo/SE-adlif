# AMT result import from external storage

On 25 October 2026, copied all nine completed AMT runs from the explicitly
supplied external root `F:/reza/results/experiments/amt_v1/AMT_SE_adLIF/` into
`results/not_improved/AMT_SE_adLIF/`. This retention category is based on SHD*
accuracy relative to MT/DTH; SSC and ECG outcomes are assessed independently.

Copied 127 external files and preserved five pre-existing local startup files
in `startup_history/local-seed42-before-import/`: 132 files and 156,732,487 bytes
in total. The original local startup attempt contains neither checkpoints nor
metrics and was not continued or counted as an independent completed run.
External originals and original local startup files were retained.

Each copy was SHA256-verified against its source. All nine best-checkpoint
hashes match their `COST_RESULTS.json` identities. Each run has its own
`last.ckpt`, full-budget validation history without duplicate validation epochs,
final test accuracy and complete cost evidence. Test metrics agree with the
cost evaluator's reproduced accuracies within floating point tolerance.

The explicit `results/experiments/amt_v1` root alias maps to
`results/not_improved`, preserving model/dataset/seed/file suffixes. Saved
configs, summaries, checkpoint bytes and cost records were not rewritten.
No existing local files were overwritten. No other models' active outputs or
processes were changed, and no training or evaluation was launched.

The main runner's non-training preflight recognizes all nine imported runs as
`complete` with matching normalized configurations. The source aggregate is
preserved at the retained AMT model root. Outcome and configuration notes are
in its [README](../../results/not_improved/AMT_SE_adLIF/README.md).

[Shared transfer manifest](../../results/provenance/reza-amt-import-2026-10-25.json)
records per-file source/destination paths, sizes, hashes, run identities,
baseline/protocol metadata and checkpoint identities. Local startup source
paths are stored literally rather than alias-normalized so their provenance
remains distinct from the imported completed runs.
