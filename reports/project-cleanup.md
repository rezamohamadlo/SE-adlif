# Project cleanup — 14 October 2026

Completed repository cleanup:

- Removed 27 tracked, generated `trial.yaml` files. Launchers regenerate them
  from the retained `SWEEP_CONFIG.json` manifests. Future generated files are
  ignored; per-run summaries, metrics and checkpoints are preserved.
- Moved the local MT progress snapshot and overnight planning notes into
  `reports/archive/`.
- Moved `table.docx` to `reports/exports/model-comparison.docx`.
- Moved 19 audio evaluation CSVs from the misleading `checkpoints/` directory
  to `reports/audio-compression/`.
- Replaced the obsolete full-runner guide with pointers to maintained docs.
  Updated controlled/DTH/MT commands and removed claims that deleted launchers
  still exist. Updated the roadmap and Windows installation guidance.
- Left all main runners, model implementations, protocol settings and tests
  unchanged. Kept the working `.ven`, extracted datasets and SHD cache.

## Disk cleanup blocked by automatic review

Automatic approval review rejected the filesystem deletion command with
`blocked by policy`. No duplicate dataset or ZIP files were removed.

All six files under `data/SSC/` were verified by SHA-256 to match the files in
`data/SSCWrapper/`. The current `SSCWrapper` loader uses the latter folder.
All extracted SSC and SHD HDF5 files were opened successfully and their label
counts checked against spike timestamp counts.

To perform the verified disk cleanup manually from the project root:

```powershell
# Redundant SSC copy: retain data/SSCWrapper.
Remove-Item -LiteralPath '.\data\SSC' -Recurse -Force

# Download archives: retain their extracted .h5 files.
Remove-Item -LiteralPath '.\data\SSCWrapper\ssc_train.h5.zip', '.\data\SSCWrapper\ssc_valid.h5.zip', '.\data\SSCWrapper\ssc_test.h5.zip', '.\data\SHDWrapper\shd_train.h5.zip', '.\data\SHDWrapper\shd_test.h5.zip' -Force
```

These targets total approximately 7.23 GB. No model weights are included.
Deleting `data/cache/SHD` would recover another 3.32 GB, but it was intentionally
kept to avoid preprocessing costs on future experiments. Small Python caches
were also left in place after disk cleanup was rejected.

Keep best checkpoints for reference models and candidates awaiting test
evaluation. Keep `last.ckpt` for runs that may resume. Completed rejected
experiments can later lose verbose logs or weights only after their
measurements and checkpoint provenance have been retained. This cleanup did
not delete experiment evidence.
