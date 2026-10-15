# Tracked experiment logs

Compact `metrics.csv` and saved `hparams.yaml` files under `results/valid/` are
tracked as research evidence. Checkpoints, TensorBoard files and verbose console
logs remain local. These files are snapshots, not a declaration that every run
completed. Consult run configs, test-result records and summaries for status.

The LG-DTH SHD* seed 123 metrics were captured during an incomplete run. Missing
final test results remain pending. The test log in `reports/lg-dth-test-results.txt`
is a previously recorded 52-test run, not a fresh verification of this commit.

SHD logs under the old dated `reference/SHD/seed_*/hydra/` paths are historical
controlled-protocol evidence. They do not describe the current paper-baseline
reference. Use `REFERENCE_RESULTS.json` and `REFERENCE_RESULT.json` in the canonical
reference directories for that reference, and preserve protocol distinctions.