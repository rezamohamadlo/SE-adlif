# Memory-safe temporal-permutation finalization — prepared 21 November 2026

The full SHD*/SSC run saved 8,280 inference chunks and passed all 36 clean
anchors, then failed in `json.dumps` while assembling the monolithic paired JSON.
The source directory contains approximately 7.8 GB of cached score JSON. Its
FAILED.json, original manifest, source files, pilot approval and cached predictions
are preserved. Do not rerun or edit the original launcher to repair this failure.

Preparation verification on 21 November: the read-only `--check-only` preflight
authenticated all 8,280 expected cache files, their original manifest/checkpoint/
batch/sample contracts, and their payload checksums. Static syntax and artifact
path checks passed, including resolution after changing the process working
directory. No repaired-output directory was created. Tests, aggregation and
scientific finalization were not executed by the agent; they remain user-run.

The separate `scripts/finalize_temporal_permutation.py` entrypoint imports no
model engine, opens no checkpoint or raw dataset, and performs no neural forward,
training, perturbation generation or new bootstrap. It authenticates the recorded
extension/scientific source identities and all cache contracts before reusing
saved predictions. It requires full source coverage, exact batch membership,
checkpoint identities in cache metadata, ordered sample IDs, finite scores and
payload checksums. Missing/corrupt inference files fail; there is no fallback
to inference or a different checkpoint. Actual weight/data files are not needed
for this saved-score finalization; their identities remain the original recorded
identities, not newly recomputed authentication of those external files.

The repair writes only to the new portable root
`results/experiments/adaptation_temporal_permutation_finalized_v1/`. The source
analysis lock and a separate output lock prevent conflicting writers. It saves
a new manifest binding all source cache/evidence file hashes and the repair's own
source/test hashes. Adding files outside the original package avoids changing
the frozen glob-based extension identity. A new COMPLETE.json is independent of
the original failed run; the latter never becomes retroactively complete.
The missing original LOADED_SOURCES.json is not fabricated. The new provenance
identifies which authenticated source contracts and saved outputs were consumed.

## Bounded memory and statistical equivalence

The finalizer holds at most one pair's original batch and three perturbation
replicates, then streams rows to compact UTF-8 JSONL. It never accumulates the
entire paired dataset or constructs a whole serialized output string. Small
per-training-seed vectors retain one value per sample/metric. Clean cached scores
are reaggregated to reproduce the recorded anchors before paired reporting.

Rows retain the original schema, predictions, transformation audits and
`benefit_clean`, `benefit_permuted` and `delta_benefit` definitions. Sharding and
row order differ from the failed export: each original sample's replicates are
adjacent. The summary keeps the original hierarchy exactly: mean over three
replicates within a sample/seed, mean over records, then mean and sample SD across
42/123/456. It retains negative/zero/ineffective outcomes and all four pairs.
No scientific selection or model change is introduced. Pure user-run contract
tests compare the new calculations with the original summary implementation.

Each paired batch has an authenticated `.meta.json` completion sidecar. Resume
checks identities and hashes; a missing data file with an existing sidecar fails.
An interrupted shard without a sidecar is unpublished and may be reconstructed
deterministically. Writes stream to a temporary file, fsync and atomically replace
the destination, with bounded Windows sharing-error retries. Old destinations
remain intact on failure; failed temporary files are retained.

Outputs are `paired/` JSONL shards, `PAIRED_RESULTS_INDEX.json` (paths, hashes and
row counts), `SUMMARY.json`, contract-test evidence, a new MANIFEST.json and
COMPLETE.json. There is intentionally no enormous `PAIRED_RESULTS.json` array.
Readers should iterate the paths in the index and parse one JSON object per line.
For SHD*/SSC, the expected paired-row count is 1,254,096. ECG remains outside this
executed source scope and is still scientifically blocked in the broader study.
JSONL/shard storage is still large: allow space for several GB of additional
paired output and temporary files; avoiding RAM exhaustion does not remove disk
requirements. No controlled hardware/runtime claim is made.

## One-PC commands from the repository root

Read-only preflight (no tests, aggregation, source writes or output creation):

```powershell
.\.venv\Scripts\python.exe -m scripts.finalize_temporal_permutation --check-only
```

User-run synthetic contracts, then CPU-only saved-score finalization (the normal
command also runs these contracts automatically):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_finalize_temporal_permutation
.\.venv\Scripts\python.exe -u -m scripts.finalize_temporal_permutation
```

Repeat the same command to resume, preserving the input/output roots and repair
source files. To authenticate the completed repaired report without aggregation:

```powershell
.\.venv\Scripts\python.exe -m scripts.finalize_temporal_permutation --verify-only
```

Explicit portable paths are supported with `--source-root` and `--output-root`.
They must be distinct direct children named `adaptation_temporal_permutation_*`
under `results/experiments/`. Resolution is anchored to project_paths.ROOT, not
the process/Hydra working directory. Keep the original failed source directory
unchanged, even after a successful finalization. Full completion remains pending
until actual user execution and review of the new completion ledger.
