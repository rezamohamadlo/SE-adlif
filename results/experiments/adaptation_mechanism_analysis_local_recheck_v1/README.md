# SHD* validation pilot: portable analysis package

The 256-sample cohort and training seeds 42, 123, 456 are unchanged.

- `clean/`: original local clean evaluation, including its matching metadata and detailed rows.
- `time_scaling/`: locally regenerated timing evaluation.
- `spike_deletion/`: locally regenerated deletion evaluation.
- `stage3_review/`, `stage4_review/`: historical audit records, preserved byte for byte.

`ARTIFACT_INDEX.json` maps historical recorded paths to the current layout. Original PC labels inside immutable provenance identify historical analysis assignments, not required hardware. Historical review paths must also be resolved through this index. No result values, manifests, completion identities or original evaluation files were rewritten.

Share this ENTIRE directory, including ignored `diagnostic_rows/`, using the same repository-relative layout. Git alone omits the detailed rows. The archive backup is no longer needed to verify this package.

Verify from the checkout root:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_analysis/evaluate.py --verify-only --output-root results/experiments/adaptation_mechanism_analysis_local_recheck_v1
```
