# Measurements live with their training runs

Each seed's test-set activity and computation measurements are stored beside
its training summary:

```text
results/not_improved/DA_SE_adLIF/SHD/seed_42/
    TRAINING_SUMMARY.md
    FULL_RUN_CONFIG.json
    COST_RESULTS.json
    COST_RESULTS.md
    logs/
    ckpt/
```

Reference measurements use `results/valid/reference/<dataset>/seed_<seed>/`.
MR (K=2) uses `results/not_improved/MR_SE_adLIF/K_2/<dataset>/seed_<seed>/`.
The model's aggregate accuracy/cost report remains `RESULTS.md` above its dataset
directories. The reference tree retains its existing per-dataset result layout.

`COST_RESULTS.json` is the original measurement evidence, including checkpoint
identity, environment, counts, timing repeats and evaluator version. Earlier
`shd_costs_v1` and `reference_test_costs_v1` records keep their original schemas
and values. `cost_records.py` reads those schemas without rewriting the files;
arithmetic/logical totals missing from older records are derived from their
retained operation breakdown using the existing convention. No model was rerun
to reorganize files.

`COST_RESULTS.md` is the human-readable summary. New runner jobs already save
both files at this location. Explicit custom evaluator `--output` directories
remain available for separate experiments; the default evaluators now save
inside the canonical seed directory.

Earlier preflight measurements are retained inside affected seeds as
`measurement_history/shd_costs_v1_preflight.json`. They are not additional
independent seeds and do not replace the accepted `COST_RESULTS.json`.

Shared evaluator snapshots, original cross-model aggregate/CSV exports and
method notes are frozen in [measurement provenance](../../archive/measurement_provenance/README.md).
That archive's `RELOCATION.json` maps every moved file to its SHA-256-verified
destination. The former top-level `results/measurements/` directory is retired.

To refresh the reference cost tables from saved canonical records only:

```powershell
.\.venv\Scripts\python.exe measure_reference_costs.py --reports-only
```

New runner jobs collect costs after final test evaluation. For a selected
completed job, `measure_run_costs.py --run-dir <seed-directory> --dataset <dataset>`
reuses a matching saved record, including recognized historical schemas. It
does not replace existing evidence with a different checkpoint or rerun it
merely because its JSON originally came from a different evaluator version.
