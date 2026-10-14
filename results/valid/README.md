# Canonical reference and model results

- `reference/SHD/`: current paper-baseline SE-adLIF SHD* reference, seeds
  42/123/456; batch 256, 300 epochs, early stopping disabled. Validation is
  20% of training with split seed 42. See `reference/SHD/RESULTS.md`.
- `reference/SSC/`: existing official validation/test reference records.
- `reference/ECG/`: existing reference records with historically inconsistent
  early-stopping settings, explicitly documented in comparison reports.
- `protocol_v2/`: proposed-model controlled-v2 runs and resumable artifacts.
  A run is complete only after its completion marker and final test metrics.

The old SHD reference tree has been replaced, not retained as another active
reference. Prior sweeps keep numerical context in their REFERENCE_BASELINE.json.
Proposed-model controlled-v2 SHD runs used batch 512 and early stopping and
are unmatched against the new canonical paper-baseline SHD reference.

`runner.py` still loads frozen `evaluation.yaml` and writes into protocol_v2;
calling it does not automatically apply the paper baseline. New fair model
comparisons must explicitly inherit `config/paper_baseline.yaml` in separate
configs/output directories. Main runners and existing model results are unchanged.
