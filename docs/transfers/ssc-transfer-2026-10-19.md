# SSC transfer - 19 October 2026

All four remaining SSC run folders were moved from the user-provided external directory `F:/reza/SE-adlif` into the matching `results/valid/` model/seed locations. Every source file was verified by SHA256 at the destination before its external source folder was removed.

Existing local DTH SSC seeds 123/456 and MT SSC seed 123 were archived under `archive/ssc_pre_transfer_20261019/`. MT seed 456 had no previous local directory. DTH local-only metrics and hyperparameters omitted from the transfer were restored from the archive without modifying them. The old snapshots are not additional independent training seeds.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Mean +/- sample SD (%) |
|---|---:|---:|---:|---:|
| DTH_SE_adLIF | 78.22 | 78.31 | 78.04 | 78.19 +/- 0.14 |
| MT_SE_adLIF | 77.60 | 78.14 | 77.46 | 77.73 +/- 0.36 |

All six active SSC runs have 40 validation epochs and one final test record. Imported runs use summed membrane-potential loss; they are separate from the mean-loss diagnostic. No training was stopped or started by the transfer. The user stated that they would stop the diagnostic; its process status was not changed or inferred here. The loss hypothesis remains preliminary pending completed matched experiments and final evaluation.

Model accuracy/cost tables, canonical README and the general comparison were updated. [File manifest and archive inventory](../../results/provenance/ssc-transfer-2026-10-19.json).
