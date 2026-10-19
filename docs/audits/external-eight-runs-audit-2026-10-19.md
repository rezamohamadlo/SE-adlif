# External eight-run inspection - 19 October 2026

Source: user-provided external transfer directory `F:/reza/SE-adlif`. Files were read in place; no import, relocation, training or test reevaluation was performed.

| Model | Dataset | Seed | Test accuracy (%) | Evidence |
|---|---|---:|---:|---|
| DTH_SE_adLIF | ECG | 123 | 87.81 | Transferred metrics |
| DTH_SE_adLIF | ECG | 42 | 87.81 | Transferred metrics |
| DTH_SE_adLIF | SSC | 123 | 78.31 | Existing local metrics + transferred checkpoints (metrics omitted from transfer) |
| DTH_SE_adLIF | SSC | 456 | 78.04 | Existing local metrics + transferred checkpoints (metrics omitted from transfer) |
| MT_SE_adLIF | ECG | 123 | 88.19 | Transferred metrics |
| MT_SE_adLIF | ECG | 456 | 87.98 | Transferred metrics |
| MT_SE_adLIF | SSC | 123 | 78.14 | Transferred metrics |
| MT_SE_adLIF | SSC | 456 | 77.46 | Transferred metrics |

All eight have best and last checkpoints with optimizer state. SSC last checkpoints reach epoch 39/step 11760; ECG last checkpoints reach epoch 399/step 3600. All saved configs disable early stopping. Transferred metric histories have one validation row per epoch and one final test row; no duplicated validation epochs were found. The two omitted DTH SSC CSVs already exist in this checkout and agree with the transferred selected checkpoints' best-validation scores.

All eight per-seed cost records have status complete and their selected-checkpoint SHA256 hashes match the transferred files. No costs were remeasured.

SSC uses LR 0.006, batch 256, 40 epochs and summed membrane-potential loss. These are evidence for the existing sum-loss protocol, not the new mean-loss diagnostic. DTH SSC seeds 123/456 supplement already reported runs and must not be counted as additional independent seeds. MT SSC seeds 123/456 supply completed records where local evidence was previously incomplete/absent.

ECG uses LR 0.01, batch 64, 400 epochs, 5% training-derived validation and split seed 42. These four are current full-budget proposed-model records, not the archived early-stopped runs. The existing local ECG reference remains unmatched in stopping/budget; importing these would not repair the reference. DTH ECG seed 456 and MT ECG seed 42 are not among these eight folders.

No original results or general comparison tables were overwritten: this is an inspection, not an import. A future import should merge complementary files with hash checks, preserve provenance, and update completion/aggregate reports without replacing historical scores based on test ranking.

Detailed machine-readable evidence: [audit JSON](../../results/provenance/external-eight-runs-audit-2026-10-19.json).

Subsequent action: the four ECG folders were moved into this checkout after user authorization; the four SSC folders remain in external storage. See the [completed transfer record](../transfers/ecg-transfer-2026-10-19.md). The inspection observations above describe the pre-transfer snapshot.

Subsequent SSC move completed: all four remaining SSC folders have now been moved into this checkout, with preexisting local evidence archived and omitted DTH metrics retained. No transferred seed folders remain at the external source. See the [SSC transfer record](../transfers/ssc-transfer-2026-10-19.md). Earlier descriptions of remaining external SSC folders are historical snapshots.
