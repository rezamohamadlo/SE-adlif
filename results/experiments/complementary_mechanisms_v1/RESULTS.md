> Superseded for discovery by `docs/experiments/complementary-mechanisms-study-v2.md`. These are preserved canonical random-split screens. No official test evaluation was performed. The150epochlaunch failed its source guard before training.

# Complementary mechanisms v1: cumulative findings

**Final success criterion: ordinary-inference test accuracy on unseen data. Current evidence is validation-only and does not establish improved test generalization.**

Study active; seed 42 discovery. Official tests have not been evaluated in this study. Test-timing clarification is pending.

| SHD* arm | Completed epochs | Best validation accuracy | Validation loss at selected checkpoint | Selected epoch index | Updates | Test accuracy |
|---|---:|---:|---:|---:|---:|---|
| SE_adLIF | 75/300 | 97.9767% | 0.103156 | 69 | 450 | pending |
| FIR_SE_adLIF | 75/300 | 98.0380% | 0.132105 | 63 | 450 | pending |

Best-validation A001 minus matched baseline: +0.0613 pp. n=1; across-seed SD unavailable. The earlier 30-epoch gain was +0.4292 pp and has shrunk. Selected loss is now worse for A, while late-curve average and stability favor A; this is inconclusive, not a generalization success.

Training and validation batches1024; LR.01 unchanged. Paperbatch256 versus1024 changes training updates/epoch25->6, so the old reference isnot the matchedcontrol. No test or paper numeric delta can be derived from these validation numbers.

All stage30/75 best+last snapshots/configs/receipts are preserved. Decision: EXTEND both matched arms to150 completed epochs from their own last checkpoints. Estimated additional training/validation time isabout24minutes total from measured epoch times; this isnot a guarantee. Official-test timing remains a separate pending choice.

SSC train/validation acquisition is still incomplete. Local complete files: SHD train/test and ECGtrain. ECG row indices are disjoint, patient identifiers unavailable; full original labels/masks retained. ECG throughput calibration uses disposable states and real training records; its weights never initialize experiments.

Fourarm ablation, transfer, replication, finaltest differences and uncertainty, paper comparisons, parameters/inferencecost and complete trainingcost remain pending.
