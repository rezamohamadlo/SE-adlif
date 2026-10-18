# Historical ECG runs excluded from active comparisons

Moved MT_SE_adLIF/ECG and DTH_SE_adLIF/ECG from results/valid/protocol_v2/
at the user's request on 2026-10-17. Each contains the historical seed-42 run.
These runs used early stopping; the current baseline requires 400 epochs with
early stopping disabled. Keep their metrics and checkpoints as historical
evidence, excluded from the active baseline comparison. ECG reference runs
were not moved.
