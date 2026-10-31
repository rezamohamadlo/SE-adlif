# VS-MT versus MT on SSC

Ordinary inference without TTA; validation-selected final-test checkpoints.
Matched baseline configuration; the intended deviation is one independent slow voltage coefficient per hidden neuron.
Previous test results have been observed. This is an experimental hypothesis; no guaranteed improvement or novelty claim.

| Model | Seed | Test accuracy (%) | Forward ms/sample | Hidden spikes/sample | Arithmetic GFLOPs/sample |
|---|---:|---:|---:|---:|---:|
| MT | 42 | 80.01 | 1.5629214601162489 | 48897.197380041216 | 1.0221899927936415 |
| MT | 123 | 79.98 | 1.8479705769769752 | 50055.69217937396 | 1.0221899927936415 |
| MT | 456 | 80.23 | 1.6856725967353214 | 49547.2552742616 | 1.0221899927936415 |
| VS-MT | 42 | pending | unavailable | unavailable | unavailable |
| VS-MT | 123 | pending | unavailable | unavailable | unavailable |
| VS-MT | 456 | pending | unavailable | unavailable | unavailable |

| Model | Completed n | Test mean +/- sample SD (%) |
|---|---:|---:|
| MT | 3 | 80.07 +/- 0.14 |
| VS-MT | 0 | pending |

Paired VS-MT minus MT (pp), n=0: pending.

Costs are available only where recorded; unavailable/failed costs never imply failed training or require retraining.
Timing was collected in different sessions and may use different environments; it is descriptive, not a controlled speedup or energy claim.
Per-neuron voltage sensitivities, shared spike coefficient, mixture weights, weighted voltage strengths, q and time constants are retained in COMPARISON.json; VS-MT also saves NEURON_PARAMETERS.json beside each completed seed.
