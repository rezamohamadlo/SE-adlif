# Canonical result policy

## SHD

The original SHD SE-adLIF runs are valid reference results. Their old
`loss_agg: softmax` mode computed `sum_t softmax(y_t)`, which is mathematically
identical to the renamed `sum_softmax_over_time` mode.

Canonical SHD statistics use exactly these three runs:

```text
reference/SHD/seed_42
reference/SHD/seed_123
reference/SHD/seed_456
```

No archive or verification-repeat directory is currently retained. If a repeat
of an existing seed is created later, label it as verification-only and exclude
it from the canonical mean, standard deviation, and run count.

## SSC

The earlier SSC runs are a separate case: they used temporal mean aggregation
instead of the paper's temporal sum of membrane-potential logits. They are not
valid for exact reference-paper reproduction. Corrected SSC runs must use
`loss_agg: summed_membrane_potentials`.

The corrected canonical SSC result set contains exactly these runs:

```text
corrected/reference/SSC/seed_42
corrected/reference/SSC/seed_123
corrected/reference/SSC/seed_456
```

The older `reference/SSC` directory is excluded from exact-reproduction
statistics because it used temporal mean aggregation.
