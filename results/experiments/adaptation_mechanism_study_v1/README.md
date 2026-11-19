# Adaptation mechanism study: cross-pair review

This directory holds consolidation of saved results and prerequisite metadata,
not newly executed neural or statistical experiments.

- [SHD_SUMMARY.md](SHD_SUMMARY.md): all four SHD* pairs, saved per-seed values,
  aggregates, feature/intervention interval summaries and limitations.
- [SHD_SOURCES.json](SHD_SOURCES.json): source completion identities, verified
  output counts and copied unrounded summary values; parent reuse is explicit.
- [SSC_PREFLIGHT.json](SSC_PREFLIGHT.json): validation-file identity and metadata,
  eighteen checkpoint identities, original metric hashes and resolved configs.
- [SSC audit](../../../docs/audits/ssc-mechanism-readiness-2026-11-12.md): scope,
  readiness limits and preparation requirements for the next user-launched pilot.

These files do not replace the source packages. For detailed evidence, share
the source directories listed in `SHD_SOURCES.json`, including their ignored
`diagnostic_rows/` and model/seed provenance. A Git pull does not transfer those
chunks, datasets or checkpoints. Statistical specs were exploratory after
descriptive validation inspection, not independent confirmation. No additional
training seeds or independence is created by this consolidation.
