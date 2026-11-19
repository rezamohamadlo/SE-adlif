# Fixed DTH + R-Drop transfer check: canonical SHD*

The user requested stopping SSC and testing the same mechanism on SHD, with
a two-dataset official-test table. Existing study authorization covers this
bounded fresh training and exploratory testing. Do not restart SSC or its
cancelled full40/test queue.

Use the existing `runner.py` workflow and the separate SHD study definition.
Train fresh DTH and DTH + R-Drop, seed42, for exactly30 epochs each, retaining
the original300epoch optimization schedule. This is a short transfer screen,
not full-budget paper reproduction. No automatic extension is authorized.

Both arms inherit the SHD section of `config/paper_baseline.yaml`: batch256,
LR0.01, dropout0.15, two360neuron DTH hidden layers, original preprocessing,
ordinary sum-softmax-over-time loss, no early stopping, no event augmentation.
R-Drop retains the SSC mechanism: alpha1, temperature1, epsilon0. Only its
explicit training objective differs. Validation/inference use one forward and
hard-label CE. Record actual fresh initial weights and RNG identities.

Use only official SHD training data for fitting/selection: random20percent
validation with splitseed42, 6525 fitting and1631 validation samples. Freeze
indices before training and verify disjoint rows and full class coverage.
This is canonical SHD*, not speaker-disjoint development; fitting and
validation speakers overlap. Never reuse SSC or speaker-holdout weights.

At completion freeze the validation-selected best checkpoint of each arm and
their hashes before reading official test contents. Evaluate ordinary single
pass, no TTA, on all2264 official SHD test samples. Keep this one-seed adaptive
study exploratory because prior project tests have been exposed. Do not use
test scores to launch another idea or adjust settings.

The final table must separate datasets, protocol and completed budgets. Reuse
only the identical SSC validation-selected checkpoint's verified test record
after checking current selection and weight hash. SSC was interrupted before
full40 and its DTH-only control is10epochs; do not call that paired difference
budget-matched. SHD's two new arms share30epochs. Compare descriptively to
published SE-adLIF means SHD*93.79+/-0.76 and SSC80.44+/-0.26; add signed local
minus paper deltas and state single-seed versus multi-run limitations.

Preserve all interrupted SSC best/last weights, optimizer/scheduler, configs,
logs and test archives. Record the exact completed SSC epoch at stopping.
Save transfer configs, split audit, run receipts, test predictions and the
cross-dataset report under the separate SHD study results directory.
