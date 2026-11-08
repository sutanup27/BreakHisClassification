# M1.4: Phase-aware evaluation, predictions and comparisons

Acceptance status: Not started. Existing metrics/export/timing code is reusable.
Depends on: M1.1-M1.3. Master: [M1_SPEC.md](M1_SPEC.md), sections 4.3-4.5 and 5.
Owns: M1-R11, M1-R12, M1-R13, M1-R14, M1-R15.
Tests: M1-T15, M1-T16, M1-T17, M1-T18, M1-T19, M1-T20, M1-T21.
Progress: [tracker](README.md).

## Goal and deliverables

Expose development results for research iteration and make final test inference
an explicit operation on fixed checkpoints. Targets: `breakhis/experiments.py`,
`breakhis/reporting.py`, low-level evaluate CLI, root comparison runner and
reporting/experiment tests. Do not add constrained search or threshold fitting.

## Implementation checklist

- [ ] Export validation reports/predictions for completed development and atomically publish the master-defined `development/selection.json` binding.
- [ ] Implement proposed `--phase final`: load the saved config/bindings, require completed development and verify selected checkpoint/config/manifest/policy digests.
- [ ] Perform test-only inference; never create training loaders, fit, prune, tune thresholds or choose a different epoch/model from test quality.
- [ ] Require only the evaluation device for final inference, allowing CPU evaluation of portable CUDA-trained weights; keep training-resume checks separate.
- [ ] Implement low-level `evaluate --split validation|test` routing; explicitly record phase/split and output schema version 2.
- [ ] Verify primary magnification-balanced and secondary image-mean grouping, fixed 0.5/tie-to-benign policy, units and denominators; represent undefined reusable-helper rates honestly.
- [ ] Export complete sequential image/group predictions, including subtype, filename/component identity, class order and config/manifest/checkpoint attribution.
- [ ] Protect prediction/report destinations from aliasing input weights, configs, manifests or other required outputs; retain atomic completion behavior on failure.
- [ ] Benchmark a reproducible recorded synthetic resident input with scoped CPU/synchronized GPU settings; retain negative/zero efficiency findings.
- [ ] Compare saved outputs for the requested phase only. Validate all referenced artifacts and measurement invariants; never run inference during comparison.
- [ ] Report deliberate model/seed/epoch/pruning differences and measured budgets. Add proposed explicit declaration for differing selection criteria; do not pretend different factors are equal or that epoch disclosure proves fairness.
- [ ] Keep development results usable if final fails; retries cannot overwrite complete final outputs or trigger fitting.

## Verification checklist

- [ ] M1-T15: missing development or changed selection/checkpoint/config bindings refuse before test iteration.
- [ ] M1-T16: CPU final succeeds from CUDA-origin saved weights with fitting/pruning/optimizer tripwires; original training GPU is not required.
- [ ] M1-T17: hand-calculated unequal group sizes/magnification counts, ties and missing support produce the declared metrics and units.
- [ ] M1-T18: ordered uneven batches preserve every identity/probability; shuffle/drop/duplicate/missing rows and unsafe output aliases fail.
- [ ] M1-T19: missing/corrupt reports, mixed phases, split differences or incompatible timing cannot produce a misleading matched comparison; no model is loaded.
- [ ] M1-T20: supported deliberate differences are listed; selection changes need explicit declaration and immutable data/measurement invariants cannot be overridden.
- [ ] M1-T21: record proper timing samples/runtime on CPU and conditional GPU checks; do not infer speedup from sparsity, noise or failed measurements.

Intended tests: extend `test_reporting.py`, `test_experiments.py` and phase-aware
CLI integration. Heavy uncertainty analysis and timing-session replication belong
to milestone 5 rather than this package.

## Completion and handoff

Both phases produce valid attributed reports, and comparisons preserve phase
boundaries and controlled settings. M1.5 can test the complete public workflow.
Saved validation predictions support later vulnerability analysis; they do not
constitute the analysis or a group-balanced pruning algorithm themselves.

## Progress evidence

Not executed against this contract: specification only. Automatic current test
evaluation, test-only low-level reporting and incomplete prediction attribution
were identified by source inspection; existing report tests are baseline evidence.
