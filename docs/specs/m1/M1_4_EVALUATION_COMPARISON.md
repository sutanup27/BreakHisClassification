# M1.4: Phase-aware evaluation, predictions and comparisons

Acceptance status: Reviewed. Bound final inference and saved phase comparisons implemented.
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

- [x] Export validation reports/predictions for completed development and atomically publish the master-defined `development/selection.json` binding.
- [x] Implement proposed `--phase final`: load the saved config/bindings, require completed development and verify selected checkpoint/config/manifest/policy digests.
- [x] Perform test-only inference; never create training loaders, fit, prune, tune thresholds or choose a different epoch/model from test quality.
- [x] Require only the evaluation device for final inference, allowing CPU evaluation of portable CUDA-trained weights; keep training-resume checks separate.
- [x] Implement low-level `evaluate --split validation|test` routing; explicitly record phase/split and output schema version 2.
- [x] Verify primary magnification-balanced and secondary image-mean grouping, fixed 0.5/tie-to-benign policy, units and denominators; represent undefined reusable-helper rates honestly.
- [x] Export complete sequential image/group predictions, including subtype, filename/component identity, class order and config/manifest/checkpoint attribution.
- [x] Protect prediction/report destinations from aliasing input weights, configs, manifests or other required outputs; retain atomic completion behavior on failure.
- [x] Benchmark a reproducible recorded synthetic resident input with scoped CPU/synchronized GPU settings; retain negative/zero efficiency findings.
- [x] Compare saved outputs for the requested phase only. Validate all referenced artifacts and measurement invariants; never run inference during comparison.
- [x] Report deliberate model/seed/epoch/pruning differences and measured budgets. Add proposed explicit declaration for differing selection criteria; do not pretend different factors are equal or that epoch disclosure proves fairness.
- [x] Keep development results usable if final fails; retries cannot overwrite complete final outputs or trigger fitting.

## Verification checklist

- [x] M1-T15: missing development or changed selection/checkpoint/config bindings refuse before test iteration.
- [x] M1-T16: CPU final succeeds from CUDA-origin saved weights with fitting/pruning/optimizer tripwires; original training GPU is not required.
- [x] M1-T17: hand-calculated unequal group sizes/magnification counts, ties and missing support produce the declared metrics and units.
- [x] M1-T18: ordered uneven batches preserve every identity/probability; shuffle/drop/duplicate/missing rows and unsafe output aliases fail.
- [x] M1-T19: missing/corrupt reports, mixed phases, split differences or incompatible timing cannot produce a misleading matched comparison; no model is loaded.
- [x] M1-T20: supported deliberate differences are listed; selection changes need explicit declaration and immutable data/measurement invariants cannot be overridden.
- [x] M1-T21: record proper timing samples/runtime on CPU and conditional GPU checks; do not infer speedup from sparsity, noise or failed measurements.

Intended tests: extend `test_reporting.py`, `test_experiments.py` and phase-aware
CLI integration. Heavy uncertainty analysis and timing-session replication belong
to milestone 5 rather than this package.

## Completion and handoff

Both phases produce valid attributed reports, and comparisons preserve phase
boundaries and controlled settings. M1.5 can test the complete public workflow.
Saved validation predictions support later vulnerability analysis; they do not
constitute the analysis or a group-balanced pruning algorithm themselves.

## Progress evidence

Implemented final inference in `breakhis/final_evaluation.py`, read-only artifact
validation in `breakhis/phase_artifacts.py` and saved comparisons in
`breakhis/saved_comparison.py`. Existing CLI/reporting primitives are reused.

Verification:

```powershell
.\.venv-stage2-cpu\Scripts\python.exe -m unittest discover -s tests
.\.venv313\Scripts\python.exe -m unittest discover -s tests -p test_m1_evaluation.py
.\.venv-stage2-cpu\Scripts\python.exe tools/smoke_experiments.py --work-dir artifacts/m1-4-phase-smoke-final
```

- CPU suite: 131 tests, 129 passed and 2 CUDA-only skips, in 75.996 seconds.
  Log: `artifacts/m1-4-tests.log`.
- CUDA environment (PyTorch 2.11.0+cu128): all 23 M1.4 tests passed in 24.391
  seconds, including synchronized CUDA timing and an actual CUDA fit followed by
  CPU final inference with CUDA availability hidden.
  Log: `artifacts/m1-4-cuda-tests.log`.
- Fresh-process real VGG dense and channel-pruning/fine-tuning workflows passed
  on 12 generated images. Matching-seed dense weights reproduced; both explicit
  final phases and development/final JSON/CSV comparisons succeeded.
  Evidence: `artifacts/m1-4-phase-smoke-final/summary.json` and
  `artifacts/m1-4-smoke-final.log`. These are implementation checks, not research
  accuracy or statistically significant latency findings.

M1-T15: missing development, drifted config/manifest/selection/weights and changed
policy bindings refuse before test iteration. M1-T16: final uses test-only
loaders, all bound roles and the designated recipe; fitting, pruning and optimizer
tripwires remain untouched. Failed child/output publication preserves development
and retries only incomplete final work. Completed final phases cannot overwrite.

M1-T17: hand-calculated group/magnification imbalance, ties and missing support
are covered. Undefined rates are null with reasons; units/support are explicit.
M1-T18: image rows preserve subtype, raw/pixel hashes, filename/component identity
and class-ordered probabilities; group records retain component attribution.
Shuffle, drop-last, duplicate/missing/reordered coverage and input/config aliases
fail. Schema-2 report/prediction envelopes bind runtime, phase, split and digests.

M1-T19: comparisons do no inference/model loading and verify all requested-phase
artifacts. Missing/corrupt/mixed-phase outputs, different split hashes, old schema-1
runs and mismatched benchmarks refuse. With two completed runs, additional
incomplete runs are separate diagnostics. JSON is the completion record and is
published last; an injected JSON failure removes its incomplete CSV sibling.
Development remains comparable after final failure.

M1-T20: seed/epoch/pruning factors and measured budgets are enumerated; deliberate
selection changes require `--vary training.selection_metric`. Data/measurement
invariants cannot be overridden. M1-T21: protocol
`resident-forward-synthetic-v1` uses a local seed-zero CPU RNG, records input
shape/dtype/hash and timing samples, preserves global input-generation RNG state,
and synchronizes CUDA. Negative efficiency findings are retained.

M1.5 remains the owning package for complete public workflow/integration
acceptance. No vulnerability analysis, new pruning mechanism, threshold fitting,
uncertainty analysis or converged research experiment is claimed here.
