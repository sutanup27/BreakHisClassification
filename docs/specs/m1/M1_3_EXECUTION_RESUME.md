# M1.3: Development execution, resume and histories

Acceptance status: Reviewed. Development execution, committed-epoch recovery and histories implemented.
Depends on: M1.1 and M1.2. Master: [M1_SPEC.md](M1_SPEC.md), sections 4.2-4.3 and 5.
Owns: M1-R07, M1-R08, M1-R09, M1-R10.
Tests: M1-T09, M1-T10, M1-T11, M1-T12, M1-T13, M1-T14.
Progress: [tracker](README.md).

## Goal and deliverables

Complete a reproducible development phase with no automatic final evaluation.
Targets: `breakhis/experiments.py`, `breakhis/cli.py`,
`breakhis/checkpoints.py`, training history handling and resume/integration tests.
Keep existing train/prune/fine-tune primitives and fixed pruning recipes.

## Implementation checklist

- [x] Make development the runner default and route all pipeline reports to validation; the test path must never execute implicitly.
- [x] Record actual fitting seed and deterministic settings before fresh model/loader creation; initialize a fresh optimizer/schedule for the declared phase.
- [x] Preserve validation-only selection: group balanced accuracy, lower image validation loss, then earlier epoch; preserve documented legacy accuracy behavior.
- [x] Persist and restore best score, tie-break loss and selected epoch with optimizer/scheduler/RNG state at committed epoch boundaries.
- [x] Reconcile interrupted history/best-checkpoint publication with the last committed epoch; detect or repair documented recoverable states without silently accepting mismatches.
- [x] Retain epoch-zero restart behavior, workers-zero exact resume and clear refusal of changed runtime/device/config/source or partial-epoch resume.
- [x] Record the learning rate actually used, optimizer steps, training/validation seconds and one history row per committed epoch.
- [x] Define expected outputs per stage, including histories/predictions, and validate hashes/completeness before skip/recovery decisions.
- [x] Record child errors, Ctrl+C and write failures as non-success with useful logs; preserve completed outputs and permit only valid stage/epoch retries.
- [x] Hand completed development artifacts to M1.4 for selection binding; do not claim final completion from development success.

## Verification checklist

- [x] M1-T09: fresh matching tiny CPU runs reproduce model state; a changed seed is recorded and affects the initialization fixture.
- [x] M1-T10: compare interrupted/resumed versus uninterrupted SGD/cosine and Adam/constant fits, including group-score ties and non-improving epochs; compare weights, optimizer, scheduler, RNG, history and chosen checkpoint.
- [x] M1-T11: incompatible settings/runtime, missing best weights and partial-epoch files fail clearly rather than degrading exactness silently.
- [x] M1-T12: the full default dense/pruning development path exports validation outputs only, with a failing test-adapter tripwire.
- [x] M1-T13: interrupted child training, nonzero child exit and publication failures leave honest phase/stage states and recover only completed boundaries.
- [x] M1-T14: duplicate/stale history and missing/corrupt secondary artifacts cannot be mistaken for a complete reusable stage.

Intended tests: extend `tests/test_resume.py` and experiment orchestration cases;
use small stochastic models for numerical resume checks and generated images
for real CLI integration. Avoid long real-data convergence runs in unit tests.

## Completion and handoff

Development outputs are complete and attributable; final evaluation has not run.
M1.4 can bind the selected checkpoint set and run inference without fitting.
Checkpoint selection by balanced accuracy is not a sensitivity safeguard.

## Progress evidence

Implemented in `breakhis/development.py`, the existing CLI/checkpoint primitives,
and `breakhis/fit_boundary.py`. The public runner now defaults to schema-2
validation-only development; schema-1 runs are not silently migrated/resumed.

Verification commands (CPU environment `.venv-stage2-cpu`):

```powershell
python -m unittest discover -s tests
python tools/smoke_experiments.py --work-dir artifacts/m1-3-development-smoke-final
```

- 108 CPU tests passed, including 16 focused cases in `test_m1_execution.py`.
  Full log: `artifacts/m1-3-tests.log`.
- M1-T09: actual small stochastic fits reproduce with matching seeds; changing
  the fitting seed changes weights and is recorded. Deterministic PyTorch/cuDNN
  and RNG settings are saved; cross-hardware equality is not claimed.
- M1-T10: SGD/cosine and Adam/constant interrupted fits match uninterrupted
  weights, optimizer/scheduler/RNG, numerical history and selected weights.
  Forced group-score ties and non-improving epochs retain epoch 2; legacy equal
  accuracy retains the earlier epoch. Wall-clock durations are excluded from
  numerical equality. Actual learning rates and optimizer steps are checked.
- M1-T11: epoch-zero restarts match fresh fits. Partial weights, missing best
  checkpoints and changed config/source/runtime/device/thread settings refuse.
  Completed-boundary stage recovery also checks the frozen identity/runtime.
- M1-T12: a dense/prune/fine-tune development test removes test images and uses
  a failing test-loader tripwire. All reports are validation; final remains
  `not_requested`. Fresh-process VGG dense/channel workflows also pass and
  reproduce dense weights. Evidence: `artifacts/m1-3-development-smoke-final/summary.json`
  and `artifacts/m1-3-smoke-final.log`. Generated-image scores are not research results.
- M1-T13: child nonzero exits, Ctrl+C, interrupted fitting and publication failures
  leave failed/interrupted states. Retries recover only committed boundaries
  or verified completed stages. Before/after-commit failures are injected.
- M1-T14: duplicate/stale/missing/corrupt history, missing predictions and missing
  config sidecars cannot be skipped as completed. Every required stage output is
  referenced by path, byte count and SHA-256.

Epoch publication uses one small pending journal. `*-last.pth` commits history,
best score/loss/epoch and a best-weight snapshot alongside optimizer/scheduler/RNG.
Resume can finish or roll back a journaled best/history publication. Without a
journal, inconsistent outputs fail explicitly. Do not remove the journal manually.
Old checkpoints without this boundary contract remain inference/fine-tuning inputs,
but require a fresh run for the new exact-resume contract.

Completed development publishes selection bindings and validation results. Final
execution and phase-specific comparisons remain M1.4; no final inference runs
implicitly. Milestone 1 as a whole is not complete.
