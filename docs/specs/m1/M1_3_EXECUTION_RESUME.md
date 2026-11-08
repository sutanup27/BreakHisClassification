# M1.3: Development execution, resume and histories

Acceptance status: Not started. Training/resume primitives exist.
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

- [ ] Make development the runner default and route all pipeline reports to validation; the test path must never execute implicitly.
- [ ] Record actual fitting seed and deterministic settings before fresh model/loader creation; initialize a fresh optimizer/schedule for the declared phase.
- [ ] Preserve validation-only selection: group balanced accuracy, lower image validation loss, then earlier epoch; preserve documented legacy accuracy behavior.
- [ ] Persist and restore best score, tie-break loss and selected epoch with optimizer/scheduler/RNG state at committed epoch boundaries.
- [ ] Reconcile interrupted history/best-checkpoint publication with the last committed epoch; detect or repair documented recoverable states without silently accepting mismatches.
- [ ] Retain epoch-zero restart behavior, workers-zero exact resume and clear refusal of changed runtime/device/config/source or partial-epoch resume.
- [ ] Record the learning rate actually used, optimizer steps, training/validation seconds and one history row per committed epoch.
- [ ] Define expected outputs per stage, including histories/predictions, and validate hashes/completeness before skip/recovery decisions.
- [ ] Record child errors, Ctrl+C and write failures as non-success with useful logs; preserve completed outputs and permit only valid stage/epoch retries.
- [ ] Hand completed development artifacts to M1.4 for selection binding; do not claim final completion from development success.

## Verification checklist

- [ ] M1-T09: fresh matching tiny CPU runs reproduce model state; a changed seed is recorded and affects the initialization fixture.
- [ ] M1-T10: compare interrupted/resumed versus uninterrupted SGD/cosine and Adam/constant fits, including group-score ties and non-improving epochs; compare weights, optimizer, scheduler, RNG, history and chosen checkpoint.
- [ ] M1-T11: incompatible settings/runtime, missing best weights and partial-epoch files fail clearly rather than degrading exactness silently.
- [ ] M1-T12: the full default dense/pruning development path exports validation outputs only, with a failing test-adapter tripwire.
- [ ] M1-T13: interrupted child training, nonzero child exit and publication failures leave honest phase/stage states and recover only completed boundaries.
- [ ] M1-T14: duplicate/stale history and missing/corrupt secondary artifacts cannot be mistaken for a complete reusable stage.

Intended tests: extend `tests/test_resume.py` and experiment orchestration cases;
use small stochastic models for numerical resume checks and generated images
for real CLI integration. Avoid long real-data convergence runs in unit tests.

## Completion and handoff

Development outputs are complete and attributable; final evaluation has not run.
M1.4 can bind the selected checkpoint set and run inference without fitting.
Checkpoint selection by balanced accuracy is not a sensitivity safeguard.

## Progress evidence

Not executed against this contract: specification only. Existing exact-resume
tests provide useful regression coverage, but the phase contract, broader
optimizer/tie cases and artifact-complete recovery still require verification.
