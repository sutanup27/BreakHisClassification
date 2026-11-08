# M2.4: Cost pilot and real baseline execution

Implementation/experiment status: Not started. Depends on: M2.1-M2.3.
Master: [M2_SPEC.md](M2_SPEC.md), sections 4.1-4.4, 5-6 and 9.
Owns: M2-R10, M2-R11, M2-R12. Acceptance: M2-T14-M2-T18.
Progress: [tracker](README.md).

## Goal and deliverables

Finish the proposed public study executor, measure actual cost and run the required
within-BreaKHis matrix. Targets: baseline-study adapter/modules, generated configs,
profiling/cost records and execution tests. Reuse M1 child commands and publication/
resume primitives instead of building a second training loop.

## Implementation checklist

- [ ] Implement proposed prepare/pilot/parents/candidates/report stage routing and deterministic dependency order; no stage defaults to final test inference.
- [ ] Exercise all stages with generated fixtures and genuine child/retry failures. Implement M2.5 report/integration code with fixtures before long scientific fits, then freeze current code/config identities.
- [ ] Profile 40X and a bounded pooled pass: actual device/epoch wall time, optimizer steps, image count, CPU timing and defined GPU allocated/reserved peaks. Record unavailable CPU RSS as null/reason.
- [ ] Estimate all planned family/recipe/recovery costs from measured values; use worst observed or clearly labeled conservative estimates. Include reporting, preparation and replayed work; do not multiply VGG epoch time as if every model cost were identical.
- [ ] Resolve available wall hours, max jobs and usable batch size; freeze study revision and chosen per-family B/F/recipe/profile. Refuse main launch if the declared capacity cannot cover the required matrix.
- [ ] Fit primary-seed VGG/ResNet18/compact references, inspect development-only convergence/absolute rates, and use only the bounded second-recipe opportunity if needed. Freeze chosen parents before candidate comparisons.
- [ ] Execute L1/random/mask grid, scratch widths and dense continuation; then the fixed repeat-seed VGG/control subset. Keep parent work shared, all unfavorable/failed ratios visible, and data/split/seed identities unchanged.
- [ ] Record incremental versus full-lifecycle steps, image presentations, selected/executed epochs, train/validation/pruning/report time, failed/replayed work and total study spend.
- [ ] Keep all checkpoints in the parent task directory and save ordinary experiment/study outputs. Retry only valid stage/epoch boundaries; changes require a new revision/ID.

## Verification checklist

- [ ] M2-T14: public generated-image study runs all stages with test factory/pixel tripwires; report stage performs no fitting/inference.
- [ ] M2-T15: child nonzero/OOM simulation, interruption and publication failure record non-success and valid retries; no refit/mutation of completed shared parents.
- [ ] M2-T16: actual 40X/pooled pilot records timing/memory, support and device/profile identity; CUDA measurement is synchronized and separately labeled.
- [ ] M2-T17: real required primary/repeat cells and controls complete, or scientific package status remains incomplete with reasons. No hidden grid/seed reduction or test feedback.
- [ ] M2-T18: hand-counted accounting fixture matches parent deduplication, replayed steps, image presentations and measured incremental/lifecycle spend; scratch discovery work remains visible.

## Completion and handoff

Software tests alone can mark executor implementation tested, not this package's
real experiment acceptance. M2.5 receives complete saved development outputs,
trained parents, histories, costs and failed/infeasible diagnostics for quality
review and frontier construction. No convergence or completion is promised from
the initial 20+5 defaults; replan explicitly if the measured recipe is inadequate.

## Progress evidence

Not executed: specification only. No new profile, cost estimate, fit or hardware
capacity measurement was performed when writing these documents.
