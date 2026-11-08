# M2.4: Cost pilot and real baseline execution

Implementation status: Tested. Real experiment status: In progress (six pilots complete; main fitting running). Depends on: M2.1-M2.3.
Master: [M2_SPEC.md](M2_SPEC.md), sections 4.1-4.4, 5-6 and 9.
Owns: M2-R10, M2-R11, M2-R12. Acceptance: M2-T14-M2-T18.
Progress: [tracker](README.md).

## Goal and deliverables

Finish the proposed public study executor, measure actual cost and run the required
within-BreaKHis matrix. Targets: baseline-study adapter/modules, generated configs,
profiling/cost records and execution tests. Reuse M1 child commands and publication/
resume primitives instead of building a second training loop.

## Implementation checklist

- [x] Implement proposed prepare/pilot/parents/candidates/report stage routing and deterministic dependency order; no stage defaults to final test inference.
- [ ] Exercise all stages with generated fixtures and genuine child/retry failures. Implement M2.5 report/integration code with fixtures before long scientific fits, then freeze current code/config identities.
- [x] Profile 40X and a bounded pooled pass: actual device/epoch wall time, optimizer steps, image count, CPU timing and defined GPU allocated/reserved peaks. Record unavailable CPU RSS as null/reason.
- [x] Estimate all planned family/recipe/recovery costs from measured values; use worst observed or clearly labeled conservative estimates. Include reporting, preparation and replayed work; do not multiply VGG epoch time as if every model cost were identical.
- [ ] Resolve available wall hours, max jobs and usable batch size; freeze study revision and chosen per-family B/F/recipe/profile. Refuse main launch if the declared capacity cannot cover the required matrix.
- [ ] Fit primary-seed VGG/ResNet18/compact references, inspect development-only convergence/absolute rates, and use only the bounded second-recipe opportunity if needed. Freeze chosen parents before candidate comparisons.
- [ ] Execute L1/random/mask grid, scratch widths and dense continuation; then the fixed repeat-seed VGG/control subset. Keep parent work shared, all unfavorable/failed ratios visible, and data/split/seed identities unchanged.
- [x] Record incremental versus full-lifecycle steps, image presentations, selected/executed epochs, train/validation/pruning/report time, failed/replayed work and total study spend.
- [x] Keep all checkpoints in the parent task directory and save ordinary experiment/study outputs. Retry only valid stage/epoch boundaries; changes require a new revision/ID.

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

Implemented `study_execution.py`, public stage routing, process lease, resolved
config freezing, deterministic parent/recipe/topology bindings, selected-parent
reuse, child committed-boundary retry and development-only JSON/CSV reports.
`tools/run_study_session.py` carries one continuous session across stages, runs
one job at a time, honors an `execution/PAUSE` request at a job boundary, refreshes
the results snapshot and records live status. No stage performs final test inference.

Software evidence:

- Full CPU regression: 206 cases in 323.295s, 200 passed/six CUDA-only skips;
  `artifacts/m2-4-regression.log`.
- Five focused executor cases passed on CPU (48.626s) and CUDA-enabled runtime
  (49.247s): all stages with generated fixtures/test pixels removed, report
  model/fitting tripwires, child failure, interruption/replay, completed-job reuse,
  capacity/session stops and tampering. Logs `artifacts/m2-4-executor-tests.log`
  and `artifacts/m2-4-cuda-executor-tests.log`.
- Numerical fitting/recovery regression: 16 passed (8.685s),
  `artifacts/m2-4-fit-tests.log`.
- Additional saved-only hand-counted frontier/tie/trade-off/failure and shared-cost
  case passed, `artifacts/m2-4-report-tests.log`. This is descriptive software;
  M2.5's full matched-comparison and baseline-strength review remain pending.

Actual study: `configs/baselines/m2_real_study.json`, ID
`m2-real-resnet-20261108`, binary v1, 40X/pooled pilots, GPU fitting batch 32,
B=20/F=5 (compact/scratch B+F), seeds 102/103, random ranking seed 202.
RTX 5070 Ti has 15.92 GiB total memory. User allows eight continuous hours then
additional sessions after rest. Provisional total study allocation is 24 hours
(three sessions), max 50 prospective fits; no main launch without measured capacity.
This does not assert all work will fit in that time.

The real six-cell pilot completed. Main pooled VGG parent fitting is now running. Logs `artifacts/m2-4-real-pilot.log`;
ordinary study outputs `artifacts/baselines/m2-real-resnet-20261108/` include
`execution/ledger.json`, CUDA profiles, measured cost inputs, session/live status
and `development/results.json`/CSV. Per-fit `*.resources.json` beside parent-folder
checkpoints records CUDA allocator peaks, actual callback steps/image presentations,
partial/replayed work and null CPU RSS with reason. Outer job wall includes prune,
report and initialization time. Missing hard-kill work is explicitly unknown.

The active worker passed measured capacity and is running the bounded
second recipes, freeze selected parents and execute controls. Its status/logs are
`execution/live_status.json`, `artifacts/m2-4-study-worker-r2.log` and
`artifacts/m2-4-study-worker-r2-error.log`. Quality/control completeness must be
reviewed after fitting; no empirical M2.4 completion or research novelty is claimed.


### Completed real pilot and active main fitting

All six one-epoch real pilots and their CPU/CUDA profiles completed. These are
validation-only cost/learning checks, not final trained-baseline or test results.
40X support: 1,326 fitting images/57 groups, 332 validation images/13 groups.
Pooled support: 5,233 fitting images/57 groups, 1,322 validation images/13 groups.

| Model | 40X image accuracy | Pooled image accuracy | 40X / pooled group balanced accuracy |
| --- | ---: | ---: | ---: |
| VGG | 79.22% | 85.78% | 87.50% / 100.00% |
| ResNet18 | 75.90% | 74.74% | 62.50% / 76.39% |
| MobileNetV3-Small | 63.55% | 62.25% | 50.00% / 50.00% |

Worst observed GPU fit epoch wall: pooled VGG 62.1s training + 9.9s validation;
ResNet18 13.8s + 2.8s; MobileNet 14.0s + 2.7s. Allocated GPU peaks were about
6.56/0.92/0.58 GiB respectively. CPU reporting is a major cost and is included in
conservative fixed overhead; no family inherits VGG throughput as measurement.

The measured maximum-plan estimate is approximately 18.16h, including bounded
optional recipes, pilots, separate CUDA profiles and preparation. Initial provisional
16h capacity correctly refused main launch. Before any main job started,
`tools/revise_study_capacity.py` recorded an explicit resource-only revision to
24h (three eight-hour sessions), consistent with the user's multiple-session
allowance. It recompiled and proved scientific jobs/configs unchanged, preserved
all completed pilots and recorded old/new plan digests in the ledger. Two targeted
capacity-revision tests passed; source/model/data/job settings were not changed.
No required grid cell or seed was removed. Capacity is now sufficient; actual
costs may differ from the deliberately conservative estimate.

Main parent `m2-real-resnet-20261108-parent-a5d31ab9f307` is fitting VGG on pooled
data for B=20. Remaining parents/controls continue through the worker until its
continuous-session headroom is exhausted. Main convergence, required grid/repeat
completion and M2.5 quality review remain pending. Live artifacts override this
start-of-main snapshot; do not mark scientific acceptance complete from the pilot.


Additional session/rest tests: two passed (0.029s),
`artifacts/m2-4-session-tests.log`. Resource-only revision tests: two passed
(2.170s), `artifacts/m2-4-capacity-revision-tests.log`. These are additional to the
206-case full CPU suite and saved-report hand-counted case above.


Reviewed executor suite: all seven cases passed in 60.272s,
`artifacts/m2-4-executor-reviewed-tests.log`, including a genuine nonzero subprocess
exit and pruning-publication failure followed by safe retry without parent refit.
The session adapter uses `tools/reconcile_study_accounting.py` to classify failed
pre-fit publication as known-zero fitting work only when frozen inputs and the
persistent M1 stage ledger prove no fit command ever began. It appends a cost
correction, retains all failed outer wall time, and never fabricates training
history/checkpoints or clears unknown partial training. Core frozen fitting/study
source stayed unchanged during these operational tool improvements.
