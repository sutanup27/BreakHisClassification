# Simple experiment milestones

Active implementation path, 7 November 2026. Replaces the governance-heavy
M0/M1 package sequence for everyday experiment development. Historical specs,
archives and optional G0 tools remain available; they are not prerequisites.

The research objective remains studying classification quality and efficiency
under pruning, including the proposed mechanism and appropriate controls.
This plan simplifies the workflow, not the scientific questions or evidence.

| Milestone | Deliverable | Current implementation status | Acceptance |
| --- | --- | --- | --- |
| 1. Setup/config | Existing environment plus one JSON per experiment | Implemented | Strict config validation and documented run commands; no research approval dependency. |
| 2. Saved splits | Existing audited group-separated manifest reused across runs | Implemented | Group/pixel/raw leakage, linked components and both-class coverage checked; preparation refuses overwrite. |
| 3. End-to-end baseline | Config-driven training, explicit resume, prediction/metric/history export | Implemented; real-study results pending | Generated-image smoke plus resume/regression tests; full-data convergence still to establish. |
| 4. Pruning/comparison | Dense/channel or masked pruning, fine-tuning, JSON/CSV run comparison | Implemented for existing pruning baselines; real-study comparison pending | Shared settings/seeds/splits and explicit budgets; synthetic execution verifies plumbing only. |
| 5. Research suite/reporting | Proposed mechanism, needed controls, selected seed/fold repeats and research tables/plots | Partially supported; scientific experiments and proposed mechanism remain | Configs/seeds/run comparison reusable now. Complete meaningful experiments, uncertainty/replication and research conclusions later. |

## Immediate experiment checklist

- [x] Prepare/reuse the real dataset split and inspect class/group support (40X).
- [x] Run and inspect a real-data one-epoch baseline execution check (40X).
- [ ] Run a sufficiently trained dense baseline; inspect convergence.
- [ ] Run fixed pruning variants and an equal-total-budget dense control.
- [ ] Repeat selected configurations with prespecified seeds.
- [ ] Review paired comparisons without tuning on the held-out test outcomes.
- [ ] Implement/test the proposed research mechanism and necessary controls.
- [ ] Add uncertainty, plots and any feasible additional folds; complete literature
  comparisons and write conclusions within measured evidence.

Use [the experiment guide](../guides/EXPERIMENTS.md) and [config reference](../guides/EXPERIMENT_CONFIG.md).
The earlier pilot and nested-fold research plans remain design references.
Threshold calibration, constrained search, full nested folds, proposed-method
implementation and scientific acceptance are not silently marked complete.

## Implementation verification

All 203 CPU regression tests passed, including 12 new config/leakage/aggregation
tests and the existing exact-resume checks. The separate generated-image workflow
completed dense and channel-pruned/fine-tuned runs, exported predictions and
comparison JSON/CSV, and reproduced identical dense weights for matching seeds.
Its local summary is `artifacts/simple-smoke-20261107T181341-f2a8/summary.json`.

The real `baseline-real-check-20261107` run also completed: 1,326 training images
in 57 filename groups, 332 validation images in 13 groups, and 303 test images
in 11 groups. It trained one epoch on the RTX 5070 Ti and evaluated/benchmarked
on CPU, saving the config, manifest, histories, predictions and metrics under
`artifacts/experiments/baseline-real-check-20261107/`. This is an execution check,
not evidence of convergence or final research performance. The example full
experiments remain prespecified; do not tune them using this test report.
