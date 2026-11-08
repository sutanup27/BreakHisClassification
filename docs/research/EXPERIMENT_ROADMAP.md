# Research experiment milestones

Progress and next actions: [milestone tracker](MILESTONE_TRACKER.md).
Milestone 1 development contract: [master spec and package tracker](../specs/m1/README.md).

Active implementation path, updated 8 November 2026. Use the five milestones
below for the current research project.

The research objective remains studying classification quality and efficiency
under pruning, including the proposed mechanism and appropriate controls.
This plan simplifies the workflow, not the scientific questions or evidence.

| Milestone | Deliverable | Current implementation status | Acceptance |
| --- | --- | --- | --- |
| 1. Reproducible experiment foundation | Config-driven runs, saved group-separated splits, leakage checks, histories, predictions, metrics and explicit resume | Mostly implemented | Repeatable setup/run/evaluate/compare workflow; scientific quality remains separate. |
| 2. Dense and conventional pruning baselines | Competitive dense model, conventional pruning, compact/scratch controls and matched training budgets | Basic workflow implemented; trained baselines and controls pending | Establish the development-set detection/latency frontier and measured costs. |
| 3. Malignant-group vulnerability analysis (O1) | Controlled channel-removal probes, per-group damage and stability across fits | Not implemented | Determine whether conventional importance hides concentrated malignant-group harm. |
| 4. Proposed method and decisive comparisons (O2) | Group-balanced vulnerability-aware pruning; selection-only, malignant-weighted and established competitors; targeted ablations | Not implemented | Isolate any improvement attributable to the proposed mechanism, or report a negative finding. |
| 5. Final validation and reporting (O3) | Selected seeds/folds, paired uncertainty, repeated CPU/GPU timing, feasible replication and research tables/plots | Basic reporting available; research validation pending | Freeze development choices before final evaluation and bound claims to the evidence. |

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
Threshold calibration, constrained search, additional grouped folds, the proposed
method and research conclusions remain explicit work. Keep the workflow simple;
implement these as experiment/analysis capabilities rather than approval gates.

## Implementation verification

The generated-image workflow completed dense and channel-pruned/fine-tuned runs,
exported predictions and comparison JSON/CSV, and reproduced identical dense
weights for matching seeds. Its local summary is
`artifacts/simple-smoke-20261107T181341-f2a8/summary.json`.

The real `baseline-real-check-20261107` run also completed: 1,326 training images
in 57 filename groups, 332 validation images in 13 groups, and 303 test images
in 11 groups. It trained one epoch on the RTX 5070 Ti and evaluated/benchmarked
on CPU, saving the config, manifest, histories, predictions and metrics under
`artifacts/experiments/baseline-real-check-20261107/`. This is an execution check,
not evidence of convergence or final research performance. The example full
experiments remain prespecified; do not tune them using this test report.
