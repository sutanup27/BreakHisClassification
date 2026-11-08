# M2 implementation tracker

Milestone: **Dense and conventional pruning baselines**.
Prepared: 8 November 2026. Master: [M2_SPEC.md](M2_SPEC.md).
Plan: [current roadmap](../../research/EXPERIMENT_ROADMAP.md).
Global progress: [milestone tracker](../../research/MILESTONE_TRACKER.md).
Assessment: [alignment review](ALIGNMENT_REVIEW.md).

**Specification package reviewed; implementation and scientific experiments have
not started.** Reuse M1, which is complete. No baseline strength, frontier, resource
measurement or research novelty is claimed from these documents.

| Package | Purpose / completion target | Owns | Acceptance | Depends on | Implementation status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [M2.1 Matrix and recipes](M2_1_MATRIX_RECIPES.md) | Strict study compiler, frozen job graph and honest capacity accounting | M2-R01-M2-R03 | M2-T01-M2-T04 | M1 | Not started | Inspected current config/runner; new study interface proposed |
| [M2.2 Initialization and models](M2_2_INITIALIZATION_MODELS.md) | Bound parent reuse, fresh continuation/scratch and compact model | M2-R04-M2-R06 | M2-T05-M2-T09 | M2.1 | Not started | VGG widths/checkpoint primitives exist; new modes not implemented |
| [M2.3 Pruning controls](M2_3_PRUNING_CONTROLS.md) | Matched-width random/L1 surgery, masks and fair recovery through phase artifacts | M2-R07-M2-R09 | M2-T10-M2-T13 | M2.1-M2.2 | Not started | Existing L1 surgery reused; random/control integration proposed |
| [M2.4 Execution and budgets](M2_4_EXECUTION_BUDGETS.md) | Tested public executor, measured pilot and complete real core/repeat matrix | M2-R10-M2-R12 | M2-T14-M2-T18 | M2.1-M2.3 | Not started | No M2 fitting/profile performed |
| [M2.5 Frontier and handoff](M2_5_FRONTIER_HANDOFF.md) | Verified comparisons, baseline-strength review and M3 inputs | M2-R13-M2-R15 | M2-T19-M2-T22 | M2.4 executor for code; real runs for final review | Not started | No real frontier/quality review produced |

## Implementation path

```text
matrix/recipes -> initialization/models -> pruning controls
             -> executor + report/integration software on fixtures
             -> 40X/pooled cost pilot -> real parent/control fits
             -> development frontier/quality review -> M3 handoff
```

Start with **M2.1**. Build reporting/integration code from small fixtures before
long fits to avoid source-drift interruption of active runs. M2.4/M2.5 acceptance
requires real experiments; their implementation can be tested earlier. A verified
parent input is distinct from exact resume of the original parent.

## Scope and decisions

Planned main task: pooled magnifications; 40X is the cost/execution pilot. Primary
full-grid seed 102, fixed-ratio repeat seed 103 and independent ranking seed 202
are proposed starting defaults. The one-recipe matrix has 22 fitting jobs before
pilots/alternative recipes/retries. All-from-scratch initialization is the primary
policy except explicitly inherited controls. CPU is the primary deployment profile;
RTX 5070 Ti is training plus a separately measured secondary profile.

M2.4 must measure costs and settle B/F/recipe/batch/profile/capacity before main
launch; no throughput or calendar guarantee is invented. Undertrained models or
missing controls leave empirical completion incomplete. Class weighting,
constrained selection, adaptive/class-aware competitors and the proposed mechanism
are M4; vulnerability measurements are M3; paired uncertainty and final study
replication/timing sessions are M5. No second dataset is mandatory here.

## Status and evidence

Package implementation: `Not started -> In progress -> Tests passed -> Reviewed`.
Track real experiment progress separately as `Not run / Partial / Complete` in
M2.4-M2.5 evidence sections. M2 is Complete only when all 15 requirements/22
acceptance cases and the real baseline/control review are satisfied. Update the
owning checklist, this tracker and the global tracker together. No mandatory
review owner, source archive or human approval gate is added.

Planning checks: sources/current CLI reviewed; master requirement/test ownership,
package dependencies, links and policy/budget consistency checked. New commands
are labeled proposed. Existing M1 worktree changes are preserved. No training,
code implementation, test execution or Git mutation is performed by this package.
