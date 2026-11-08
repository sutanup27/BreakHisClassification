# Milestone 1 package tracker

Current roadmap milestone: **Reproducible experiment foundation**.
Prepared: 8 November 2026. Master: [M1_SPEC.md](M1_SPEC.md).
Plan: [roadmap](../../research/EXPERIMENT_ROADMAP.md).
Overall progress: [milestone tracker](../../research/MILESTONE_TRACKER.md).
Specification review: [alignment review](ALIGNMENT_REVIEW.md).

**Specification package complete; milestone implementation acceptance remains
in progress.** Existing code is reused and revalidated. Package statuses below
describe acceptance against this new contract, not the absence of all code.
Checklists are completed only from current evidence; earlier implementation
results are not silently promoted to coverage of new phase/role requirements.

| Package | Target | Requirements | Tests | Depends on | Acceptance status |
| --- | --- | --- | --- | --- | --- |
| [M1.1 Config, storage and identity](M1_1_CONFIG_IDENTITY.md) | Valid inputs, no-overwrite paths and phase/run contracts | M1-R01, M1-R02, M1-R03 | M1-T01-M1-T04 | None | Reviewed; 14 new tests, 77 total passed |
| [M1.2 Splits and roles](M1_2_SPLITS_ROLES.md) | Full-source leakage checks and requested-role data access | M1-R04, M1-R05, M1-R06 | M1-T05-M1-T08 | M1.1 | Not started |
| [M1.3 Execution and resume](M1_3_EXECUTION_RESUME.md) | Development-only default, exact resume and complete histories | M1-R07, M1-R08, M1-R09, M1-R10 | M1-T09-M1-T14 | M1.1, M1.2 | Not started |
| [M1.4 Evaluation and comparison](M1_4_EVALUATION_COMPARISON.md) | Bound explicit final inference and phase-specific outputs/tables | M1-R11, M1-R12, M1-R13, M1-R14, M1-R15 | M1-T15-M1-T21 | M1.1-M1.3 | Not started |
| [M1.5 Integration and docs](M1_5_INTEGRATION.md) | Complete public workflow and accurate guides | M1-R16, M1-R17 | M1-T22-M1-T25 | M1.1-M1.4 | Not started |

## Implementation path

```text
M1.1 config/identity -> M1.2 role-safe data -> M1.3 development/resume
                    -> M1.4 evaluation/comparison -> M1.5 integration/docs
```

Next: implement/revalidate **M1.2 split and role isolation**. Do not start a new
scientific training matrix merely to finish this foundation. The acceptance
fixtures should be small and deterministic, with fresh-process smoke coverage.

## Status and evidence

Package status: `Not started -> In progress -> Tests passed -> Reviewed`.
The global milestone is Complete only after every package is reviewed and the
integrated master acceptance matrix passes. Document review and implementation
review are separate. Ordinary test logs and run outputs suffice; no approval
service, mandatory review owner or elaborate archive is required.

M1.1 implementation base: HEAD `c4b368e` plus current changes; exact source hashes
are saved in the real preparation's `inputs.json`. The current CPU suite passed
77 tests, including 14 new preparation/identity tests. See M1.1's evidence section
for commands and logs. M1-T01-M1-T04 are covered; M1-T05-M1-T25 remain outstanding.
Preparation succeeds but fitting/final inference through phase flags is not yet
implemented. Existing no-phase execution remains the legacy test-evaluating path.

Update the owning checklist, this table and the global tracker after actual
implementation/test work. If evidence invalidates a completed item, reopen it
with the concrete reason rather than retaining an inaccurate completion claim.
