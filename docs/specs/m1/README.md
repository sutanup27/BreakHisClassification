# Milestone 1 package tracker

Current roadmap milestone: **Reproducible experiment foundation**.
Prepared: 8 November 2026. Master: [M1_SPEC.md](M1_SPEC.md).
Plan: [roadmap](../../research/EXPERIMENT_ROADMAP.md).
Overall progress: [milestone tracker](../../research/MILESTONE_TRACKER.md).
Specification review: [alignment review](ALIGNMENT_REVIEW.md).

**Milestone 1 implementation acceptance is Complete; all five packages are reviewed.** Existing code is reused and revalidated. Package statuses below
describe acceptance against this new contract, not the absence of all code.
Checklists are completed only from current evidence; earlier implementation
results are not silently promoted to coverage of new phase/role requirements.

| Package | Target | Requirements | Tests | Depends on | Acceptance status |
| --- | --- | --- | --- | --- | --- |
| [M1.1 Config, storage and identity](M1_1_CONFIG_IDENTITY.md) | Valid inputs, no-overwrite paths and phase/run contracts | M1-R01, M1-R02, M1-R03 | M1-T01-M1-T04 | None | Reviewed; 14 new tests, 77 total passed |
| [M1.2 Splits and roles](M1_2_SPLITS_ROLES.md) | Full-source leakage checks and requested-role data access | M1-R04, M1-R05, M1-R06 | M1-T05-M1-T08 | M1.1 | Reviewed; 15 new tests, 92 total passed |
| [M1.3 Execution and resume](M1_3_EXECUTION_RESUME.md) | Development-only default, exact resume and complete histories | M1-R07, M1-R08, M1-R09, M1-R10 | M1-T09-M1-T14 | M1.1, M1.2 | Reviewed; 16 new tests, 108 total passed |
| [M1.4 Evaluation and comparison](M1_4_EVALUATION_COMPARISON.md) | Bound explicit final inference and phase-specific outputs/tables | M1-R11, M1-R12, M1-R13, M1-R14, M1-R15 | M1-T15-M1-T21 | M1.1-M1.3 | Reviewed; 23 new cases; CPU 129 passed/2 skipped, CUDA 23 passed |
| [M1.5 Integration and docs](M1_5_INTEGRATION.md) | Complete public workflow and accurate guides | M1-R16, M1-R17 | M1-T22-M1-T25 | M1.1-M1.4 | Reviewed; 135 CPU cases: 133 passed, 2 CUDA-only skips |

## Implementation path

```text
M1.1 config/identity -> M1.2 role-safe data -> M1.3 development/resume
                    -> M1.4 evaluation/comparison -> M1.5 integration/docs
```

Next: specify **milestone 2: dense and conventional pruning baselines**, including
convergence checks and appropriate controls. Foundation acceptance used small
generated fixtures; it does not establish competitive scientific results.

## Status and evidence

Package status: `Not started -> In progress -> Tests passed -> Reviewed`.
The global milestone is Complete only after every package is reviewed and the
integrated master acceptance matrix passes. Document review and implementation
review are separate. Ordinary test logs and run outputs suffice; no approval
service, mandatory review owner or elaborate archive is required.

M1.1 implementation commit: `b077f23`; exact source hashes
are saved in the real preparation's `inputs.json`. The M1.1 CPU suite passed
77 tests, including 14 new preparation/identity tests. See M1.1's evidence section
for commands and logs. M1-T01-M1-T04 are covered; M1-T05-M1-T08 are now covered by M1.2; M1-T09-M1-T14 are covered by M1.3; M1-T15-M1-T21 are covered by M1.4; M1-T22-M1-T25 are covered by M1.5. All 25 master cases have acceptance evidence.
Development execution is now the default and reports validation only. Final
inference and phase-specific comparisons are now implemented and reviewed in M1.4.

Update the owning checklist, this table and the global tracker after actual
implementation/test work. If evidence invalidates a completed item, reopen it
with the concrete reason rather than retaining an inaccurate completion claim.

M1.2 reviewed: 92 CPU tests passed, including 15 new split/role tests. Real-data
preparation/revalidation succeeded without fitting; see the M1.2 evidence section.

M1.3 reviewed: 108 CPU tests passed; fresh-process VGG dense/channel development
passed with reproduced dense weights. See M1.3 for resume/failure test evidence.

M1.4 reviewed: CPU suite ran 131 cases (129 passed, 2 CUDA-only skips); all 23
focused cases passed in the CUDA environment. Fresh-process VGG dense/channel
flows completed both phases and saved comparisons. See M1.4 for evidence.

M1.5 reviewed and Milestone 1 Complete: 135 CPU cases (133 passed, 2 CUDA-only
skips), four focused integration checks, retained fresh-process public dense/mask/
channel workflows, real failure/resume checks and documentation/link verification.
See M1.5 for logs and the distinction from scientific experiment completion.
