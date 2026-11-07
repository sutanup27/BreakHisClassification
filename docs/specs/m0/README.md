# M0 implementation checklist tracker

Prepared: 7 November 2026. Scope: implementation planning only.

[M0_SPEC.md](M0_SPEC.md) is the authoritative contract for schemas, semantics and Gate G0. These six checklists organize delivery; they do not replace or duplicate that specification. M0.1 is implemented and self-reviewed; M0.2-M0.6 have not started.

All M0-specific documents live together in this folder:

```text
docs/specs/m0/
  README.md                         # Tracker and starting point
  M0_SPEC.md                        # Master source of truth
  M0_1_RECORDS_INTEGRITY.md
  M0_1_RECORDS_API.md               # Implemented foundation API
  M0_2_BASELINE_PRESERVATION.md
  M0_3_DATA_READINESS.md
  M0_4_EXPOSURE_GOVERNANCE.md
  M0_5_PROTOCOL_LITERATURE.md
  M0_6_RUNNER_GATE.md
```

## Work packages and progress

| Package | Master work packages | Depends on | Status | Completion evidence |
| --- | --- | --- | --- | --- |
| [M0.1 Records and integrity](M0_1_RECORDS_INTEGRITY.md) | WP0 | None | Reviewed | 21 foundation tests; 70 total CPU tests passed, no skips; [API/evidence](M0_1_RECORDS_INTEGRITY.md#progress-evidence) |
| [M0.2 Baseline preservation](M0_2_BASELINE_PRESERVATION.md) | WP1-WP2 | M0.1 reviewed | Not started | Pending |
| [M0.3 Data readiness](M0_3_DATA_READINESS.md) | WP3-WP4 | M0.1-M0.2 reviewed | Not started | Pending |
| [M0.4 Exposure governance](M0_4_EXPOSURE_GOVERNANCE.md) | WP5 | M0.1-M0.3 reviewed | Not started | Pending |
| [M0.5 Protocol and literature](M0_5_PROTOCOL_LITERATURE.md) | WP6 | M0.1-M0.4 reviewed | Not started | Pending |
| [M0.6 Runner and Gate G0](M0_6_RUNNER_GATE.md) | WP7-WP9 | M0.1-M0.5 reviewed | Not started | Pending |

The default path is M0.1 through M0.6. Literature collection and locating lineage evidence may begin earlier, but dependency acceptance is required before marking a package reviewed. A dependency's reviewed unknowns may be carried forward as restrictions; review does not assert that every scientific question is resolved.

## Status rules

Use `Not started -> In progress -> Tests passed -> Reviewed`.

- **In progress:** implementation exists or active development is recorded.
- **Tests passed:** required package tests have actual logs/results; planned tests or unexplained required skips do not qualify.
- **Reviewed:** acceptance evidence was checked against the master contract; reviewer, source identity and unresolved restrictions are recorded.
- A blocker is recorded separately with owner, reason and resolution condition. Do not substitute blocked scientific readiness with an engineering pass.
- Reopen a package when later changes invalidate its acceptance evidence. Update this tracker and its checklist together.
- Package review is an engineering gate, not a new blanket permission/approval request. Scientific attestations and real-run authorization follow the master spec and current user instructions.

For every package retain: source commit plus dirty snapshot identity; test command/log/result; output/schema references; review date/reviewer; open decisions and affected later packages. Generated evidence belongs under the ignored research run directory; never fabricate evidence paths in this tracker.

## Acceptance-test ownership

| Package | Master acceptance tests |
| --- | --- |
| M0.1 | M0-T05, M0-T21, M0-T22 (record/hash/path foundation; full integration repeated in M0.6) |
| M0.2 | M0-T01, M0-T02, M0-T03, M0-T04, M0-T19, M0-T20 |
| M0.3 | M0-T06, M0-T07, M0-T08, M0-T09, M0-T10, M0-T15 |
| M0.4 | M0-T11, M0-T12, M0-T13, M0-T14 |
| M0.5 | M0-T16, M0-T17, M0-T18 |
| M0.6 | M0-T23, M0-T24, M0-T25, M0-T26; integrates all earlier cases including M0-T21/M0-T22 |

All M0-T01 through M0-T26 must have final traceability. Test IDs refer to the master acceptance matrix, not tests already implemented. M0.6 cannot declare full completion using package-level unit tests alone.

## Starting point

M0.1 is complete at package scope; use its [API contract](M0_1_RECORDS_API.md) for subsequent work. [M0.2](M0_2_BASELINE_PRESERVATION.md) is the next package and remains not started. No M0 runner/G0 execution or scientific experiment is implied by package completion.
