# M1 implementation checklist tracker

Prepared: 7 November 2026. Scope: specification and delivery planning only.

[M1_SPEC.md](M1_SPEC.md) is the source of truth for M1 schemas, permissions,
semantics and Gate G1. These six checklists organize gradual implementation;
they do not duplicate the master contract. **All M1 packages are Not started.**

M0 engineering acceptance is complete. Its current G0 allows CPU/interface and
frozen-inventory diagnostics, while real scientific fits and untouched design
remain blocked. Creating these documents does not clear those restrictions or
start M1 code/experiments.

```text
docs/specs/m1/
  README.md
  M1_SPEC.md
  M1_1_CONTRACTS_PERMISSIONS.md
  M1_2_FOLDS_ROLES.md
  M1_3_SEEDS_RESUME.md
  M1_4_DEVELOPMENT_REPORTING.md
  M1_5_AGGREGATION_SELECTION.md
  M1_6_RUNNER_GATE.md
```

## Work packages and progress

| Package | Master work packages | Depends on | Status | Completion evidence |
| --- | --- | --- | --- | --- |
| [M1.1 Contracts and permissions](M1_1_CONTRACTS_PERMISSIONS.md) | WP0 | M0 engineering acceptance | Not started | Pending |
| [M1.2 Folds and roles](M1_2_FOLDS_ROLES.md) | WP1 | M1.1 reviewed | Not started | Pending |
| [M1.3 Seeds, history and resume](M1_3_SEEDS_RESUME.md) | WP2-WP3 | M1.1-M1.2 reviewed | Not started | Pending |
| [M1.4 Development reporting](M1_4_DEVELOPMENT_REPORTING.md) | WP4 | M1.1-M1.3 reviewed | Not started | Pending |
| [M1.5 Aggregation and selection](M1_5_AGGREGATION_SELECTION.md) | WP5-WP6 | M1.1-M1.4 reviewed | Not started | Pending |
| [M1.6 Runner and Gate G1](M1_6_RUNNER_GATE.md) | WP7-WP9 | M1.1-M1.5 reviewed | Not started | Pending |

Default path: M1.1 -> M1.2 -> M1.3 -> M1.4 -> M1.5 -> M1.6.
The [master dependency table](M1_SPEC.md#7-work-packages-and-dependencies) defines
package acceptance order. Synthetic scaffolding may be shared; no dependent
package is reviewed before its prerequisites.

## Status rules

Use `Not started -> In progress -> Tests passed -> Reviewed`.

- **In progress:** implementation has started; record scope and remaining issues.
- **Tests passed:** required current checks have actual commands/results/logs.
- **Reviewed:** evidence is checked against the master semantics; source identity,
  reviewer/date, input versions and restrictions are recorded.
- Track blockers separately with reason, owner, affected operation and resolution.
  Unresolved scientific facts cannot become an engineering pass or implied fit permission.
- Reopen packages if later changes invalidate their contracts/evidence; update
  the package checklist and this tracker together.
- Package review is an engineering gate, not a repeated blanket permission request.
  Real operations still require the applicable G0/G1 scope and agreed policies.

Every package retains current commit plus dirty-source identity, commands/logs,
schema/output references, acceptance IDs, reviewer/date and open restrictions.
Generated evidence belongs under ignored research outputs, never invented paths.

## Acceptance-test ownership

| Package | Master acceptance IDs |
| --- | --- |
| M1.1 | M1-T01, M1-T02, M1-T03 |
| M1.2 | M1-T04 through M1-T10 |
| M1.3 | M1-T11 through M1-T15 |
| M1.4 | M1-T16 through M1-T19 |
| M1.5 | M1-T20 through M1-T26 |
| M1.6 | M1-T27 through M1-T30; integrated replay of earlier integrity/permission cases |

All M1-T01 through M1-T30 need final pass/fail/not-run traceability. IDs describe
required tests, not tests already implemented. Historical 167-test M0 runs do not
satisfy new M1 acceptance.

## Starting point

Begin with [M1.1](M1_1_CONTRACTS_PERMISSIONS.md). Use the completed
[M0 handoff/acceptance](../m0/M0_6_RUNNER_GATE.md#progress-evidence) as the entry
boundary. No package is implemented or marked reviewed by this planning change.
