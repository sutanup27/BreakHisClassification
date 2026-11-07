# M0.4 — Exposure governance

Status: Not started. Depends on: M0.1-M0.3 reviewed. Master mapping: WP5; section 5.5 and G0 exposure rules.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

## Build checklist

- [ ] Implement exposure-record validation and resolution against historical manifests plus current inventory/component identity.
- [ ] Resolve prior real evaluate/compare access using metadata, partition and magnification filters; preserve evidence references and confidence.
- [ ] Without per-image access logs, mark the selected partition/filter as a conservative superset; unresolved manifests/contradictions remain unresolved.
- [ ] Propagate restrictions to duplicate-linked components and all associated magnifications, including confirmed equivalent derivative identities.
- [ ] Keep synthetic evidence under separate source identity; realistic filenames must not quarantine real groups accidentally.
- [ ] Produce known-exposed/development-only, unresolved and candidate-untouched-with-reviewed-coverage records; no-log evidence cannot prove untouched status.
- [ ] Require a history-coverage declaration and reviewer evidence before confirmatory eligibility. Record missing coverage as a blocker.
- [ ] Inspect only metadata needed for governance; do not rerun old models or inspect probability/error values to decide exposure.

## Tests and completion gate

- [ ] Implement M0-T11 through M0-T14: exact historical filters, conservative coverage, missing/contradictory history and synthetic-real separation.
- [ ] Assert component-wide propagation, no accidental unrelated-group exposure and no test inference during resolution.
- [ ] Review confirms that reseeding or rebuilding splits cannot erase exposure; eligibility is not chosen using historical metric quality.
- [ ] Carry unresolvable history forward with explicit owners/restrictions; engineering review cannot silently approve independent confirmation.

Deliverables: exposure resolver/register and component eligibility record, tests and reviewed coverage limitations. No held-out split selection or prediction collection.

## Progress evidence

- Source identity: Pending.
- Test commands/results/logs: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Not assessed.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M0.5 dependency released.
