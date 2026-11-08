> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.4 — Exposure governance

Status: Reviewed (implementation/self-review, 7 November 2026). Depends on: M0.1-M0.3 reviewed. Master mapping: WP5; section 5.5 and G0 exposure rules.
Contract: [M0 specification](../../../specs/m0/M0_SPEC.md). Progress: [tracker](README.md).

Implemented schemas and usage: [exposure API](M0_4_EXPOSURE_API.md).

## Build checklist

- [x] Implement exposure-record validation and resolution against historical manifests plus current inventory/component identity.
- [x] Resolve prior real evaluate/compare access using metadata, partition and magnification filters; preserve evidence references and confidence.
- [x] Without per-image access logs, mark the selected partition/filter as a conservative superset; unresolved manifests/contradictions remain unresolved.
- [x] Propagate restrictions to duplicate-linked components and all associated magnifications, including confirmed equivalent derivative identities.
- [x] Keep synthetic evidence under separate source identity; realistic filenames must not quarantine real groups accidentally.
- [x] Produce known-exposed/development-only, unresolved and candidate-untouched-with-reviewed-coverage records; no-log evidence cannot prove untouched status.
- [x] Require a history-coverage declaration and reviewer evidence before confirmatory eligibility. Record missing coverage as a blocker.
- [x] Inspect only metadata needed for governance; do not rerun old models or inspect probability/error values to decide exposure.

## Tests and completion gate

- [x] Implement M0-T11 through M0-T14: exact historical filters, conservative coverage, missing/contradictory history and synthetic-real separation.
- [x] Assert component-wide propagation, no accidental unrelated-group exposure and no test inference during resolution.
- [x] Review confirms that reseeding or rebuilding splits cannot erase exposure; eligibility is not chosen using historical metric quality.
- [x] Carry unresolvable history forward with explicit owners/restrictions; engineering review cannot silently approve independent confirmation.

Deliverables: exposure resolver/register and component eligibility record, tests and reviewed coverage limitations. No held-out split selection or prediction collection.

## Progress evidence

- Source identity: HEAD `9ddc9c937ff5d8b7d5a9cef9e08b12b8805a98fd` plus dirty snapshot `a893777c6245b323ec6940f9b29af361dbcf6b75aa5d0c28a8ff45a0dacbee56`. Commit alone does not identify the tested implementation. Snapshot precedes these final acceptance notes; exact code/test hashes are in `acceptance-summary.json`.
- Current local evidence: `artifacts/research/m0/m0-4-acceptance-20261107T111937Z/` (ignored local outputs, not committed). Contains current source/environment/check evidence, synthetic `audit/` plus `governance/`, and `fixtures/missing-history/` blocked evidence. `acceptance-summary.json` explicitly labels inputs synthetic-only and real-history review false; `verification-summary.json` records source/log/reference hashes, component propagation and eligibility accounting.
- Actual command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 128 tests passed, no skips, in 63.375 seconds. Logs: `logs/regressions.stdout.log` / `logs/regressions.stderr.log` in that run. Imports, six CLI help commands, pip commands and synthetic CPU environment checks passed; source remained consistent throughout checks.
- Coverage: 23 `tests/test_m0_exposure.py` cases trace M0-T11-T14 and reviewer/coverage failure handling. They check exact/conservative filters, missing/stale/contradictory references, changed equivalence evidence, partial coverage, operator recollections, synthetic identity collision, derivative linkage, inventory tampering, no image decoding/metric inputs, no overwrite and reseed-invariant exposure. The prior 105 tests passed in this fresh run.
- Saved synthetic result: one component containing three images across linked filename groups/magnifications becomes development-only; two unrelated components remain candidates solely within the fixture's reviewed boundary. A valid synthetic event maps to no real images/components. The missing-history fixture leaves all three components unknown and blocks history review. These are software acceptance fixtures, not statements about real BreaKHis access.
- Reviewer/date: Codex implementation/self-review, 7 November 2026; engineering review, not independent history adjudication.
- Restrictions: Real historical artifact discovery, coverage completeness and reviewer attestations remain unreviewed and are not inferred from missing logs. Hash-matched statements are attributable declarations, not proof of their truth. M0.4 always leaves final confirmatory-design readiness false; lineage/protocol, fold support and Gate G0 remain downstream. GPU execution is unvalidated in this CPU acceptance run. No actual historical prediction, metric quality or model inference was inspected for exposure decisions.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M0.5 dependency released.
