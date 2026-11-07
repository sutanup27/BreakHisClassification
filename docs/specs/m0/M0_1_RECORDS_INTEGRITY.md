# M0.1 — Records and integrity

Status: Not started. Depends on: none. Master mapping: WP0; sections 3-6, 7 input validation, 10.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

## Build checklist

- [ ] Add `breakhis/research/records.py` and package initialization with no experiment side effects on import.
- [ ] Define versioned generated-record envelopes and separate pre-run input validation; enforce required fields, supported versions, finite values and explicit unknowns.
- [ ] Define exact-byte hashes and separate canonical semantic hashes; stable path/order conventions must not depend on timestamps or absolute roots.
- [ ] Implement UTF-8 atomic JSON writes and referenced-artifact verification; reject missing/tampered evidence rather than regenerating it silently.
- [ ] Validate IDs, config keys/types/ranges and config-directory-relative paths; reject traversal, source/output overlap and escaping symlinks/reparse points.
- [ ] Refuse existing nonempty run destinations; prevent self-capture and historical/data overwrite.
- [ ] Define the evidence-index interface needed later by M0.6 without implementing the full gate/runner here.

## Tests and completion gate

- [ ] Implement foundation coverage for M0-T05, M0-T21, M0-T22: unsafe paths/nonempty output, unsupported schema, deleted/tampered records, and semantic versus file hash behavior.
- [ ] Include ordering/relocation fixtures, interrupted-write behavior and read-only verification checks.
- [ ] Preserve existing CLI/checkpoint behavior; run relevant regressions for changed shared code.
- [ ] Review confirms correct hash meanings, explicit unknowns and no data mutation. Full evidence-index/gate integration remains M0.6.

Deliverables: records helper/API, schema contract references, `tests/test_research_records.py`, and actual test/review evidence. Do not create dataset audits or fit models in this package.

## Progress evidence

- Source identity: Pending.
- Test commands/results/logs: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Not assessed.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M0.2 dependency released.
