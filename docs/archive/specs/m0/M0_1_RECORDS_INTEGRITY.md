> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.1 — Records and integrity

Status: Reviewed (Codex implementation/self-review, 7 November 2026). Depends on: none. Master mapping: WP0; sections 3-6, 7 input validation, 10.
Contract: [M0 specification](../../../specs/m0/M0_SPEC.md). Progress: [tracker](README.md).

## Build checklist

- [x] Add `breakhis/research/records.py` and package initialization with no experiment side effects on import.
- [x] Define versioned generated-record envelopes and separate pre-run input validation; enforce required fields, supported versions, finite values and explicit unknowns.
- [x] Define exact-byte hashes and separate canonical semantic hashes; stable path/order conventions must not depend on timestamps or absolute roots.
- [x] Implement UTF-8 atomic JSON writes and referenced-artifact verification; reject missing/tampered evidence rather than regenerating it silently.
- [x] Validate IDs, config keys/types/ranges and config-directory-relative paths; reject traversal, source/output overlap and escaping symlinks/reparse points.
- [x] Refuse existing nonempty run destinations; prevent self-capture and historical/data overwrite.
- [x] Define the evidence-index interface needed later by M0.6 without implementing the full gate/runner here.

## Tests and completion gate

- [x] Implement foundation coverage for M0-T05, M0-T21, M0-T22: unsafe paths/nonempty output, unsupported schema, deleted/tampered records, and semantic versus file hash behavior.
- [x] Include ordering/relocation fixtures, interrupted-write behavior and read-only verification checks.
- [x] Preserve existing CLI/checkpoint behavior; run relevant regressions for changed shared code.
- [x] Review confirms correct hash meanings, explicit unknowns and no data mutation. Full evidence-index/gate integration remains M0.6.

Deliverables: records helper/API, schema contract references, `tests/test_research_records.py`, and actual test/review evidence. Do not create dataset audits or fit models in this package.

Implemented interface and schema binding: [M0.1 records API](M0_1_RECORDS_API.md).

## Progress evidence

- Source identity: base commit `bf32956024bac49dff4099223f137b2b11e0cd46` plus exact implementation/test SHA-256 values in `artifacts/research/m0_1_acceptance/20261107-16299b64/summary.json`. This is package evidence, not the future M0.2 source snapshot.
- Test commands/results/logs: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`: 70 tests passed, zero skips, including 21 M0.1 tests. Retained log: `artifacts/research/m0_1_acceptance/20261107-16299b64/full-suite.log`; evidence is local/ignored, not a public repository artifact.
- Reviewer/date: Codex self-review, 7 November 2026; no independent scientific/human attestation is implied.
- Open issues/restrictions: full evidence graph/completeness/G0 integration remains M0.6. M0.2 source policy/consistency and later domain payload validators remain required. Atomic no-clobber publication requires hard-link support; static containment does not defend against hostile concurrent directory replacement. See the API contract.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M0.2 dependency released.
