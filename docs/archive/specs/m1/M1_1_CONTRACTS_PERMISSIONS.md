> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M1.1 — Contracts and permissions

Status: Reviewed (implementation/self-review, 7 November 2026). Depends on: M0 engineering acceptance.
Master mapping: WP0; [M1 specification](M1_SPEC.md), sections 2-5.1 and 6.
Progress: [tracker](README.md).

Implemented usage/schema binding: [M1.1 API](M1_1_CONTRACTS_API.md).

## Build checklist

- [x] Define strict M1 config/envelopes/JSONL types and content identities, distinct from M0 spec identity.
- [x] Verify M0 index/receipt/gate/handoff and bind frozen source/inventory/component/exposure/protocol versions.
- [x] Enforce engineering, diagnostic and real-design operation permissions; carry G0 blockers forward.
- [x] Define role-sealed data/reporting contracts; deny development access to outer-test construction, iteration and export.
- [x] Implement portable paths, new-run/no-overwrite rules, atomic finite writes and partial failure evidence.
- [x] Generalize shared helpers without weakening M0 validation or invalidating its archives; define legacy adapters explicitly.
- [x] Draft engineering-only config inputs with synthetic settings; retain unknown real scientific choices and owners.

## Target and completion gate

Deliverable: record/config/lineage and operation-permission APIs used by all later packages.

- [x] M1-T01: altered/missing M0 evidence cannot grant access.
- [x] M1-T02: unknown schemas/config/roles, unsafe paths and nonempty output refuse before mutation.
- [x] M1-T03: canonical identity survives ordering/relocation and changes with content/policy.
- [x] Confirm current G0 permits infrastructure work only; neither templates nor copied flags authorize real fits/design.
- [x] Existing M0 archive verification and applicable record/path regressions still pass.

## Progress evidence

- Source identity: HEAD `65cdc50c3eab0f7520e7820ac143abe15ad4e2bf` plus dirty snapshot `08b86d0fb975e5e87d5115d8d2d2064d63ba30ef04fc6db460afd152972ee43d`. Snapshot precedes final acceptance notes. The source-only preservation directory is `artifacts/research/m0/m1-1-source-20261107T135106Z/`; it uses the existing M0-specific source capture contract and is not another completed M0 or M1 run. M1 full source orchestration remains M1.6.
- Current local evidence: `artifacts/research/m1_1_acceptance/m1-1-acceptance-20261107T135106Z/summary.json`, `logs/full-suite.stdout.log` and `logs/full-suite.stderr.log` (ignored outputs, not committed). Summary records exact implementation/test/config hashes, source identity, parent pins, permissions and denials. `verification-summary.json` records independent reconstruction/log/parent/schema checks.
- Actual command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 187 tests passed, no skips, in 88.248 seconds. Source entries remained unchanged during execution. The previous 167 tests passed alongside 20 new contract tests.
- Coverage: 20 `tests/test_m1_records.py` cases cover M1-T01-T03, namespaced registries, import safety, atomic/no-clobber/failure handling, output junctions, config/JSONL types, unknown pins, semantic identity, late parent tampering, copied flags, detached/immutable entry state, pre-callback outer-role denial, source/component collisions, strict development/calibration roles, diagnostic-only access and failed partial scaffolds.
- Actual parent reverification: `artifacts/research/m0/m0-real-20261107T130421Z/` verified successfully after shared IO/schema generalization. All parent file hashes and mtimes stayed unchanged. Real diagnostic metadata access succeeded; fit, real image iteration, real prediction export and active diagnostic fold planning were denied. Existing G0 restrictions are inherited, not cleared.
- Created scaffold: `artifacts/research/m1/m1-1-created-20261107T135106Z/` contains run/config/verified-entry records only, with state `created`, scientific authorization false and current M1 full capture pending. No folds, loaders, fitting, calibration or G1 were launched.
- Reviewer/date: Codex implementation/self-review, 7 November 2026; engineering package review, not independent scientific approval.
- Open restrictions: Current lineage/history/protocol/owner/literature/fold-support blockers remain. JSONL currently defines identity rows only; later packages validate full assignment/history/prediction semantics. Role guards require future adapters to use the boundary and are not an arbitrary-code sandbox. GPU/cross-environment acceptance is not claimed for this stdlib-only package; CPU suite and the actual M0 archive establish current compatibility.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M1.2 dependency released.
