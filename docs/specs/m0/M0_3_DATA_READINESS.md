# M0.3 — Data readiness

Status: Reviewed (implementation/self-review, 7 November 2026). Depends on: M0.1-M0.2 reviewed. Master mapping: WP3-WP4; sections 5.3-5.4, 6.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

Implemented usage and record contracts: [data readiness API](M0_3_DATA_API.md).

## Build checklist

- [x] Adapt existing inventory/parser/duplicate logic for `cancer_binary`; preserve CLI/manifest compatibility and use a new research-only output.
- [x] Account for every supported image, including failures, with available raw hashes, decoded-pixel hashes, parse/decode status and exclusion reasons.
- [x] Preserve filename suffixes/all-magnification grouping; create stable image/component IDs and deterministic duplicate representatives.
- [x] Detect mixed-label duplicates/components as blockers; never silently fix labels or merge groups to match published counts.
- [x] Produce class/subtype/magnification/group, size, dimension, exclusion and historical-discrepancy summaries; record absent magnifications without imputation.
- [x] Ingest lineage evidence with verified/self-reported/unknown/disputed states and owners for missing derivative/original facts.
- [x] Record bounded near-duplicate screening/review or its incompleteness; no automatic exclusion of merely similar tissue.
- [x] Validate boundaries if reusing a legacy pilot manifest; label it as legacy pilot assignment, not research outer folds. Recheck content/file-set consistency.

## Tests and completion gate

- [x] Implement M0-T06 through M0-T10: synthetic magnifications, alternate encodings/duplicate names, label conflicts, corrupt/misplaced/unreadable images, missing/empty sources and absent magnifications.
- [x] Implement M0-T15: unknown lineage/clinical identity stays unknown and yields truthful restrictions.
- [x] Assert discovered/included/excluded accounting reconciles and original inputs are not modified.
- [x] Review verifies evidence quality, not just output existence. Valid unresolved lineage may permit restricted exploratory work; label conflicts must still block affected readiness.

Deliverables: audit adapters, complete inventory/component/summary records, lineage/review records and tests. Real local audit execution is coordinated in M0.6; no fold generation or model fitting here.

## Progress evidence

- Source identity: HEAD `3c9b542fb2789d97c1bd9bf5a6b2c491c42bc6a0` plus dirty source snapshot `94ec5769fb20eeeb4d9240fe0621e622a18438f0c1cf6bc6337bfe6fc24d9ddd`. The commit alone is not the tested implementation. Snapshot precedes final acceptance-note edits; exact implementation/test hashes are recorded in `acceptance-summary.json`.
- Current local evidence: `artifacts/research/m0/m0-3-acceptance-20261107T103919Z/` (ignored local outputs, not committed). Includes attributable M0.2 source/environment/check evidence, `audit/` synthetic inventory and summaries, and `fixtures/label-conflict/audit/` blocked evidence. `acceptance-summary.json` explicitly labels inputs synthetic-only and real-dataset audit false; `verification-summary.json` records independent source/log/image-hash/accounting checks.
- Actual full-suite command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 105 tests passed, no skips, in 60.765 seconds. Logs: `logs/regressions.stdout.log` / `logs/regressions.stderr.log` in that run. Current imports, six CLI help commands, pip commands and synthetic CPU environment checks also passed; source remained consistent throughout checks.
- Coverage: 19 `tests/test_m0_data.py` tests cover M0-T06-T10/M0-T15 plus source drift, unsafe paths, immutable historical-manifest validation and no-overwrite failures. The previous 86 tests also passed. Canonical parser/RGB hashes are compared directly, not redefined for compatibility.
- Saved fixture result: 9 supported synthetic images = 8 included + 1 exact decoded duplicate excluded, with stable representative/group references. The two-image mixed-label duplicate fixture reports `DATA_LABEL_CONFLICT` and data-integrity readiness false; labels are preserved.
- Reviewer/date: Codex implementation/self-review, 7 November 2026; engineering package review, not independent scientific adjudication.
- Restrictions: Real BreaKHis audit remains M0.6; no real inventory totals, verified identities, untouched groups, folds or scientific fits are claimed. Perceptual near-duplicate screening is explicitly incomplete in v1; no suspicious-pair adjudication or image-specific clinical mapping is implemented. Unknown lineage needs a reviewed scope decision; exposure/protocol/fold support remain downstream. Package completion does not grant exploratory/confirmatory readiness or G0. GPU execution remains unvalidated in this CPU acceptance run.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M0.4 dependency released.
