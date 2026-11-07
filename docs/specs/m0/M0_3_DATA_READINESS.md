# M0.3 — Data readiness

Status: Not started. Depends on: M0.1-M0.2 reviewed. Master mapping: WP3-WP4; sections 5.3-5.4, 6.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

## Build checklist

- [ ] Adapt existing inventory/parser/duplicate logic for `cancer_binary`; preserve CLI/manifest compatibility and use a new research-only output.
- [ ] Account for every supported image, including failures, with available raw hashes, decoded-pixel hashes, parse/decode status and exclusion reasons.
- [ ] Preserve filename suffixes/all-magnification grouping; create stable image/component IDs and deterministic duplicate representatives.
- [ ] Detect mixed-label duplicates/components as blockers; never silently fix labels or merge groups to match published counts.
- [ ] Produce class/subtype/magnification/group, size, dimension, exclusion and historical-discrepancy summaries; record absent magnifications without imputation.
- [ ] Ingest lineage evidence with verified/self-reported/unknown/disputed states and owners for missing derivative/original facts.
- [ ] Record bounded near-duplicate screening/review or its incompleteness; no automatic exclusion of merely similar tissue.
- [ ] Validate boundaries if reusing a legacy pilot manifest; label it as legacy pilot assignment, not research outer folds. Recheck content/file-set consistency.

## Tests and completion gate

- [ ] Implement M0-T06 through M0-T10: synthetic magnifications, alternate encodings/duplicate names, label conflicts, corrupt/misplaced/unreadable images, missing/empty sources and absent magnifications.
- [ ] Implement M0-T15: unknown lineage/clinical identity stays unknown and yields truthful restrictions.
- [ ] Assert discovered/included/excluded accounting reconciles and original inputs are not modified.
- [ ] Review verifies evidence quality, not just output existence. Valid unresolved lineage may permit restricted exploratory work; label conflicts must still block affected readiness.

Deliverables: audit adapters, complete inventory/component/summary records, lineage/review records and tests. Real local audit execution is coordinated in M0.6; no fold generation or model fitting here.

## Progress evidence

- Source identity: Pending.
- Test commands/results/logs: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Not assessed.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M0.4 dependency released.
