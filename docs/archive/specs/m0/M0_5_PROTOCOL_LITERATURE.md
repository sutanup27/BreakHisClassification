> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.5 — Protocol and literature

Status: Reviewed (implementation/self-review, 7 November 2026). Depends on: M0.1-M0.4 reviewed. Master mapping: WP6; sections 5.6-5.7 and related G0 restrictions.
Contract: [M0 specification](../../../specs/m0/M0_SPEC.md). Progress: [tracker](README.md).

Implemented contract: [protocol API](M0_5_PROTOCOL_API.md). Human draft:
[research protocol](../../research/RESEARCH_PROTOCOL.md).

## Build checklist

- [x] Add initial protocol/config templates and `docs/research/RESEARCH_PROTOCOL.md`; validate all mandatory master-spec topics without copying schema definitions into this checklist.
- [x] Mark decisions agreed/proposed/deferred/blocked with rationale, evidence, owner and freeze-by milestone; preserve revision/supersedes history.
- [x] Record source/group/exposure boundaries, proposed primary/secondary aggregation, missing-magnification policy, CPU/GPU roles and provisional architecture/control scope.
- [x] Declare seed namespaces, role/fold intent, distinct epoch/candidate/threshold selection, safeguards, costs/budgets, A/B/C/D/E comparisons and negative-result route.
- [x] Keep unresolved numerical margins/thresholds/trial counts explicit; prevent draft values from becoming approved clinical or confirmatory policy.
- [x] Build the nearest-literature matrix with actual access/review status, method/selection/recovery/runtime details, code/license and precise unresolved overlaps.
- [x] Distinguish full-method reading from abstracts/code access; assign owners to central inaccessible/unreviewed methods. No automated novelty-established claim.
- [x] Bind protocol/decisions/review inputs to source/audit/exposure versions; M0 cannot emit a confirmatory-frozen protocol.

## Tests and completion gate

- [x] Implement M0-T16 through M0-T18: proposed choices remain proposed, missing critical topic/owner/deadline is detected, and abstract/code-only access cannot count as full methods.
- [x] Review verifies all required topics, evidence references and lifecycle transitions. Tests check semantics, not whether literature/provenance assertions are scientifically true.
- [x] Unresolved later-stage decisions may pass engineering review only with owners and freeze milestones; central novelty gaps remain explicit restrictions.

Deliverables: protocol/decision/literature validators, versioned templates, human protocol, curated input records and tests. Do not fit thresholds, train models or claim novelty here.

## Progress evidence

- Source identity: HEAD `3e0c81f0bbc268a3af2e5fa2e54ec6c9e759ec39` plus dirty snapshot `aecc1d9d38b8fc63b68b88c2b874c591ea96c750f87371f35957b87f2eeb749e`. Commit alone is not the tested source. Snapshot precedes final acceptance-note edits; code/test/template hashes are in `acceptance-summary.json`.
- Current local evidence: `artifacts/research/m0/m0-5-acceptance-20261107T120425Z/` (ignored local outputs, not committed). Contains attributable baseline/current checks, synthetic audit/exposure records and bound protocol/decisions/literature/review outputs. `verification-summary.json` checks source/log/template hashes and all five bindings. Fixtures explicitly report no real audit/history approval or scientific authorization.
- Actual command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 147 tests passed, no skips, in 64.306 seconds. Logs: `logs/regressions.stdout.log` and `logs/regressions.stderr.log` in that run. Current imports, six CLI help commands, pip commands and synthetic CPU environment checks passed; source remained consistent throughout checks.
- Coverage: 19 `tests/test_m0_protocol.py` tests trace M0-T16-T18, required topic/subpolicy/owner/evidence/deadline checks, valid and invalid lifecycle declarations, unresolved/agreed numbers, seed/budget limits, preserved supersedes chains, stale binding rejection, current config compatibility and non-promoting publication. The prior 128 tests passed in the same fresh run.
- Inputs: six versioned `configs/research/` JSON templates; 17 topic/decision policies with distinct rationales. Seven primary-source literature targets have actual access levels: one focused full-method reading, three abstract-only, one code-only and two inaccessible publisher targets. Remaining bibliography/method/identity/selection/recovery/license/overlap questions have reasons and owners; all seven retain central-review restrictions. Source URLs/version/date are recorded in the matrix and human protocol.
- Saved integration result: draft lifecycle and all proposed topic statuses preserved; source/audit/exposure/decision/literature identities bound; unknown real history and incomplete central literature review retained. No automatic agreement, clinical margin, novelty assertion or confirmatory freeze was produced.
- Reviewer/date: Codex implementation/self-review and focused primary-source reading, 7 November 2026; not independent scientific or provenance adjudication.
- Restrictions: Final reviewer assignments, numerical policy agreements, original lineage, historical coverage, closest-method completeness/reproduction and scientific fit approval remain unresolved. M0.6 must provide runner/config-sidecar integration, evidence graph and Gate G0; no runnable full-M0 command is claimed here. GPU execution remains unvalidated in this CPU acceptance run. Publisher access failures do not establish that no alternative full-text source exists.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M0.6 dependency released.
