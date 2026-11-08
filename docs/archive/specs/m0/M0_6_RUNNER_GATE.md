> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.6 — Runner and Gate G0

Status: Reviewed (engineering implementation/real-run acceptance, 7 November 2026). Depends on: M0.1-M0.5 reviewed. Master mapping: WP7-WP9; sections 4, 6-11.
Contract: [M0 specification](../../../specs/m0/M0_SPEC.md). Progress: [tracker](README.md).

Implemented CLI, schemas and readiness boundaries: [runner reference](M0_6_RUNNER_API.md).

## Build checklist

- [x] Integrate package outputs into a complete evidence index without circular hashing; verify referenced bytes before evaluating readiness.
- [x] Implement G0's engineering, exploratory, confirmatory-design, GPU and novelty dimensions, reason codes, blockers, owners and restrictions.
- [x] Test hard-failure precedence and requested-mode eligibility; a successful process cannot imply scientific readiness.
- [x] Implement `tools/run_m0.py` with validated `--config`, `--run-id`, `--verify-only`, help and documented exit codes.
- [x] Keep config-directory-relative path resolution, allowlisted command execution, explicit timeouts, logs and partial-run failure records.
- [x] Implement verify-only as read-only validation, without silently recomputing/replacing invalid evidence.
- [x] Produce a human M0 report and M1 handoff binding source, inventory/components, exposure eligibility, decisions and allowed claims.
- [x] Integrate docs/CI and all package tests; then execute real local M0 only within current user authorization and a fresh research destination.

## Tests and completion gate

- [x] Implement M0-T23/M0-T24 for isolated synthetic orchestration and read-only verification, and rerun integrated M0-T21/M0-T22 integrity cases.
- [x] Satisfy M0-T25 through existing CLI/checkpoint/CPU regressions and input-preservation checks.
- [x] Exercise the full G0 matrix, including restricted/blocked cases, missing evidence, unavailable GPU and unresolved exposure/lineage/literature.
- [x] Satisfy M0-T26 with actual local evidence when execution is authorized; synthetic evidence alone cannot satisfy this item.
- [x] Verify all M0-T01 through M0-T26 have pass/fail/skip evidence and requirement traceability; skipped required checks cannot yield unrestricted readiness.
- [x] Review the master definition of done and report unresolved issues. Release M1 only for the allowed engineering/research scope.

Deliverables: runner/verifier, evidence index, G0 evaluator/report, synthetic integration, regressions and real-run handoff. No M1 code or scientific training/search is included in M0.6.

## Progress evidence

- Source identity: HEAD `153e5bae0aa9dfccfb7a14dd44035f925b441433` plus dirty source snapshot `a9c1652d0920391306acde691b9718f002d480de595353717b3ccc5cd96ca43d`. Commit alone does not identify the tested implementation. The snapshot precedes these final acceptance-note edits; current non-document source identities were independently checked afterward.
- Full CPU-only command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 167 tests passed, no skips, in 87.369 seconds. Evidence: `artifacts/research/m0_6_checks/cpu-20261107T130420Z/summary.json` and `logs/full-suite.stderr.log`. Source hashes remained unchanged during this run.
- Full CUDA-environment CPU regressions: the same suite passed all 167 tests, no skips, in 87.856 seconds during the real M0 run. Actual CPU and CUDA synthetic environment checks passed separately; GPU observed as NVIDIA GeForce RTX 5070 Ti, CUDA runtime 12.8. This is operational correctness evidence, not a deployment latency result.
- Real local command: `.venv313/Scripts/python.exe tools/run_m0.py --config artifacts/research/m0_6_checks/m0-real-20261107T130421Z/config.json --run-id m0-real-20261107T130421Z`. The locating config uses absolute resolved inputs and requests CPU plus CUDA. It is ignored local evidence, not a tracked machine-specific default.
- Real run: `artifacts/research/m0/m0-real-20261107T130421Z/`. See `acceptance/evidence_index.json`, `gate_g0.json`, `M0_REPORT.md`, `m1_handoff.json`, `receipt.json` and `tests.json`. All 26 acceptance IDs pass with current log/real-run traceability. 231 indexed artifacts verify; repeated verification preserves both hashes and mtimes. The run is completed/validated, with requested pilot exit 1 because scientific readiness is blocked, not because evidence generation failed.
- Observed audit: 7,909 supported images = 7,784 included + 125 exact decoded-pixel duplicate exclusions; 81 full filename groups. No audit label/path/hash blockers. All 7,909 original image hashes and the complete supported file set were independently checked again after execution; inputs were not modified. Counts are observations, not enforced published totals or verified-patient counts.
- Reverification evidence: `artifacts/research/m0_6_checks/m0-real-20261107T130421Z/reverification.json` records both full suites, all acceptance cases, original-image hash checks, 123 current non-document source files and read-only repeated verification. Developer verification outputs stay outside the immutable run graph.
- New coverage: 18 `tests/test_m0_runner.py` cases plus two CPU-report/environment regressions. Tests cover ready/restricted/blocked G0, hard-failure precedence, archive relocation, changed/missing/unindexed artifacts, schema declarations, forged ready gates, actual subprocess timeout, operator interruption before audit, immutable historical metadata and CLI error codes. Existing M0.1-M0.5 contracts and CLI/checkpoint regressions pass in both environments.
- Repairs found by reverification: CPU execution in the CUDA wheel could crash while querying cuDNN/GPU name with CUDA hidden. CPU runtime and environment reports now leave GPU-only fields explicitly unqueried; CUDA metadata is collected by the separate CUDA check. Earlier blocked runs are retained rather than overwritten.
- Reviewer/date: Codex implementation/self-review and real local acceptance, 7 November 2026. CI workflow includes the synthetic tests/archive upload; hosted CI has not been claimed as executed here.
- G0 outcome: engineering **ready**, GPU **ready**, exploratory **blocked**, confirmatory design **blocked**, novelty review **restricted**. M1 CPU/interface development and frozen-inventory diagnostics are allowed. Real scientific fits and untouched confirmation remain blocked pending attributable history, lineage scope, named reviewers, agreed policies, fold support and closest-method review. M0 completion does not approve those facts or claims.
- [x] Tests passed.
- [x] Reviewed; tracker updated and master M0 completion checklist reconciled.
