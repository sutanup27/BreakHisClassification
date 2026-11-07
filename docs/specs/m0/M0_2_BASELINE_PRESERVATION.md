# M0.2 — Baseline preservation

Status: Reviewed (implementation/self-review, 7 November 2026). Depends on: M0.1 reviewed. Master mapping: WP1-WP2; sections 5.1-5.2, 6.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

Implemented usage and record details: [baseline API](M0_2_BASELINE_API.md).

## Build checklist

- [x] Implement capture in `breakhis/research/m0.py` using the M0.1 record/hash/path helpers.
- [x] Record HEAD/ref and staged/unstaged/untracked state; capture relevant dirty and untracked source as well as a binary-capable tracked diff.
- [x] Store source inventory and reconstructible snapshots with explicit inclusion/exclusion policy; exclude datasets, environments, weights and generated output.
- [x] Recheck source status/hashes after capture; retry once or fail on inconsistency. Never identify a dirty run by commit alone.
- [x] Record CPU/Python/dependency/venv/pip state and available GPU/runtime/driver metadata; preserve unknown/unavailable fields honestly.
- [x] Capture actual current import/help/environment/regression outcomes with allowlisted argv, cwd, duration, timeout, return code and stdout/stderr logs.
- [x] Keep CPU-only checks functional; distinguish GPU failure from eligibility for CPU engineering.

## Tests and completion gate

- [x] Implement M0-T01 through M0-T04 using temporary Git repositories, staged/unstaged/untracked files and concurrent-source-change fixtures.
- [x] Implement M0-T19 and M0-T20: CPU pass/GPU unavailable, failed check and timed-out command with retained evidence.
- [x] Demonstrate that included source can be reconstructed and hashes checked; no recursive capture or unrelated generated payloads.
- [x] Review verifies that evidence is current and actual. Historical test counts cannot satisfy baseline acceptance; failure fixtures must fail readiness.

Deliverables: source/environment/check capture functions; `baseline/` record contracts and relevant `tests/test_m0.py` coverage. No scientific training or historical test inference.

## Progress evidence

- Source identity: HEAD `5e71ec541ed76819b86b24c8c93081300e106990` plus dirty worktree snapshot `46d5cc6def17ddfabae6d17d13fef334ac747f093d6eaf739839e390713baa84`. The commit alone is not the tested identity. Snapshot includes tested source before these final acceptance notes; `acceptance-summary.json` records exact implementation/test hashes.
- Current local evidence: `artifacts/research/m0/m0-2-acceptance-20261107T102050Z/` (ignored local outputs, not shipped in Git). `baseline/git.json`, `source_inventory.json`, `source_snapshot/`, both patches, device records, `checks.json` and `consistency.json` preserve source and actual execution evidence. `acceptance-summary.json` records the outcome; `verification-summary.json` records independent snapshot/log verification.
- Actual command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`; 86 tests passed, no skips, in 60.094 seconds. Logs: `logs/regressions.stdout.log` and `logs/regressions.stderr.log` within that run. Six CLI help checks, imports, interpreter metadata, pip commands and the existing synthetic CPU tensor/VGG environment check also passed.
- Coverage: 16 `tests/test_m0.py` cases cover M0-T01-T04 and M0-T19-T20, including staged deletion, binary diff, clean reconstruction, dirty source, excluded inputs, retry/failure, post-check drift, unavailable GPU, failed/timed-out regressions, malformed observations and skipped-suite rejection. The prior 70 tests also passed in this fresh run.
- Reviewer/date: Codex implementation/self-review, 7 November 2026; engineering package review, not independent scientific adjudication.
- Restrictions: GPU execution was not requested or validated in this CPU run; actual NVIDIA driver metadata was captured separately. LFS/submodules are recorded without hydration and block self-contained baseline eligibility if present. Full M0 CLI, data/exposure/protocol gates, evidence graph and G0 remain M0.3-M0.6. This is baseline acceptance, not full M0-T26 or scientific readiness.
- [x] Tests passed.
- [x] Reviewed; tracker updated and M0.3 dependency released.
