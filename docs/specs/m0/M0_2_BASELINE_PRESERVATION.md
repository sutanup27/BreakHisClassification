# M0.2 — Baseline preservation

Status: Not started. Depends on: M0.1 reviewed. Master mapping: WP1-WP2; sections 5.1-5.2, 6.
Contract: [M0 specification](M0_SPEC.md). Progress: [tracker](README.md).

## Build checklist

- [ ] Implement capture in `breakhis/research/m0.py` using the M0.1 record/hash/path helpers.
- [ ] Record HEAD/ref and staged/unstaged/untracked state; capture relevant dirty and untracked source as well as a binary-capable tracked diff.
- [ ] Store source inventory and reconstructible snapshots with explicit inclusion/exclusion policy; exclude datasets, environments, weights and generated output.
- [ ] Recheck source status/hashes after capture; retry once or fail on inconsistency. Never identify a dirty run by commit alone.
- [ ] Record CPU/Python/dependency/venv/pip state and available GPU/runtime/driver metadata; preserve unknown/unavailable fields honestly.
- [ ] Capture actual current import/help/environment/regression outcomes with allowlisted argv, cwd, duration, timeout, return code and stdout/stderr logs.
- [ ] Keep CPU-only checks functional; distinguish GPU failure from eligibility for CPU engineering.

## Tests and completion gate

- [ ] Implement M0-T01 through M0-T04 using temporary Git repositories, staged/unstaged/untracked files and concurrent-source-change fixtures.
- [ ] Implement M0-T19 and M0-T20: CPU pass/GPU unavailable, failed check and timed-out command with retained evidence.
- [ ] Demonstrate that included source can be reconstructed and hashes checked; no recursive capture or unrelated generated payloads.
- [ ] Review verifies that evidence is current and actual. Historical test counts cannot satisfy baseline acceptance; failure fixtures must fail readiness.

Deliverables: source/environment/check capture functions; `baseline/` record contracts and relevant `tests/test_m0.py` coverage. No scientific training or historical test inference.

## Progress evidence

- Source identity: Pending.
- Test commands/results/logs: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Not assessed.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M0.3 dependency released.
