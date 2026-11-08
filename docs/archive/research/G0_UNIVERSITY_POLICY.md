# Historical G0 policy for the university pilot

Optional technical reference. Normal experiments use the
[experiment guide](../../guides/EXPERIMENTS.md) and do not require this gate.

Policy profile: `university-pilot-v1`, 7 November 2026.

The [pilot protocol](../../research/UNIVERSITY_PILOT_PROTOCOL.md) records the researcher's
declared dataset source, prior use and fresh-workflow decision. The opt-in
configuration is `configs/research/m0.university.json`, with its explicit
component scope in `configs/research/exposure.university.json`.

## What G0 now means

| Condition | University pilot treatment |
| --- | --- |
| Dataset origin declared by researcher | Accept for scoped pilot; original-release byte verification is not claimed. |
| Earlier experiments occurred | Retain prior-use uncertainty; allow a fresh workflow without historical untouched claims. |
| Working protocol and owner declared | Use the reviewed, hash-bound pilot declaration; the historical full-protocol draft need not be promoted. |
| Literature unfinished | Reporting limitation; does not block this pilot. Complete before novelty claims. |
| Full confirmatory design or fold support unfinished | Remains blocked for confirmatory design. Actual pilot partitions still require role/support validation. |
| Corrupt/conflicting data, failed baseline, inconsistent source or incomplete evidence | Hard blocker. |
| Missing/invalid/unbound component scope or declaration | Hard blocker; no automatic opt-in. |
| GPU unavailable when selected for training | Hard blocker for the requested GPU pilot; CPU engineering remains separate. |

Successful pilot scope evaluation is **restricted**, not unrestricted scientific
approval. `real_fit_scope_allowed` means the declared components may support an
exploratory pilot after downstream split/role checks. It does not construct a
loader, launch training, approve arbitrary components or authorize outer-test
feedback. The report lists `PILOT_PARTITION_VALIDATION_REQUIRED` explicitly.
M1.1 still denies real images/fits; a later tested pilot execution adapter is
required. M1.2 must establish feasible component-disjoint roles with both classes.

## Generate and verify a fresh report

From the codebase in the existing CUDA virtual environment:

```powershell
python tools/run_m0.py --config configs/research/m0.university.json --run-id university-pilot-NEW-ID
python tools/run_m0.py --config configs/research/m0.university.json --run-id university-pilot-NEW-ID --verify-only
```

Use a unique run ID. Both CPU and CUDA readiness checks, regressions, data audit
and governance checks execute; no scientific model fitting is launched. Inspect
`artifacts/research/m0/<run-id>/acceptance/M0_REPORT.md` and `gate_g0.json`.

The pilot protocol is referenced by SHA-256 in the exposure input and preserved
with the archive. If its content changes, deliberately update that reference
and use a new run. If inventory components change, review and update the scope;
unknown components cannot silently inherit approval. Scope is bound to audited
component identities, not to a directory name alone.

Historical configurations without the opt-in declaration keep their existing
policy and verification results. Existing archives are never rewritten. Unknown
history stays unknown in exposure records; pilot approval does not relabel it
as reviewed non-exposure or establish verified patient identities.

## Recorded acceptance

Fresh local run: `artifacts/research/m0/m0-university-20261107-v1/`.
Generation completed and verification passed. The actual regression suite ran
191 tests successfully, including four new pilot-policy tests. CPU and CUDA
readiness passed; the real dataset audit completed. The new exploratory status
is `restricted`, with `real_fit_scope_allowed: true` for the 81 declared
components, subject to the listed partition/role preconditions. Confirmatory
design remains blocked. No real scientific fitting or outer-test inference ran.

Independent verification rechecked the new archive and its handoff component
scope, and verified the historical `m0-real-20261107T130421Z` archive without
changing its file hashes or modification times. Verification summary:
`artifacts/research/university_g0_acceptance/verification.json` (ignored local
evidence). These acceptance notes were added after the run's source snapshot;
the tested implementation did not change.
