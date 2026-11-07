# M0 runner, verification and Gate G0

The full M0 implementation integrates preservation, current checks, image audit,
exposure governance and the draft protocol/literature records. It never launches
scientific training, pruning search, threshold fitting or real-image test inference.
The existing regression suite includes generated-image training fixtures only.

## Run and verify

From the repository root with the intended virtual environment active:

```powershell
python tools/run_m0.py --config configs/research/m0.example.json --run-id m0-unique-id
python tools/run_m0.py --config configs/research/m0.example.json --run-id m0-unique-id --verify-only
```

Choose a fresh ID for every generation run. Existing nonempty destinations are
refused; there is no resume/overwrite option. Output stays under repository
`artifacts/research/m0/<run-id>/`, separately from datasets and parent checkpoints.
Only the declared binary source is audited; original images, manifests and
checkpoints are never rewritten. A subprocess bounds the audit timeout and
retains partial evidence/logs if killed. CPU checks hide CUDA. Selecting both
`cpu` and `cuda` records separate actual synthetic device outcomes; optional
driver-query failure is not a CPU failure.

CPU reports leave GPU name/cuDNN/availability metadata explicitly unqueried;
the separate CUDA check supplies operational GPU evidence. Ctrl+C records a
failed partial run and does not continue from interrupted baseline checks into
the audit.

| CLI option | Meaning |
| --- | --- |
| `--config PATH` | Required JSON input; relative configured paths resolve against this file's directory |
| `--run-id ID` | Required unique 1-64 character M0.1 ID |
| `--verify-only` | Validate preserved bytes, domain consistency, gate, report and handoff; no checks/audit/inference/repairs or writes |
| `--help` | Describe supported arguments without executing M0 |

Exit **0** means the evidence is valid and the requested mode is eligible
(possibly restricted); **1** means execution/integrity failure or a blocked
requested mode; **2** means invalid configuration/arguments. A valid blocked run
therefore returns 1 even in verify-only mode, while reporting `validated: true`.
`state: completed` means evidence/report generation completed, not scientific
readiness. `state: failed` preserves partial files but cannot pass verification.

The library `breakhis.research.runner.verify_run(path)` returns validated status
and the frozen gate without requiring original metadata/images/environments to
exist. It supports archive relocation using recorded command locations, not the
current machine's checkout path. The CLI additionally needs its locating config
and valid configured roots. To verify a relocated archive directly:

```python
from breakhis.research.runner import verify_run
result = verify_run('path/to/copied/run')
print(result['validated'], result['gate']['overall_status'])
```

Verification checks saved evidence; it does not re-audit current external image
bytes. The producing audit rechecks byte/file-set consistency, and M1 consumers
must reverify input identity before use. Hashes detect changes relative to the
recorded graph; this is not a cryptographic reviewer signature or proof that a
human history declaration is true.

## Configuration

The example config lists the existing M0.1 fields. Two additive optional v1 keys
are now supported: `decisions_path` (default `decisions.initial.json` beside
`protocol_path`) and `run_kind` (`real` default, or explicitly `synthetic`).
Other unknown keys and arbitrary shell commands remain forbidden. The explicit
defaults preserve previous M0.1-M0.5 input compatibility.

| Fields | Contract |
| --- | --- |
| `schema_version`, `spec_id`, `spec_version` | `1`, `M0-SPEC-001`, `1.0` |
| `repo_root`, `data_root`, `output_root` | Git top level, existing binary root, confined research output root |
| `sources` | Exactly `["cancer_binary"]` |
| `audit_workers`, `audit_seed` | Positive worker count and uint32 recorded seed; no research split is generated |
| `check_devices` | CPU required, CUDA optional |
| `baseline_checks` | All of imports, CLI help, environment and regressions required |
| `lineage_path`, `exposure_path` | M0.3 lineage and M0.4 history input files |
| `protocol_path`, `decisions_path`, `literature_path` | Matching M0.5 declarations; decisions sidecar fallback documented above |
| `mode` | `pilot` or `confirmatory_design`; neither launches fits or freezes confirmation |
| `run_kind` | Honest declaration of real versus synthetic input scope; synthetic acceptance cannot stand in for M0-T26 |
| `timeouts` | Positive `environment_seconds`, `regression_seconds`, `audit_seconds` |

Default lineage facts and history remain unknown. Null history instance/root with
empty assertions receives an explicitly unreviewed instance ID tied to the
configured root; this is an operational identifier, not a verified release.
An empty register never approves untouched groups. Nonempty historical assertions
require explicit source identity and metadata root. Original input bytes, spec,
declarations and available hash-matching historical metadata are frozen under
`inputs/`; absent/broken references remain unresolved, never empty successes.

## Explicit exploratory scope

The optional `pilot_scope` in the exposure input is a reviewer declaration with
exactly: `status`, `reviewer`, `component_ids`, `lineage_exception`, `claims`,
`evidence_refs`, `rationale`, `training_device`. Reviewed status requires named
reviewer, actual current component IDs, nonempty rationale/hash-backed metadata
references, device `cpu|cuda`, and claims including `exploratory-only`,
`filename-group-only`, `derivative-only`. Hardware policy and declared device must
agree. Unreviewed is the default. Missing history may permit only an explicitly
reviewed development-only scope; it never produces untouched confirmation.

Known exposures and scope-designated development components are excluded from
confirmatory candidate support. Data conflicts, failed checks, stale source,
missing evidence and pending required scientific policies cannot be waived by
scope text. Unknown clinical identities do not become verified patients; a
filename-group design remains the only supported identity claim.

## Evidence graph and verification

Preserved artifacts include all earlier package outputs, frozen inputs and
`logs/operations.json`. The audit worker records actual argv/cwd, timeout,
duration, return code, logs and input hashes. Source is checked again after
governance. `acceptance/tests.json` traces M0-T01-T25 to successful method lines
in the current regression log and T26 to actual real local orchestration.
Not-run cases prevent unrestricted readiness; historical counts are not reused.

`acceptance/evidence_index.json` covers every nonfinal file with bytes/type/hash,
operation and status. JSONL entries additionally declare `row_schema_version`
and `row_record_type`; unsupported row versions fail. Preserved failed source
attempts are partial artifacts, distinct from an accepted coherent snapshot.

The index excludes itself and the final gate/report/handoff/receipt. Gate and
handoff reference the index digest. `receipt.json` hashes index, gate, handoff
and report afterward, avoiding circular hashes. Verification requires complete
graph coverage, expected schemas/run IDs, source reconstruction, actual command
plan/log consistency, image/exclusion/component accounting, lineage, replayed
metadata-only exposure decisions, protocol bindings and independently derived G0.
It rejects extra files, omitted evidence and a manually edited ready gate even
if its final receipt hash was updated. No invalid evidence is regenerated.

## G0 and M1 handoff

G0 independently reports engineering, exploratory, confirmatory-design, GPU and
novelty-review dimensions. Hard failures override restrictions. Passing CPU
checks may allow CPU engineering while GPU fits are blocked. Unknown lineage,
history, review ownership, numerical policy or closest-method review remains
named and owned. Novelty review completion never establishes novelty.

The report lists observed image/group/test counts, operations, source identity,
restrictions and handoff. `m1_handoff.json` binds inventory/components, exposure,
protocol/decisions/literature versions and allowed work/claims. A blocked pilot
does not prevent independently passing CPU interface development. No M1 code,
outer folds or fits are generated by M0.

The old package checklists retain their original acceptance snapshots. Their
contracts are rechecked through the full current suite and this integrated graph.
CI runs the same synthetic runner tests and retains their archives; real local
data execution is separate and is not represented as hosted CI evidence.
