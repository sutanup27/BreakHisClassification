> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M0.1 records API and schema binding

Implementation: `breakhis/research/records.py`. Contract: [M0 master spec](../../../specs/m0/M0_SPEC.md).
Delivery checklist: [M0.1](M0_1_RECORDS_INTEGRITY.md). This is an implemented
foundation API, not an M0 runner, source capture, audit, or readiness evaluator.
Importing the research package performs no experiment/IO and imports no PyTorch.

## Record envelope

Generated records use strict schema version 1. A noncollection has exactly:

```json
{
  "schema_version": 1,
  "record_type": "run",
  "run_id": "m0-example",
  "created_at_utc": "2026-11-07T00:00:00Z",
  "data": {"status": "created"}
}
```

Collection types replace `data` with `items`, a list of objects. The registry
`RECORD_SHAPES` defines object/collection envelopes for the planned record types.
This is the concrete envelope binding for master section 5: object payload fields
described there live under `data`; collections use `items`. Unknown root fields,
record types, schema versions and non-UTC timestamps are rejected.

`validate_record` validates envelope/JSON structure only. It does not establish
that an audit is complete, a source snapshot exists, or G0 is ready. Later packages
must validate each record's scientific/domain payload and referenced evidence.

| Function | Contract |
| --- | --- |
| `new_record(record_type, run_id, payload, created_at_utc=...)` | Selects data/items envelope, validates and detaches caller-owned mutable payload |
| `validate_record(record, expected_type=...)` | Strict root/type/version/UTC/JSON validation; returns record without creating outputs |
| `validate_fact(fact)` | Validates explicit knowledge status/value/reason/evidence references, not their scientific truth |
| `read_json(path)` | UTF-8/BOM-aware strict JSON; rejects duplicate keys, malformed input and nonfinite values |
| `read_record(run_dir, relative, expected_type=...)` | Confined read plus envelope validation |

Explicit material unknowns use `{value: null, status: unknown, reason: text,
evidence_refs: []}`. Verified/self-reported facts require a value and reference;
unknown/disputed facts require a reason. `validate_fact` must be called by domain
validators for material assertions; optional JSON nulls alone do not imply an
eligibility state. No missing fact is filled with an invented default.

## Content identity

| Function | Identity |
| --- | --- |
| `file_sha256(path)` | SHA-256 over actual file bytes; use safe_child first for confined reads |
| `canonical_json_bytes(value)` | UTF-8, sorted object keys, compact separators, finite JSON only |
| `semantic_sha256(value, domain=...)` | Hash of version/domain/content; separate namespaces do not share identity |
| `content_id(namespace, identity)` | Stable 64-character content ID, independent of row position |
| `snapshot_sha256(entries, policy)` | Sorted portable path/SHA pairs plus capture policy; ignores per-entry root/timestamp metadata |

This canonical encoding is deliberately defined here, not claimed as RFC 8785.
List order matters unless a specialized helper sorts it; numeric `1` and `1.0`
retain their JSON representations. Snapshot entries must be unique, including
case-folded path collisions, and use lowercase SHA-256. Capture policy is part
of identity and must itself be root/timestamp-independent in M0.2.
Semantic image-inventory field selection remains M0.3 work.

## Paths and atomic writes

| Function | Contract |
| --- | --- |
| `portable_relative_path(value)` | POSIX relative artifact path; no traversal, drives/UNC, streams, ambiguous separators, reserved Windows names or trailing dot/space aliases |
| `safe_child(root, relative, must_exist=...)` | Resolve inside trusted root; reject symlinks/junctions below it, even internal links |
| `reject_overlap(output, protected_roots)` | Reject both containment directions against data/protected source roots |
| `create_run_directory(output_root, run_id, protected_roots=...)` | New/empty destination only; never delete/empty prior content |
| `capture_relative_path(path, repo_root=..., output_root=...)` | Validate an individual source file; prohibit outside inputs/self-capture |
| `write_record(run_dir, relative, record, overwrite=False)` | UTF-8/fsynced temporary sibling, atomic no-clobber publication; explicit overwrite for intentional state updates |

Run roots must first be configuration-validated; these generic helpers do not
independently identify a caller's dataset. Full source inclusion/exclusion policy
and capture consistency checks remain M0.2. Path containment covers static
filesystem state, not hostile concurrent directory replacement.

No-clobber publication uses a same-filesystem hard link; unsupported filesystems
fail rather than use a racing check-then-replace fallback. Explicit overwrite uses
atomic replacement. Temporary files are cleaned on failed flush/publication and
prior destination bytes are preserved. A Windows junction fixture tests real
link rejection without requiring symbolic-link privilege; directory junctions
are removed without following their targets.

## Pre-run configuration

`validate_m0_config(config, config_file=...)` validates a separate input schema
without creating outputs/running checks. Required keys:

| Keys | Required values / semantics |
| --- | --- |
| `schema_version`, `spec_id`, `spec_version` | `1`, `M0-SPEC-001`, `1.0` |
| `repo_root`, `data_root` | Existing directories; data root supplies the configured source inventory location for later adapters |
| `output_root` | Within resolved repository `artifacts/research/m0`, nonoverlapping with data |
| `sources` | Exactly `["cancer_binary"]` for M0 version 1 |
| `audit_workers`, `audit_seed` | Positive integer workers; integer seed in 0 through 2^32-1; booleans rejected |
| `check_devices` | Unique nonempty CPU/CUDA list containing `cpu` |
| `baseline_checks` | All four allowlisted identities: imports, cli_help, environment, regressions |
| `lineage_path`, `exposure_path`, `protocol_path`, `literature_path` | Nonempty governance input paths, resolved but not yet ingested |
| `mode` | pilot or confirmatory_design; does not authorize an experiment |
| `timeouts` | Exactly environment_seconds, regression_seconds, audit_seconds; positive integer seconds |

Unknown/missing keys, arbitrary commands, invalid modes/versions and unsafe output
paths are rejected. Native config paths resolve relative to the configuration
file, not cwd, and can use ordinary parent navigation to locate the repo/data.
Portable relative artifact paths never allow `..`. Relative config outputs are
normalized and then checked against the repository artifact anchor.
Governance files need not exist during structure validation; M0.4-M0.6 must ingest
and verify them before granting readiness. No runnable M0 CLI/template is added
in this package.

## Evidence-index interface foundation

`artifact_entry` creates exact-byte metadata with these keys: `path`, `record_type`,
`schema_version`, `sha256`, `bytes`, `operation_id`, `status`. Raw text/binary files
use null schema version; versioned JSON uses a registered record type and version 1.
Statuses are complete, partial or failed. Metadata creation does not prove payload
validity; verification checks it separately.

`verify_artifact` checks safe path, size, hash, optional envelope and run identity.
Incomplete status is rejected by default; explicitly allowing partial evidence
does not change its status or make it scientifically eligible.
`verify_evidence_index` verifies entries in an evidence_index envelope, rejects
duplicate/case-colliding/self-index paths, and returns a count/run identity.
It is read-only and never regenerates invalid artifacts. An empty index can be
structurally valid but grants no readiness; completeness, dependency cycles beyond
self-indexing, required artifacts and G0 evaluation remain M0.6 responsibilities.

Errors use `RecordError` for malformed records/config/paths and `IntegrityError`
for missing/changed/incomplete evidence. Underlying IO errors during publication
are preserved. No error is converted into a successful empty record.

## Test coverage and review limits

`tests/test_research_records.py` covers M0-T05/M0-T21/M0-T22 foundation requirements:
unsafe output/paths, actual/mock links, source self-capture, malformed versions,
explicit unknowns, semantic/file hash distinctions, no-clobber races, interrupted
writes, tampering/missing evidence and read-only verification. M0.6 still needs
integrated gate/runner coverage before those master acceptance cases are complete.
