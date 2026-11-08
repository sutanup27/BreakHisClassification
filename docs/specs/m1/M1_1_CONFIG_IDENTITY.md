# M1.1: Configuration, storage and run identity

Acceptance status: Not started. Existing helpers are available for reuse.
Depends on: none. Master: [M1_SPEC.md](M1_SPEC.md), sections 3 and 5.
Owns: M1-R01, M1-R02, M1-R03. Tests: M1-T01, M1-T02, M1-T03, M1-T04.
Progress: [tracker](README.md).

## Goal and deliverables

Establish validated inputs and phase-aware run/output contracts without changing
the scientific recipe. Primary targets: `breakhis/experiments.py`,
`breakhis/paths.py`, `breakhis/cli.py`, root runner argument parsing and
`tests/test_experiments.py`. Reuse JSON/config/path/checkpoint utilities.

## Implementation checklist

- [ ] Verify schema-1 config fields against the master; keep examples valid and make malformed scalar types fail cleanly.
- [ ] Add proposed phase-aware argument validation: development requires config; final requires saved run ID; reject unsupported combinations and record effective overrides.
- [ ] Define schema-2 phase statuses, results index and complete artifact-reference structure for later packages; do not implement test inference here.
- [ ] Centralize run/report/checkpoint path checks; reject input/output aliases, traversal, escaping links, existing fresh IDs and source/output overlap.
- [ ] Preserve parent `checkpoint/binary` placement and avoid introducing weights under the codebase.
- [ ] Save exact frozen manifest bytes, resolved config, digests, split/fitting seeds, source map and runtime; handle absent Git metadata without inventing a commit.
- [ ] Validate inputs before creating outputs where possible; retain a clear failed state for publication failures after run reservation.
- [ ] Reuse finite atomic JSON/checkpoint publication; specify required secondary artifacts so later stage recovery can verify them all.

## Verification checklist

- [ ] M1-T01: exercise valid config/overrides, unknown keys, duplicate keys, invalid booleans/types, nonfinite values and phase combinations; no partial run for preflight rejection.
- [ ] M1-T02: occupied ID, path traversal, links and aliases cannot overwrite data, input configs/checkpoints or another run.
- [ ] M1-T03: inject a write/publication failure; confirm no truncated successful output and an incomplete stage.
- [ ] M1-T04: modify frozen config/manifest/source identity and remove Git availability; distinguish drift failure from honest unavailable Git metadata.

Intended current test entry: `python -m unittest discover -s tests -p test_experiments.py -v`.
Add focused cases rather than merely checking field names. New flags are proposed
until implemented; CLI help should distinguish supported options then.

## Completion and handoff

Validated config, safe paths, stable identity records and phase-state contracts
are usable by M1.2/M1.3. New-run outputs cannot silently replace an existing run.
Passing config tests does not complete split isolation, resume or final evaluation.

## Progress evidence

Not executed against this package contract: specification only. Existing code
was inspected at HEAD `6b59b00`; the earlier 63-test log is baseline context,
not evidence that M1-T01-M1-T04 are all satisfied. Record actual results here
when implementation/revalidation is performed.
