# M1.1: Configuration, storage and run identity

Acceptance status: Reviewed (implementation/self-review, 8 November 2026).
Depends on: none. Master: [M1_SPEC.md](M1_SPEC.md), sections 3 and 5.
Owns: M1-R01, M1-R02, M1-R03. Tests: M1-T01, M1-T02, M1-T03, M1-T04.
Progress: [tracker](README.md).

## Goal and deliverables

Establish validated inputs and phase-aware run/output contracts without changing
the scientific recipe. Primary targets: `breakhis/experiments.py`,
`breakhis/paths.py`, `breakhis/cli.py`, root runner argument parsing and
`tests/test_experiments.py`. Reuse JSON/config/path/checkpoint utilities.

## Implementation checklist

- [x] Verify schema-1 config fields against the master; keep examples valid and make malformed scalar types fail cleanly.
- [x] Add phase-aware preparation validation: development requires config; final requires saved run ID; reject unsupported combinations and record effective overrides.
- [x] Define schema-2 phase statuses, results index and complete artifact-reference structure for later packages; do not implement test inference here.
- [x] Centralize run/report/checkpoint path checks; reject input/output aliases, traversal, escaping links, existing fresh IDs and source/output overlap.
- [x] Preserve parent `checkpoint/binary` placement and avoid introducing weights under the codebase.
- [x] Save exact frozen manifest bytes, resolved config, digests, split/fitting seeds, source map and runtime; handle absent Git metadata without inventing a commit.
- [x] Validate inputs before creating outputs where possible; retain a clear failed state for publication failures after run reservation.
- [x] Reuse finite atomic JSON/checkpoint publication; specify required secondary artifacts so later stage recovery can verify them all.

## Verification checklist

- [x] M1-T01: exercise valid config/overrides, unknown keys, duplicate keys, invalid booleans/types, nonfinite values and phase combinations; no partial run for preflight rejection.
- [x] M1-T02: occupied ID, path traversal, links and aliases cannot overwrite data, input configs/checkpoints or another run.
- [x] M1-T03: inject a write/publication failure; confirm no truncated successful output and an incomplete stage.
- [x] M1-T04: modify frozen config/manifest/source identity and remove Git availability; distinguish drift failure from honest unavailable Git metadata.

Intended current test entry: `python -m unittest discover -s tests -p test_experiments.py -v`.
Additional tests: `python -m unittest discover -s tests -p test_m1_setup.py -v`.
`--phase` and `--prepare-only` now support preparation/validation. Phase execution
is explicitly refused until M1.3/M1.4; the old no-phase execution path is unchanged.

## Completion and handoff

Validated config, safe paths, stable identity records and phase-state contracts
are usable by M1.2/M1.3. New-run outputs cannot silently replace an existing run.
Passing config tests does not complete split isolation, resume or final evaluation.

## Progress evidence

- Source: HEAD `c4b368e` plus current implementation changes. The real preparation
  records exact package hashes in `inputs.json`; no source archive is required.
- Implementation: `experiment_setup.py`, shared `experiment_io.py`, path guards,
  config/phase validation, output-alias checks and honest unavailable-Git metadata.
- Targeted: 14 new `test_m1_setup.py` tests passed, alongside the existing config
  tests. Full command: `.venv-stage2-cpu/Scripts/python.exe -m unittest discover -s tests -v`.
- Full regression result: **77 tests passed, zero skips, 48.245 seconds**.
  Logs: `artifacts/m1-1-tests.stdout.log` and `artifacts/m1-1-tests.stderr.log`.
- Real config preparation and read-only revalidation succeeded for run
  `m1-1-preparation-check-20261108-final`, with `--device cpu --phase development --prepare-only`
  (and then `--resume`). Frozen config/manifest, seeds, digests, runtime and expected
  outputs were saved. Both phases remain `not_requested`; no weights, fits or
  predictions were created by that preparation.
- M1.2 role isolation, M1.3 phase execution/recovery and M1.4 final inference remain
  outstanding. Shared publication protects static paths, not hostile concurrent
  filesystem replacement. Storage failure records are best-effort if storage itself
  remains permanently unavailable. This package does not complete milestone 1.
