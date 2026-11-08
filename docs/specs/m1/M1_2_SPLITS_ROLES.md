# M1.2: Saved splits and role-isolated data access

Acceptance status: Reviewed. Full-source metadata checks and role-isolated loaders implemented.
Depends on: M1.1. Master: [M1_SPEC.md](M1_SPEC.md), section 4.1.
Owns: M1-R04, M1-R05, M1-R06. Tests: M1-T05, M1-T06, M1-T07, M1-T08.
Progress: [tracker](README.md).

## Goal and deliverables

Make split checks consistent and ensure development cannot accidentally construct
or iterate test data. Targets: `breakhis/manifest.py`, `breakhis/data.py`,
experiment preflight, CLI loader calls, manifest/loading tests and role-tripwire tests.

## Implementation checklist

- [x] Revalidate full filename identifiers including suffixes, raw/pixel hashes, labels and stored component metadata for the complete selected source before magnification filtering.
- [x] Verify transitive filename/duplicate linkage and one split per component across all magnifications; reject inconsistent components instead of trusting stored IDs.
- [x] Preserve split assignments for a fixed manifest/seed; reject preparation overwrite and keep fitting-seed changes independent of split creation.
- [x] Check unique case-safe relative paths, known metadata, supported roles and class support. Use real SHA-256 values in synthetic fixtures testing the production manifest contract.
- [x] Validate/read pixels only for requested roles. Development can inspect test metadata for integrity but must work without test image files being accessible.
- [x] Add requested-role parameters to existing dataset/loader helpers and pass roles explicitly from the new runner/CLI adapters.
- [x] Train/fine-tune use train/validation; validation reports use validation; explicit final inference uses test only.
- [x] Preserve transform policies and raw-byte verification on access; retain clear filename-group terminology and actionable corrupt/changed-image errors.

## Verification checklist

- [x] M1-T05: same-seed preparation reproduces assignments; a different fitting seed does not alter them; an existing split cannot be overwritten.
- [x] M1-T06: reject filename, raw/pixel and transitive-component leakage, including a collision visible only outside the chosen magnification.
- [x] M1-T07: reject malformed metadata, conflicts, missing selected-role classes/files and changed bytes; final checks test support without forcing test access in development.
- [x] M1-T08: use factory and pixel-access tripwires to prove development never constructs/iterates a test dataset or reads its images. Confirm final constructs only its requested evaluation role.

Intended tests: `test_manifest.py`, `test_manifest_loading.py`,
`test_preprocessing.py` and focused experiment-role cases. No real image fitting
is needed to prove these invariants.

## Completion and handoff

M1.3 receives validated manifest metadata and train/validation-only loaders.
M1.4 can request a test-only evaluation loader explicitly. Group balancing in
reporting is not represented as group-balanced training or vulnerability scoring.

## Progress evidence

15 focused tests in `tests/test_m1_roles.py` cover M1-T05-M1-T08, including
real CLI training/evaluation with a tiny model and unavailable-role file tripwires.
The full CPU suite passed 92 tests; log: `artifacts/m1-2-tests.log`.
Real 40X preparation and read-only revalidation succeeded with run ID
`m1-2-role-check-20261108`, without fitting or inference.

Stored group IDs are stable labels: recomputed membership must match their
partition, but a label may name a representative excluded during deduplication.
No manifest is silently relabeled. Filename groups do not establish clinical identity.

CLI routing is explicit: train/fine-tune request train/validation, prune requests
validation, evaluate requests `--split validation|test` (default test), and compare
requests test. At M1.2 acceptance, higher-level phase execution remained pending. M1.3 now
defaults the runner to validation-only development; final execution remains M1.4.
