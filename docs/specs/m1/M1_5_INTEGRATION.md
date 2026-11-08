# M1.5: End-to-end acceptance and user documentation

Acceptance status: Reviewed. M1.5 and the complete Milestone 1 foundation are accepted against the implementation contract.
Depends on: M1.1-M1.4. Master: [M1_SPEC.md](M1_SPEC.md), sections 5, 7 and 8.
Owns: M1-R16, M1-R17. Tests: M1-T22, M1-T23, M1-T24, M1-T25.
Progress: [tracker](README.md).

## Goal and deliverables

Verify that all packages compose into the simple public workflow. Targets:
`tools/smoke_experiments.py`, relevant end-to-end tests, README,
`docs/guides/EXPERIMENTS.md`, `EXPERIMENT_CONFIG.md`, testing/checkpoint guides,
this package tracker and the [global tracker](../../research/MILESTONE_TRACKER.md).

## Implementation checklist

- [x] Update generated-image smoke tooling to run development first, then explicit final evaluation, then separate phase-specific comparisons.
- [x] Cover dense, masked and channel-pruned/fine-tuned runs in fresh processes with tiny CPU fixtures and parent-directory checkpoints.
- [x] Verify frozen input bytes, repeated-seed dense weights, reports, predictions, complete histories and the absence of automatic test outputs.
- [x] Re-run relevant existing regression coverage; validate actual failure/retry paths rather than only the happy path.
- [x] Add legacy schema-1 read/compare cases without rewriting old artifacts, fabricating phase metadata or silently promoting a completed old run to final-ready.
- [x] Keep explicit individual train/evaluate/prune/fine-tune commands usable; document that the high-level runner now defaults to development.
- [x] Replace guide examples with supported phase-aware commands only after implementation, covering first setup, fresh rerun, resume, final evaluation and comparison.
- [x] Preserve checkpoint locations, clear metric units and the distinction between implementation verification and scientific findings.
- [x] Rebuild the existing offline HTML docs and update package/global status from actual results; retain concise normal logs/results rather than a new evidence platform.

## Verification checklist

- [x] M1-T22: dense generated-image development/final/comparison runs in fresh processes with complete outputs and a test-access tripwire before final.
- [x] M1-T23: mask/channel invariants survive save/reload/fine-tuning; role identity and histories remain consistent through both phases.
- [x] M1-T24: old checkpoints/reports remain readable where supported; unsafe resume/promotion is refused without modifying the originals.
- [x] M1-T25: CLI help, examples, local links, generated HTML and both trackers agree with implemented behavior.
- [x] Integration repeats M1-T08, M1-T10, M1-T12 and M1-T15-M1-T20 as applicable; every master acceptance case has real evidence or a clearly identified outstanding condition.

Current regression entry: `python -m unittest discover -s tests -v`.
Current smoke entry: `python tools/smoke_experiments.py`; it now uses fresh public
CLI processes for dense, mask and channel pipelines, both phases and comparisons.
Docs build: `python tools/build_docs.py`.

## Completion and handoff

Milestone 1 can be marked Complete only when the master acceptance matrix and
end-to-end path are verified and documentation is accurate. Conditional GPU
checks are disclosed; CPU execution checks do not prove GPU correctness or
scientific accuracy. No new full-data fit is required merely to finish this
infrastructure milestone. Milestone 2 then establishes competitive models and controls.

## Progress evidence

Focused integration: four cases in `tests/test_m1_integration.py` passed in
126.612 seconds. Log: `artifacts/m1-5-integration-tests.log`.
Retained standalone acceptance also passed: `artifacts/m1-5-public-smoke/summary.json`
and `artifacts/m1-5-smoke.log`. Commands and per-command logs are in that smoke
folder. It creates only generated fixture data, with ordinary experiment records
and parent-directory checkpoint outputs.

M1-T22: all workflow actions invoke public `prepare-data`, `run_experiment.py`
and `compare_experiments.py` in fresh processes. Dense, masked and channel runs
use the same frozen split/config recipe and fitting seed. Development runs with
test image files unavailable; no final directory/results appear implicitly.
Matching dense weights reproduce after a real epoch-zero failure/resume.
Final inference runs with training/validation files unavailable. Saved final
comparison succeeds with all pixel files unavailable. Each phase's reports,
ordered predictions, histories, selection and artifact references validate.

M1-T23: real masked and channel models are saved/reloaded/fine-tuned. Mask tensors
stay identical and masked original weights stay zero; channel metadata/topology
stay consistent and physical parameter count is lower. All roles retain fixed
phase attribution and complete one-epoch histories. Separate individual-command
integration remains in `tests/test_end_to_end.py` and the full suite.

M1-T24: old schema-1 run/report JSON reads without modification. Phase-aware
resume/final/comparison refuse without rewriting originals. Supported version-1
state dictionaries load for inference and refuse exact resume before data access.
Other format/mask compatibility remains in checkpoint/reporting regressions.

M1-T25: public help covers all six command entry points; actual smoke commands
exercise the published prepare/develop/resume/final/compare options. User guides
cover installation, fresh reruns, relative config paths, both resume modes,
legacy compatibility, output locations, units and research limitations. Offline
HTML is rebuilt and local links/anchors checked before final acceptance. CI now
retains the public phase-smoke command/summary/comparison evidence as well as the
existing individual-command smoke outputs. A local pass does not claim hosted CI.

Actual failure/retry evidence: corrupt generated training bytes cause a failed
child and a saved epoch-zero boundary; restoring the exact original bytes allows
`--resume`. Corrupt test bytes fail final; development hashes remain unchanged
and development comparison still works. Final resume succeeds; a completed final
request refuses. Existing M1.2-M1.4 tests re-run role tripwires, exact SGD/Adam
resume, publication failures and final/comparison refusal cases.

No real-data fit is needed for this integration acceptance. The old real one-epoch
run remains execution context, not evidence of converged quality or novelty.

## Final acceptance

The complete CPU suite ran 135 tests in 200.089 seconds: 133 passed, two CUDA-only
cases skipped. Log: `artifacts/m1-5-tests.log`. Every M1-T01-M1-T25 case has owning
checklist evidence and a linked implementation test in the master matrix. The new
four-case focused integration pass and retained public smoke are recorded above.
Markdown file/heading links, generated HTML anchors, CLI help and `git diff
--check` passed. The offline HTML guide is current.

Package training/reporting code is unchanged from M1.4, where all 23 focused cases
passed in the CUDA environment, including real CUDA fit/CPU final portability and
synchronized GPU timing (`artifacts/m1-4-cuda-tests.log`). M1.5 adds public CPU
orchestration acceptance, compatibility tests, documentation and CI evidence
retention; it does not claim a new GPU integration run or hosted CI pass.

Milestone 1 is Complete. Next: specify milestone 2's dense/conventional-pruning
matrix, convergence recipe, compact/scratch and equal-budget controls before
running the scientific baseline study. Research milestones 2-5 remain outstanding.
