# M1.5: End-to-end acceptance and user documentation

Acceptance status: Not started. Existing smoke workflows/tests are available.
Depends on: M1.1-M1.4. Master: [M1_SPEC.md](M1_SPEC.md), sections 5, 7 and 8.
Owns: M1-R16, M1-R17. Tests: M1-T22, M1-T23, M1-T24, M1-T25.
Progress: [tracker](README.md).

## Goal and deliverables

Verify that all packages compose into the simple public workflow. Targets:
`tools/smoke_experiments.py`, relevant end-to-end tests, README,
`docs/guides/EXPERIMENTS.md`, `EXPERIMENT_CONFIG.md`, testing/checkpoint guides,
this package tracker and the [global tracker](../../research/MILESTONE_TRACKER.md).

## Implementation checklist

- [ ] Update generated-image smoke tooling to run development first, then explicit final evaluation, then separate phase-specific comparisons.
- [ ] Cover dense, masked and channel-pruned/fine-tuned runs in fresh processes with tiny CPU fixtures and parent-directory checkpoints.
- [ ] Verify frozen input bytes, repeated-seed dense weights, reports, predictions, complete histories and the absence of automatic test outputs.
- [ ] Re-run relevant existing regression coverage; validate actual failure/retry paths rather than only the happy path.
- [ ] Add legacy schema-1 read/compare cases without rewriting old artifacts, fabricating phase metadata or silently promoting a completed old run to final-ready.
- [ ] Keep explicit individual train/evaluate/prune/fine-tune commands usable; document that the high-level runner now defaults to development.
- [ ] Replace guide examples with supported phase-aware commands only after implementation, covering first setup, fresh rerun, resume, final evaluation and comparison.
- [ ] Preserve checkpoint locations, clear metric units and the distinction between implementation verification and scientific findings.
- [ ] Rebuild the existing offline HTML docs and update package/global status from actual results; retain concise normal logs/results rather than a new evidence platform.

## Verification checklist

- [ ] M1-T22: dense generated-image development/final/comparison runs in fresh processes with complete outputs and a test-access tripwire before final.
- [ ] M1-T23: mask/channel invariants survive save/reload/fine-tuning; role identity and histories remain consistent through both phases.
- [ ] M1-T24: old checkpoints/reports remain readable where supported; unsafe resume/promotion is refused without modifying the originals.
- [ ] M1-T25: CLI help, examples, local links, generated HTML and both trackers agree with implemented behavior.
- [ ] Integration repeats M1-T08, M1-T10, M1-T12 and M1-T15-M1-T20 as applicable; every master acceptance case has real evidence or a clearly identified outstanding condition.

Current regression entry: `python -m unittest discover -s tests -v`.
Current smoke entry: `python tools/smoke_experiments.py`; its implementation must
be updated before it can serve as evidence for this new contract.
Docs build: `python tools/build_docs.py`.

## Completion and handoff

Milestone 1 can be marked Complete only when the master acceptance matrix and
end-to-end path are verified and documentation is accurate. Conditional GPU
checks are disclosed; CPU execution checks do not prove GPU correctness or
scientific accuracy. No new full-data fit is required merely to finish this
infrastructure milestone. Milestone 2 then establishes competitive models and controls.

## Progress evidence

Not executed against this contract: specification only. The prior 63-test suite,
synthetic smoke and real one-epoch run remain context; none is retroactively
relabeled as evidence that the new phase-aware integration already passes.
