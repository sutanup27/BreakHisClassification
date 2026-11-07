# M1.3 — Independent seeds, epoch history and resume

Status: Not started. Depends on: M1.1-M1.2 reviewed.
Master mapping: WP2-WP3; [M1 specification](M1_SPEC.md), sections 5.3-5.4.
Progress: [tracker](README.md).

## Build checklist

- [ ] Add explicit split/training/initialization/loader/augmentation/pruning/search seed namespaces and effective-state records.
- [ ] Seed before model creation; use dedicated loader/sampler/augmentation streams without resetting unrelated RNG.
- [ ] Start with workers zero; record determinism/device/environment and reject unsupported worker/resume assumptions.
- [ ] Persist actual epoch history, development-only selection evidence, optimizer/scheduler and checkpoint/config/parent hashes.
- [ ] Extend research checkpoint references with complete RNG/generator state while preserving legacy reconstruction/resume compatibility.
- [ ] Restore supported epoch-boundary state before stochastic work; never overwrite it with default seeding.
- [ ] Refuse conflicting/mid-epoch/incompatible resume or explicitly fork a new run with honest provenance.
- [ ] Keep all weights in the parent checkpoint hierarchy; use temporary equivalent paths for tiny synthetic acceptance fits.

## Target and completion gate

Deliverable: namespaced randomness, attributable epoch history and supported exact CPU resume.

- [ ] M1-T11-T12: controlled CPU repeatability and namespace isolation.
- [ ] M1-T13: uninterrupted versus resumed sample order, RNG, history, weights and optimizer/scheduler state.
- [ ] M1-T14: conflicting/unsupported resume fails without silent fallback.
- [ ] M1-T15: uneven-batch units/weighting, no duplicate epochs and no test access.
- [ ] Confirm CPU work requires no GPU metadata initialization; no real fitting or pretrained downloads occur.

## Progress evidence

- Source identity: Pending.
- Commands/results/logs and acceptance references: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Real seed values/recipes remain unagreed; GPU/cross-version bitwise equality is not promised.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M1.4 dependency released.
