> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M1.4 — Development predictions and reporting boundaries

Status: Not started. Depends on: M1.1-M1.3 reviewed.
Master mapping: WP4; [M1 specification](M1_SPEC.md), section 5.4.
Progress: [tracker](README.md).

## Build checklist

- [ ] Export image identity, raw/pixel hashes, groups/components, labels, magnifications/subtypes and actual fit/role/fold/seed provenance.
- [ ] Bind predictions to checkpoint/config/assignment/protocol hashes and explicit binary class mapping.
- [ ] Carry sample identity through samplers and batches; validate completeness/uniqueness and reload alignment.
- [ ] Reject invalid logits/probabilities, stale provenance and mixed fits/roles before downstream use.
- [ ] Add validation-only reporting that constructs only authorized development loaders.
- [ ] Add resident benchmark-only entry using a supplied tensor and no labels/test loader; retain timing-region/runtime metadata.
- [ ] Keep legacy test-only evaluate/compare semantics and clearly separate future outer outputs.
- [ ] Validate synthetic stored OOF provenance/completeness without implementing real held-out inference.

## Target and completion gate

Deliverable: aligned development export and validation/benchmark interfaces with enforced outer-access denial.

- [ ] M1-T16-T17: shuffled/partial-batch round trips and malformed/duplicate/stale prediction failures.
- [ ] M1-T18: fault-injected outer constructors, iterators and predictors are never reached.
- [ ] M1-T19: only declared validation roles are read; benchmark uses neither test data nor labels.
- [ ] Verify no real probabilities or held-out labels are collected during engineering acceptance.
- [ ] Existing evaluate/compare/report/checkpoint compatibility remains intact.

## Progress evidence

- Source identity: Pending.
- Commands/results/logs and acceptance references: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Real development prediction generation waits for later permitted fits; outer inference remains M6.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M1.5 dependency released.
