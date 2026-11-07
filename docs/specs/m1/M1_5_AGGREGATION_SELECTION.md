# M1.5 — Group aggregation, metrics and selection

Status: Not started. Depends on: M1.1-M1.4 reviewed.
Master mapping: WP5-WP6; [M1 specification](M1_SPEC.md), sections 5.5-5.6.
Progress: [tracker](README.md).

## Build checklist

- [ ] Implement proposed magnification-balanced primary and image-weighted secondary group pooling with versioned policies.
- [ ] Keep available-only missing-magnification handling, representative policy and per-group support explicit; no imputation or duplicate inflation.
- [ ] Implement hand-checkable group/image metrics with explicit units, class support and null/reason handling for undefined statistics.
- [ ] Define AUROC/Brier/AP versus PR-area semantics where used; preserve legacy percent-accuracy compatibility.
- [ ] Separate fixed reference, calibration, epoch selection and candidate selection artifacts and permitted roles.
- [ ] Implement finite declared grid/floor/tie policy and full feasibility records; infeasible remains infeasible.
- [ ] Preserve the distinction between benign-on-tie legacy argmax and proposed malignant-on-tie threshold >= rule.
- [ ] Require actual agreed policies for real use; keep synthetic values and proposed numerical choices from becoming scientific approvals.

## Target and completion gate

Deliverable: pooling/metric and development selection interfaces, not a new pruning/search method.

- [ ] M1-T20-T22: hand pooling, missing/duplicate/mixed-label groups, confusion/rank/Brier and absent-class cases.
- [ ] M1-T23-T25: threshold ties/boundaries, undeclared calibration policy and infeasible floor without relaxation.
- [ ] M1-T26: distinct epoch/candidate policies, declared reuse and outer-test traps.
- [ ] Verify policies, probabilities and denominators are attributable and the M0 scientific restrictions remain unchanged.
- [ ] No real threshold fitting, test-informed tuning or clinical/noninferiority claim is performed.

## Progress evidence

- Source identity: Pending.
- Commands/results/logs and acceptance references: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Real floors, margins, grids, ties and policy agreements remain unresolved.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M1.6 dependency released.
