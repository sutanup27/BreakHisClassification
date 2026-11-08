> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M1.2 — Component folds and inner roles

Status: Not started. Depends on: M1.1 reviewed.
Master mapping: WP1; [M1 specification](M1_SPEC.md), sections 2.1 and 5.2.
Progress: [tracker](README.md).

## Build checklist

- [ ] Build deterministic, versioned stratified assignments over frozen linked components with stable IDs/hashes.
- [ ] Keep full filename suffixes, all magnifications and transitive raw/pixel-linked groups together.
- [ ] Enforce exposure eligibility: exposed groups development-only; unknown history never becomes untouched by reseeding.
- [ ] Define outer and inner role views with zero forbidden group/component/raw/pixel overlap.
- [ ] Reserve separate selection/calibration/optional confirmation support; accept reuse only through an explicit reviewed policy.
- [ ] Report class/subtype/magnification/component support and assignment exclusions; reject infeasible folds without silent reduction.
- [ ] Recheck image identity on loader consumption and preserve old manifests/legacy split behavior.
- [ ] Emit only metadata diagnostics under the current G0; no active real untouched assignment until real-design entry clears.

## Target and completion gate

Deliverable: assignment planner, role views, leakage validator and support report.

- [ ] M1-T04-T06: across-process reproducibility, indivisible components and transitive leakage/conflict cases.
- [ ] M1-T07-T09: eligibility, inadequate support and calibration-role disjoint/reuse cases.
- [ ] M1-T10: changed input bytes rejected before use.
- [ ] Verify same inputs/seed survive row permutations and no original manifest/image is changed.
- [ ] Confirm diagnostic and legacy assignments cannot be passed off as scientific outer folds.

## Progress evidence

- Source identity: Pending.
- Commands/results/logs and acceptance references: Pending.
- Reviewer/date: Pending.
- Open issues/restrictions: Real candidate-untouched support/history remain unreviewed.
- [ ] Tests passed.
- [ ] Reviewed; tracker updated and M1.3 dependency released.
