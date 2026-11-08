# M2.5: Development frontier, quality review and M3 handoff

Implementation/experiment status: Not started.
Depends on: M2.1-M2.3 plus M2.4 executor fixtures for software; M2.4 real outputs for
scientific acceptance. Master: [M2_SPEC.md](M2_SPEC.md), sections 5-9.
Owns: M2-R13, M2-R14, M2-R15. Acceptance: M2-T19-M2-T22.
Progress: [tracker](README.md).

## Goal and deliverables

Produce verified matched/contextual comparison tables, a complete descriptive
sensitivity/specificity/CPU-latency frontier, baseline-strength review and usable
trained-parent handoff. Targets: saved-comparison extensions and report adapter,
plots, public docs/tests and study `development/` outputs. Build this software with
fixtures before M2.4's main fits; complete its empirical review afterward.

## Implementation checklist

- [ ] Extend saved comparisons for baseline kind, topology/init lineage and shared/incremental/lifecycle budgets; keep data/role/policy/timing invariants strict.
- [ ] Emit matched subsets with controlled recipes and separate contextual model-family strata. Declare seed/model/recipe/width/ratio differences; do not loosen invariants through an unrestricted vary flag.
- [ ] Export all valid candidates plus failed/infeasible/undertrained diagnostics. CPU and CUDA profiles are separate outputs; no inference or image reads during report.
- [ ] Implement the master's three-objective dominance rule, keep tied/nondominated identities, and plot sensitivity/latency with specificity and support visible. Preserve zero/negative gains.
- [ ] Review real train/validation curves, absolute rates, best-versus-last epochs, optimization behavior and recipe opportunity across families. Record limitations instead of asserting competitive quality from a fixed epoch count.
- [ ] Check all required compact/random/scratch/continuation/mask controls and seed-repeat results, accounting for extra recovery/discovery work and family-specific capacity. Do not pick a favorable subset of ratios or test outcomes.
- [ ] Save parent hashes/selected epochs/seeds, development image/group probabilities, subtype/component identities, selected indices/widths and complete costs as the M3 starting inputs.
- [ ] Add public fresh-process VGG/compact/control integration and resume tests, retain M1 regressions, and update README/config/experiment/testing/checkpoint docs plus existing HTML build.
- [ ] Update master/package/global trackers from actual software and real-run evidence. Leave M2 incomplete for missing real controls, unresolved weak parents or resource-limited scope; no synthetic claim of O1/O2/O3.

## Verification checklist

- [ ] M2-T19: saved-report model/fit/pixel tripwires; missing/changed artifacts and mixed profiles/phases refuse; differing declared recipes appear only in contextual strata.
- [ ] M2-T20: hand-computed dominance, ties, missing class support, zero/negative speedup and failure fixtures produce correct frontier and full point/diagnostic records.
- [ ] M2-T21: real baseline-strength review, complete controls/repeat matrix and accessible parent/prediction/width/cost handoff; M3 can reconstruct parents and join groups without interpreting a test report.
- [ ] M2-T22: public small generated workflow covers compact, shared parents, mask/L1/random/scratch/continuation and explicit final compatibility; M1 regression suite, current help, examples/links/HTML/trackers all agree.

## Completion and handoff

M2 completes only after the code and real baseline/control evidence pass the master
contract. M3 owns new layer/channel perturbation damage and stability analysis.
M4 owns the proposed algorithm, class weighting, constrained selection, adaptive/
established competitors and decisive ablations. M5 owns final frozen inference,
paired uncertainty, timing-session replication and bounded research conclusions.

## Progress evidence

Not executed: specification only. No real frontier, quality review, competitive
baseline, significant speedup or vulnerability finding is claimed.
