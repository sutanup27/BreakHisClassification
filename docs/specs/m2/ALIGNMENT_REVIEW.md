# M2 plan alignment review

Date: 8 November 2026. Outcome: **Aligned for implementation**.
This is a specification assessment, not scientific approval or evidence of a novel
method. M2.1 planning is implemented and reviewed; later implementation
and scientific experiments remain Not started.
Master: [M2_SPEC.md](M2_SPEC.md). Packages: [tracker](README.md).

## Reviewed sources and interpretation

- [Current roadmap](../../research/EXPERIMENT_ROADMAP.md), milestones 1-5 and the baseline/control completion target./
- [Research guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md), sections 1-3, 6-19, especially steps 12-17 and 23-24.
- [Current experiment/config behavior](../../guides/EXPERIMENT_CONFIG.md), [workflow](../../guides/EXPERIMENTS.md), [M1 acceptance](../m1/M1_5_INTEGRATION.md).
- Source: `breakhis/experiments.py`, `experiment_setup.py`, `development.py`, `cli.py`, `models/vgg.py`, `checkpoints.py`, `pruning.py`, `reporting.py`, `phase_artifacts.py`, `saved_comparison.py`; existing example configs and docs builder.
- Current public runner/comparison help was inspected. `run_baselines.py`, config-2 baseline initialization, compact architecture and random ranking are proposed, not existing capabilities.

Latest user scope controls: simple university within-BreaKHis experiments, accepted
local dataset, CPU/RTX 5070 Ti, a 7-9-day project window, retained reproducibility
and leakage controls without elaborate approval/archive requirements. Older guide
ambitions are not silently made M2 prerequisites.

## Objective-to-spec coverage

Coverage below describes the specification, not completed experiments.

| Objective / plan step | Master requirements | Packages | Behavior/evidence to implement | Coverage | Gap/action |
| --- | --- | --- | --- | --- | --- |
| Roadmap M2; guide 12-14: credible dense references | M2-R01, M2-R05, M2-R10, M2-R14 | M2.1, M2.2, M2.4, M2.5 | VGG/ResNet18/compact real fits, bounded recipe opportunity, curves/absolute rates, reproducible artifacts; T01-T02, T07, T14, T17, T21 | Covered | Competitive quality must actually be reviewed; 20 epochs alone is not acceptance |
| Guide 7 and 9: magnification/data/operating point | M2-R02, M2-R09 | M2.1, M2.3 | 40X pilot versus pooled main, same manifest, equal-magnification group reporting, argmax/0.5 tie-to-benign; T03, T13 | Covered | Full fold/threshold-calibration design belongs to later work; do not claim it exists |
| Guide 15: physical surgery | M2-R07 | M2.3 | L1/random same checked VGG reconstruction, exact index/BN/classifier/mask tests; T10-T11 | Covered | No residual/depthwise surgery or compact-model pruning claimed |
| Guide 16: bounded conventional pilot | M2-R01, M2-R10, M2-R11 | M2.1, M2.4 | Fixed ratios, cost profile, equal F and measured CPU/CUDA profiles; T02, T14-T17 | Covered | Conditional resource limits remain measured, not assumed |
| Guide 17: dense/mask/L1/random/scratch/compact minimum controls | M2-R04-M2-R09, M2-R14 | M2.2, M2.3, M2.5 | Same parent per seed, same widths, scratch B+F, continuation, random rank and masks; T05-T13, T21 | Covered | Core controls cannot be silently dropped when unhelpful |
| Guide 17 full comparator list: adaptive/hardware-aware and class-aware methods | M2-R13, M2-R14 provide reusable comparison/cost foundation | M2.5 handoff; M4 future implementation | Current matrix does not test benefit beyond these competitors | Partial | M4 must implement/faithfully adapt established competitors under matched selection/search/recovery budgets before O2 claims |
| Guide 13/23-24: extra training and scratch controls | M2-R03, M2-R06, M2-R08, M2-R12 | M2.1-M2.4 | Fresh optimizer, same starting parent, incremental/lifecycle/discovery costs, B versus F; T04, T09, T12, T18 | Covered | Equal passes do not equal FLOPs/time; tables must not certify fairness automatically |
| O1 / guide 18-20: malignant-group vulnerability | M2-R14 handoff | M2.5 -> M3 | Parents, probabilities, group/component/subtype/width identities allow later paired perturbation analysis | Deferred | M3 still needs signed margin/loss/probability damage, concentrated-harm and across-fit stability analyses |
| O2 / guide 21-24: proposed mechanism, class weighting, constraints, ranking/allocation ablations | M2-R07-M2-R14 supply controls only | M2.3-M2.5 -> M4 | Random ranking/scratch controls do not implement group-balanced malignant scoring or selection constraints | Deferred | M4 owns explicit algorithm, image/group weighting, selection-only and class-weighted controls, allocation ablations and A-E decisive comparisons |
| RQ5 / guide 16-17: measured efficiency versus theoretical reduction | M2-R07, M2-R11-M2-R13 | M2.3-M2.5 | Physical parameters/MACs, synthetic resident latency, slowdowns kept, memory scope and device-specific strata; T11, T16, T18-T20 | Partial | M5 adds independent timing sessions/end-to-end/export measurement for stronger deployment claims |
| O3 / guide 25-27 and statistics | M2-R02, M2-R14 preserve identities/results | M2.5 -> M5 | Pooled development grid and one repeated VGG/control subset are not OOF or paired uncertainty | Deferred | M5 owns frozen final study, selected folds/seeds, paired group/component inference and feasible replication |
| Multiclass, clinical identity/safety, mandatory external dataset, broad backbone surgery | None | Later separately scoped work | No fabricated supported task, clinical verification or second-dataset prerequisite | Out of scope | Bound conclusions to accepted binary within-dataset scope |

## Master-to-package completeness

15 stable requirements and 22 acceptance IDs are assigned exactly once:

| Owner | Requirements | Acceptance |
| --- | --- | --- |
| M2.1 | M2-R01-M2-R03 | M2-T01-M2-T04 |
| M2.2 | M2-R04-M2-R06 | M2-T05-M2-T09 |
| M2.3 | M2-R07-M2-R09 | M2-T10-M2-T13 |
| M2.4 | M2-R10-M2-R12 | M2-T14-M2-T18 |
| M2.5 | M2-R13-M2-R15 | M2-T19-M2-T22 |

Each package has separate implementation/verification checklists and a handoff.
M2.5 report software is built with M2.4 executor fixtures before long fits; its
real quality review depends on completed M2.4 runs. There is no circular scientific
prerequisite or demand to fit models merely to finish this spec package.

## Consistency repairs made during review

1. Kept pooled magnifications as planned main task and 40X as pilot instead of
   copying the current 40X example default into a new scientific conclusion.
2. Added compact, exact-width random, scratch and dense-continuation controls;
   did not equate L1 masks with physical speedup or equal removal ratios with equal latency.
3. Distinguished shared parent input from exact resume. Compatible immutable
   parent lineage can survive code revisions as a new input; exact active-child
   resume/final keeps M1 source identity checks and no legacy phase promotion.
4. Specified experiment schema 2 for new mode fields; existing config schema 1
   remains strict, while the study document has its own schema-1 namespace.
5. Kept unknown capacity fields null for plan/pilot, requiring real values before
   main launch; count 22 required fitting cells plus explicit extra trials/retries.
6. Required fresh numerical/BN state for scratch, fresh optimizer for inherited
   recovery, and both incremental/lifecycle/discovery cost views.
7. Kept data/measurement invariants strict while permitting named contextual
   family-recipe strata. No arbitrary comparison override or automatic fairness label.
8. Defined strict recipe/profile subfields and unambiguous null/required fields
   for every initialization mode; no hidden data/budget override in a recipe.
9. Separated scientific M2 completion from software/synthetic acceptance and kept
   M3's mechanism-motivating analysis out of a vague reporting package.

No unresolved scope conflict remains within the M2 contract. The full research
guide is deliberately broader than M2; deferred mechanisms and evidence are
named above rather than advertised as covered by the baseline matrix.

## Decisions and practical limits

Defaults (20+5 epochs, ratios, seeds, all-from-scratch training, MobileNetV3-Small,
two bounded recipe options and CPU-primary timing) are planning choices, not
validated best settings. M2.4 must measure hardware/capacity, pilot stability and
profile adequacy, then freeze the actual main revision. Weak/undertrained parents
or missing controls keep real acceptance incomplete. A 7-9-day completion estimate
cannot be certified before those measurements.

These are experimental settings to settle during the bounded pilot, not blockers
to implementing M2.1-M2.3. No literature superiority, effect size, clinical margin,
novel method or final test result is invented. No model fitting, regression run,
code implementation or Git commit/push was performed by this specification task.

## M2.1 implementation alignment

The implemented planner preserves frozen split bytes, 40X-versus-pooled role
identity, bounded matrix counts and explicit unresolved parent/topology bindings.
It creates no model or CUDA dependency and reads no test pixels. Draft config-2
envelopes are not represented as runnable configs; future stages refuse. Cost
capacity is conservative and includes optional/pilot/replay work without claiming
measurements or permitting a hidden subset. The full experiment/novelty boundaries
above remain unchanged; M2.1 software acceptance does not complete M2.

M2.1 final evidence: 20 focused tests passed; full CPU suite ran 155 cases
(153 passed, two CUDA-only skips). Real metadata-only preparation, source/artifact
hashes, help and documentation checks passed. See the owning checklist for logs.
No M2 real fitting, cost measurement or scientific baseline claim was made.
