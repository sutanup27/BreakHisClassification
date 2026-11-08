# Milestone 1 experiment-plan alignment review

Date: 8 November 2026. Outcome: **Aligned for implementation**.
Review type: specification/source self-review, not scientific approval or proof
of implementation. Master: [M1_SPEC.md](M1_SPEC.md). Progress: [tracker](README.md).

## Sources and interpretation

Reviewed the current [roadmap](../../research/EXPERIMENT_ROADMAP.md),
[global tracker](../../research/MILESTONE_TRACKER.md),
[experiment guide](../../guides/EXPERIMENTS.md),
[config reference](../../guides/EXPERIMENT_CONFIG.md) and
[scientific guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md),
especially objectives O1-O3 and sections 8-10, 12-18. Inspected experiment/config,
manifest/loader, training/checkpoint, metrics/comparison source and existing tests
at HEAD `6b59b00`; current root CLI help was checked without running experiments.

The agreed roadmap has five research milestones. M1 is the reproducibility
foundation; it must not absorb vulnerability analysis, algorithm development
or final-study inference into a generic reporting task. The scientific guide's
larger grouped-fold and replication ambitions remain later work, narrowed by the
agreed BreaKHis university scope. They are not mandatory blockers for M1.

## Objective and plan coverage

Coverage here describes what the **specification** covers. It is not a claim
that the required code/tests already exist.

| Objective/plan step | Requirements | Packages | Deliverable and acceptance | Coverage | Gap/action |
| --- | --- | --- | --- | --- | --- |
| M1 config-driven reproducible runs; guide section 18 | M1-R01, M1-R02, M1-R03 | M1.1 | Frozen inputs, safe outputs and attributable configuration; M1-T01-M1-T04 | Covered | Implement phase-aware records and close preflight/publication gaps |
| Saved group-separated data; guide section 8 | M1-R04, M1-R05 | M1.2 | Full-source linkage/support checks; M1-T05-M1-T07 | Covered | Existing checks need consolidated cross-magnification/component verification |
| Prevent test feedback; guide section 9 | M1-R06, M1-R10, M1-R11 | M1.2-M1.4 | Test-access tripwires, development default, bound final inference; M1-T08, M1-T12, M1-T15, M1-T16 | Covered | Current runner automatically tests and current loaders construct all roles |
| Matched reproducible fitting; guide section 10 | M1-R07, M1-R08, M1-R09 | M1.3 | Seeds, exact resume, committed history and timing/steps; M1-T09-M1-T14 | Covered | Broaden Adam/group-selection/tie/history recovery coverage |
| Separate initialization/loader/augmentation experimentation | M1-R03, M1-R07 | M1.1, M1.3 | One recorded fitting seed plus independent manifest split seed | Partial | Reproducible combined seeding is sufficient now; independent factor studies can extend config later |
| Group aggregation and basic outcomes; guide sections 9 and 16 | M1-R12, M1-R13 | M1.4 | Hand-computable metrics and complete metadata/probabilities; M1-T17, M1-T18 | Covered | Validate undefined rates and add subtype/component attribution |
| Comparable ordinary experiments; guide sections 10, 11 and 14 | M1-R09, M1-R14 | M1.3, M1.4 | Recorded costs/factors, matched invariants and phase tables; M1-T19, M1-T20 | Covered | Provides comparison plumbing; does not establish equal-budget scientific controls |
| Basic deployment measurement; guide section 17 | M1-R15 | M1.4 | Reproducible scoped resident-forward measurement; M1-T21 | Covered | Full-session replication, end-to-end timing and memory claims remain M5 |
| End-to-end rerunning and readable artifacts | M1-R16, M1-R17 | M1.5 | Fresh-process integration, legacy read behavior and accurate docs; M1-T22-M1-T25 | Covered | All new phase-aware acceptance remains to be performed |
| O1: malignant vulnerability, guide section 12 | M1-R06, M1-R13 support future work | M1.2, M1.4 provide inputs | Development-only per-image/group outputs with linked metadata | Deferred | Actual perturbation/damage/stability analysis is milestone 3 |
| O2: candidate mechanism and A-E comparisons, sections 13-14 | M1-R09, M1-R14 support future work | M1.3, M1.4 provide interfaces | Role isolation, recorded budgets and declared comparison factors | Deferred | Algorithm, class weighting, constraints, search and ablations are milestone 4 |
| Competitive dense/compact/scratch controls | M1-R16 provides execution plumbing | M1.5 | Existing dense/pruning integration only | Deferred | Competitive fits, additional controls and recipes are milestone 2 |
| O3: folds, uncertainty, replication and repeat timing, sections 15-17 | M1-R11, M1-R13, M1-R15 provide inputs | M1.4 | Attributed fixed-model inference and export | Deferred | Final-study design/statistics/replication are milestone 5 |
| Multiclass, clinical validation and external dataset deployment | None | None | No unsupported claims or invented implementation | Out of scope | Require separately agreed future scope; not acceptance conditions here |

## Master-to-package completeness

All 17 requirement IDs have an owning package and one or more acceptance cases.
All 25 acceptance IDs have exactly one primary package owner. Later integration
repeats selected checks rather than taking ownership away from their packages.
The sequence covers inputs -> data roles -> development/resume -> evaluation ->
full integration. No orphan master requirement or circular dependency was found.

## Consistency decisions and corrections

- Development becomes the default; final is a separate explicit invocation.
  Old automatic-test behavior is documented as a current gap, not relabeled safe.
- Config/manifest schema 1 remain; new phase/run/report records use schema 2.
  Old runs are not silently promoted to development-only or final-ready.
- Train/validation/test remain the existing roles. No extra calibration role,
  invented outer fold or mandatory source-history investigation is introduced.
- Benchmark seed/input/protocol are explicit and separate from fitting RNG;
  final CPU inference does not require the original training GPU.
- Histories and predictions are required stage artifacts, not optional files
  ignored by recovery. Resume checks include best/tie state and actual used LR.
- Comparison keeps data/measurement invariants fixed, but exposes deliberately
  varied factors. A selection-policy experiment is not rejected simply because
  it intentionally changes a method factor; the difference must be declared.
- Group-balanced reporting and checkpoint selection are not described as
  group-balanced training, vulnerability-aware pruning or proven novelty.
- Parent checkpoint storage and current transform settings are retained.
  Code source maps/ordinary logs suffice; no retired approval or archive system
  is reintroduced.

The earlier experiment guide describes the implementation that currently exists.
This spec deliberately targets changes to that behavior. Until implemented, its
proposed flags are labelled proposed; M1.5 updates user-facing commands after they
work. This is a documented target/current gap, not an unresolved plan conflict.

## Remaining decisions and implementation readiness

No unresolved scientific choice prevents M1.1 implementation. Future method
weights/tail fractions, sensitivity margins, compact models, outer folds and
second-dataset choices remain with milestones 2-5. Their absence does not block
this infrastructure. Component/role support is checked from actual data when used.

Local document links, ID ownership and proposed/current command distinctions are
checked before delivering the spec package. No model fit or new regression
pass is claimed from this review. Existing test/experiment results remain baseline
context only; the package tracker stays uncompleted until actual acceptance.

## Implementation alignment after M1.3

M1.1-M1.3 now cover frozen inputs, full-source linkage/role isolation and
validation-only development with exact committed-epoch recovery. No test quality
is computed implicitly, no scientific pruning mechanism is added, and the fixed
baseline/pruning recipes remain unchanged. The current package tracker records
108 CPU tests and generated-image fresh-process checks; this supplements the
original planning review above. M1.4 still owns bound final inference/comparison;
M1.5 still owns complete public-workflow integration acceptance. Research
novelty and converged experiments remain later-milestone work.

## Implementation alignment after M1.4

Explicit final inference now verifies fixed development selections before test
access, needs only the evaluation device and does no fitting/pruning. Saved
comparisons preserve phase/data/policy/measurement invariants while declaring
supported experimental factors and budgets. Primary group aggregation remains
equal magnification means, with image-mean grouping secondary. No threshold
search, new pruning method or research novelty is inferred from these outputs.
CPU/CUDA tests and generated-image VGG phase workflows are recorded in M1.4;
M1.5 still owns integrated public-workflow acceptance. Later research milestones
remain unchanged.

## Milestone 1 integration alignment

M1.5 verifies the existing foundation through public fresh-process dense,
unstructured and channel-pruning workflows, including genuine failures/retries,
unavailable-role images, fixed final weights and saved phase comparisons. It does
not change the experiment's split, aggregation, threshold, backbone scope or
research objectives. Legacy records are read or refused without rewriting them.
The 25-case master matrix now links implementation tests; package/global trackers
record actual acceptance and next scientific work. No new approval gate or archive
system is introduced. Competitive baselines, vulnerability analysis, the proposed
mechanism and final-study inference remain milestones 2-5.

Final M1 acceptance: all five packages reviewed; 135 CPU cases (133 passed, two
CUDA-only skips) and the retained public workflow passed. Documentation and the
master/package/global trackers agree. No M1 acceptance case remains outstanding;
scientific milestones 2-5 remain unchanged and uncompleted.
