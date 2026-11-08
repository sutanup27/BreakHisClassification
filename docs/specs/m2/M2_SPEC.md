# M2: Dense and conventional pruning baselines

Specification: **M2-BASELINES-SPEC-001**, version 1.0. Date: 8 November 2026.
Document status: Reviewed for implementation. M2.1 planning implemented;
M2.2-M2.5 execution/model/control work and scientific experiments remain pending.
Tracker: [README.md](README.md). Assessment: [alignment review](ALIGNMENT_REVIEW.md).
Plan: [current roadmap](../../research/EXPERIMENT_ROADMAP.md), milestone 2.
Scientific source: [research guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md),
sections 7, 10-11 (steps 12-17), 16-19; later-objective boundaries in sections 12-15.
Prerequisite: [M1 acceptance](../m1/M1_5_INTEGRATION.md).

## 1. Objective and completion boundary

Establish credible dense references and a measured **development-set** detection/
latency frontier for conventional pruning, including controls that distinguish
physical compression, channel ranking, inherited weights and extra training.
Deliver reusable trained parent checkpoints for M3's vulnerability experiments.
This is empirical baseline work plus necessary supporting software, not the
proposed pruning novelty and not a final held-out research conclusion.

M2 has two completion conditions: tested software AND completed, interpretable
real-data baseline/control experiments. A working runner, synthetic scores, one
real epoch or a large test count alone cannot complete M2. If a dense recipe is
undertrained or compact/scratch controls are missing, record partial progress;
do not certify a competitive baseline or silently remove an unfavorable control.

## 2. Scope and interpretation of the sources

Use the accepted local BreaKHis binary dataset at
`D:/Sutanu_WorkSpace/BreakHis/Dataset/dataset_cancer_v1/classificacao_binaria`.
Reuse the saved manifest; never resplit for a fitting seed, architecture or ratio.
No second dataset, archive provenance investigation, human sign-off roles or new
approval service is required. The university scope and five-milestone roadmap
supersede the older guide's mandatory-looking gates, lineage prerequisites,
five-fold/second-dataset ambitions and 8-13-week estimate. The user's 7-9-day
project window calls for measured scheduling, not a claim that all experiments
will fit that time without profiling.

Main planned analysis: `magnification=all` (guide section 7); `40X` is the cheap
execution/cost pilot, not silently the main pooled study. Preserve all four
magnifications' group assignments. If compute forces a 40X-only scientific scope,
record a new study revision and bound its claims; pooled acceptance remains
incomplete until that scope change is explicitly agreed. M1 examples use 40X;
this is a documented pilot/main distinction rather than changing their defaults.

Primary deployment measurement: CPU, batch 1, float32, resident forward.
RTX 5070 Ti is the intended training device and secondary inference target.
Do not describe a desktop GPU/ROI classification experiment as clinical or edge
validation. Current preprocessing remains resize-short-edge 256/crop 224,
ImageNet normalization, random crop/flip on train only.

### Owned and deferred scientific responsibilities

| Topic | M2 responsibility | Later owner |
| --- | --- | --- |
| Dense/compact strength and conventional frontier | Train, diagnose convergence and retain negative results | M2 |
| Magnification/group weighting | Keep M1 group evaluation; image-weighted cross-entropy fitting and conventional L1 ranking | M3 analyzes group damage; M4 compares image/group-balanced importance |
| Random ranking, inherited versus scratch widths, extra epochs | Implement and measure these controls now | M4 uses them in decisive comparisons |
| Vulnerable malignant-group damage, margins, tail stability | Preserve predictions, subtype/component identities and trained parents; no vulnerability score yet | M3 / O1 |
| Selection-only constraints, malignant weighting, adaptive/class-aware competitors | Reserve clearly identified comparator slots; do not claim current group-BA selection implements these | M4 / O2, guide sections 13-14 |
| Adaptive allocation/ranking ablations and candidate-search accounting | Fixed uniform allocation now; record future budget-compatible interfaces | M4 / O2 |
| Paired uncertainty, folds, repeated timing sessions, replication | Save paired identities and bounded resident measurements | M5 / O3 |

## 3. Existing implementation and concrete gaps

Inspected current source, not assumed capabilities:

| Reuse | Gap owned here |
| --- | --- |
| `breakhis/experiments.py`, `experiment_setup.py`, `development.py` | Existing schema-1 configs only; each pruning run retrains its parent. Add explicit control initialization and shared-parent execution without breaking M1. |
| `breakhis/models/vgg.py` | Custom eight-convolution VGG accepts eight widths internally; expose fresh narrow-topology fits. It is not VGG16. |
| `breakhis/checkpoints.py` | VGG/ResNet18 reconstruction and portable state dictionaries; add one compact backbone and random-channel metadata validation. |
| `breakhis/pruning.py` | L1 masks, uniform/map L1 VGG channel surgery; add seeded random retained indices and reuse the same surgery. Residual/depthwise surgery stays unsupported. |
| CLI fit/epoch publication | Fresh optimizer/schedule, histories, exact workers-zero resume; distinguish initialization from resume and shared from incremental cost. |
| `reporting.py`, `phase_artifacts.py`, `saved_comparison.py` | Correct role reports and timing; extend verification/rows to baseline kind, external parent bindings, scratch topology and actual budgets. |
| `tools/smoke_experiments.py`, M1 tests/docs | Retain the complete M1 workflow and add control/study integration. |

No recipe/model is presently proven competitive by M1 acceptance. Existing
20-epoch examples and 5-epoch recovery are starting defaults, not measured
convergence or published benchmark matches. No implementation change, model fit
or Git operation is authorized merely by this specification.

## 4. Study matrix and configuration contract

### 4.1 Small study plan, not a second protocol platform

Implemented M2.1 plan file: `configs/baselines/m2_study.json`, schema 1, with exact
fields below. This does not enable experiment config-2 fitting yet.
Reject unknown/duplicate fields, bool-as-int, nonfinite values, duplicate job IDs,
ratios/seeds outside range, invalid dependencies and unsafe/occupied outputs.
Resolve input paths relative to the study file; freeze effective inputs before
launch. Do not place proposed fields inside current schema-1 experiment configs.

| Field | Type/default | Meaning |
| --- | --- | --- |
| `schema_version` | integer 1 | Baseline-study document, distinct from experiment/run formats |
| `study_id` | M1-safe identifier | New study revision; no overwrite |
| `experiment_template` | path | Existing six-section experiment config to inherit and validate |
| `pilot_magnification` | `40X` | Execution, epoch-time and memory pilot |
| `main_magnification` | `all` | Main within-dataset baseline matrix |
| `primary_seed` | uint32, 102 | Full fixed compression grid and recipe development |
| `repeat_seed` | distinct uint32, 103 | Fixed 0.50 VGG/control repeat; not another independent group sample |
| `ranking_seed` | uint32, 202 | Local random-channel generator, independent of fitting/split seeds |
| `ratios` | distinct fractions, `[0.25,0.50,0.75]` | Uniform removal requests; record rounded realized widths/sparsity |
| `repeat_ratio` | fraction in ratios, 0.50 | Prespecified smaller repeated matrix; never choose the best first-seed ratio |
| `dense_epochs` | integer B, initially 20 | Fixed recipe-development budget; not an early-stop assertion |
| `recovery_epochs` | integer F, initially 5 | Same fresh adaptation budget for L1/random/mask/dense-continuation |
| `compact_model` | `mobilenet_v3_small` | One proposed compact comparator, added in M2.2 |
| `recipe_candidates` | ordered 1-2 recipe overrides | Defaults: existing Adam/1e-4/weight-decay 1e-4 with `none`, then the same with `cosine`; no test-driven additions |
| `evaluation_profiles` | CPU plus optional CUDA profiles | Fixed M1 timing fields; CPU batch 1 is required; CUDA separately measured |
| `max_fit_jobs` | positive integer or null before main launch | Count every parent, recipe trial, scratch/recovery job and retry from epoch zero |
| `available_wall_hours` | positive finite number or null before main launch | Actual available compute declared before launch; no fabricated hardware estimate |

Each recipe candidate is an object with exactly `optimizer`, `learning_rate`,
`weight_decay`, `scheduler`; use M1 types/ranges (`adam|sgd`, finite positive LR,
finite nonnegative decay, `none|cosine`). No data/seed/epoch/batch overrides hide
inside recipe candidates. List order breaks exact validation ties.
Each evaluation profile has exactly `profile_id`, `device`, `cpu_threads`,
`batch_size`, `warmup`, `repeats`, with M1 scalar constraints and a unique M1-safe
profile ID. Require one primary CPU profile first; allow one separate CUDA profile.
No profile changes image shape, float32 precision, synthetic seed or metric policy.

Batch size 32, workers 0, deterministic true and two CPU threads inherit from the
working template unless the pilot requires a documented change. OOM must not
silently change the batch size/precision for one comparator. Replan all affected
matched cells in a new revision. Use float32; no mixed precision addition in M2.

The plan records every generated config/job/dependency and expected cost before
scientific launch. `max_fit_jobs` and `available_wall_hours` may be null for preparation/pilot and
are supplied before main launch; preparation
can compile/count without a wall-time estimate, but launching the main matrix
requires a populated cost estimate and explicit capacity values. This is ordinary
resource validation, not scientific approval. Never silently launch a subset
while describing it as the complete planned study.

### 4.2 Required initial matrix

All fits are from random initialization unless the row explicitly inherits the
same trained VGG parent. No ImageNet download or unequal hidden pretraining.
ResNet18 is a dense context comparator; physical surgery is VGG-only.

| Cell | Primary seed 102 | Repeat seed 103 | Incremental fitting |
| --- | --- | --- | --- |
| Dense VGG parent | B epochs | B epochs | Once per seed/recipe/source/split |
| Dense ResNet18 | B epochs | Not mandatory in initial repeat | Contextual standard backbone; report budget difference |
| Compact MobileNetV3-Small | B+F epochs | Not mandatory in initial repeat | Practical dense alternative; full-lifecycle epoch allowance |
| Uniform L1 channels + recovery | Each ratio | Fixed repeat ratio | F epochs from bound parent's pruned weights |
| Random channels at identical layer widths + recovery | Each ratio | Fixed repeat ratio | F epochs; matching realized widths/MACs, different retained IDs |
| Narrow VGG from scratch | Each L1 width vector | Fixed repeat ratio's width vector | B+F epochs, all parameters and BN state fresh |
| L1 masked weights + recovery | Each ratio | Fixed repeat ratio | F epochs; nominal ratio is a mask control, not matched structural cost |
| Dense continuation | Once per seed | Once | F epochs from exactly the same selected parent |

Cell counts: primary seed = 3 dense references + 12 ratio/control fits + 1 dense
continuation; repeat seed = 1 parent + 4 fixed-ratio controls + 1 continuation.
This is 16 primary-seed fits plus 6 repeat-seed fits = **22 fitting jobs** for one
recipe per family, excluding pilot/alternative recipes/retries. Count actual
recipe-development fits additionally; reuse selected parents instead of charging
another fit when beginning candidate jobs. Pruned-before-recovery reports require
inference but are not additional fitting jobs. The primary grid is descriptive;
do not compare equal percentages as if they were equal physical compute/latency.

All ranking/selection uses development data only. Fixed ratios may be infeasible
or slower; retain their records. A two-recipe cap is an equal opportunity per
model family, not evidence that arbitrary defaults are strong. Try the second
recipe only under the declared resource/quality rule, before candidate comparison;
choose on validation group BA, lower validation loss, then earlier recipe order.
Freeze the chosen recipe per family before its control matrix. Record differences
across families rather than forcing a misleading matched-recipe table.

Use train/validation curves, best epoch versus B, absolute sensitivity/specificity,
training versus validation loss and observed optimizer behavior to assess strength.
No invented accuracy target or clinical margin. If still improving at budget end,
or optimization fails, record `undertrained`/`unstable` and keep the scientific
acceptance incomplete. A larger B/new recipe is an explicit bounded replan, not
exact resume with a changed schedule. The fixed B/F numbers are adjustable pilot
starting values; they must be frozen for each scientific matrix revision.

### 4.3 Minimal proposed experiment extension

Keep existing experiment schema 1 and M1 commands unchanged. **Proposed schema 2**
adds exactly one top-level `baseline` object to the six existing fields; existing
schema-1 configs never acquire guessed initialization semantics.

| `baseline` field | Allowed values / rules |
| --- | --- |
| `kind` | `standard`, `prune_from_parent`, `dense_continue`, `narrow_scratch` |
| `parent_selection` | null or path to completed parent `development/selection.json`; binds its designated dense checkpoint |
| `channels` | null or eight positive integer VGG widths; fresh topology only |
| `topology_checkpoint` | null or verified L1-pruned checkpoint defining those widths; numeric weights must not initialize scratch |
| `ranking` | null, `l1` or `random`; random allowed only for channel pruning |
| `ranking_seed` | null or uint32; required for random ranking only |

`standard`: pruning null, all other baseline fields null, existing random
initialization; model may be VGG, ResNet18 or the new compact model.
`prune_from_parent`: parent required, pruning required; ranking L1 or random;
no channels/topology override; fitting seed/source/split/recipe must match the
parent. Unstructured masks require L1. `dense_continue`: parent required,
pruning null, other fields null, fresh fit for F epochs.
`narrow_scratch`: VGG only, parent selection/channels/topology checkpoint required,
pruning/ranking/ranking seed null. The topology must descend from that same VGG
parent/seed; parent selection records topology lineage only, with no inherited
numerical state. Its fit budget is B+F. Reject meaningless combinations before model creation.

Freeze parent-selection/checkpoint/topology hashes and origin IDs at preparation.
External references use only declared, validated parent-run/checkpoint roots in
`inputs.json`; arbitrary path roots cannot be supplied at verification time.
Do not forge a completed train stage, optimizer boundary or history for a parent
that was not trained in the child. Read parent histories by verified reference;
record a completed `parent` acquisition stage with zero child fitting steps.
A verified schema-2 parent is an immutable input to a NEW fit, not exact parent
resume: retain its original code/runtime identity even if current child code is
newer, provided its saved format/policies are supported and every referenced
artifact validates. Never relabel its origin as current or promote schema-1 run
history. Child exact resume/final keeps M1's own source-drift checks unchanged.

Run/checkpoint formats can retain version 2 with additive baseline attribution
and mode-specific expected stages. Extend strict readers deliberately, preserving
M1 rejection of unsupported legacy runs and existing config-1 behavior. Selection
bindings, predictions, final inference and artifact verification must understand
the baseline kind and external references before any real parent fit is launched.

Roles: standard/scratch use `dense`; pruning uses referenced `dense`, `pruned`,
`finetuned`; continuation uses referenced `dense` and `finetuned`. Designated role
is fixed to `finetuned` for recovery/continuation and `dense` otherwise. Report
`baseline.kind`, model and lineage separately: a scratch dense role is not the
large dense parent. Child reports may re-evaluate the parent with child attribution,
but must bind the exact original checkpoint and charge that evaluation cost.

### 4.4 Interfaces: current versus proposed

Current, verified via help: `run_experiment.py --config ... [--resume]`, explicit
`--phase final`, `compare_experiments.py --phase development|final`, and individual
`python -m breakhis train/evaluate/prune/finetune` commands.

**Implemented in M2.1 (planning only):**

```powershell
python run_baselines.py --study configs/baselines/m2_study.json --stage prepare
```

**Later stages remain proposed and currently refuse execution:**

```powershell
python run_baselines.py --study configs/baselines/m2_study.json --stage pilot
python run_baselines.py --study configs/baselines/m2_study.json --stage parents
python run_baselines.py --study configs/baselines/m2_study.json --stage candidates
python run_baselines.py --study configs/baselines/m2_study.json --stage candidates --resume
python run_baselines.py --study configs/baselines/m2_study.json --stage report
```

No study stage invokes final testing. Pilot and full jobs have different IDs and
magnification identities. Preparation is read-only except new plan/config output;
`--resume` requires an existing stage and identical frozen settings. Launch jobs
sequentially by default; no concurrent model training during timing measurements.
Existing final inference remains an explicit later operation, extended to recognize
new model/control metadata; M2 acceptance does not require opening real test pixels.

## 5. Algorithms, fairness and measurements

### Parent reuse and conventional controls

1. Validate frozen manifest/recipe and completed parent selection before fitting.
2. Checkpoint hash, topology, classes, initialization policy and seed identify the
   parent. Never resume its optimizer or mutate its files for a child job.
3. For each uniform ratio s and convolution l: keep
   `k_l=max(1, C_l-round(C_l*s))`, matching existing rounding semantics.
4. L1 uses sum of absolute output-filter weights, stable original-index ties;
   random uses a private seed-local permutation, selects the same k_l and sorts
   retained indices. Record seed, IDs and widths; do not reset global fitting RNG.
5. Reuse one checked surgery primitive to copy convolution inputs/outputs, BN
   parameters/buffers and final classifier columns. Source parent stays unchanged.
6. A scratch control builds the same widths with fresh convolution/classifier,
   fresh BN affine/running state, optimizer/schedule and fitting RNG. It copies
   no parent/pruned tensors, gradients, masks, optimizer or RNG state.
7. Recovery/continuation reset optimizer and schedule with the same declared F,
   LR, loss, batch/data-pass policy and validation selection. They are new fits,
   not exact parent resume. Exact child retries still use M1 completed boundaries.

All training uses ordinary image-weighted cross-entropy, not group sampling or
malignant weighting. Validation checkpoint criterion stays group balanced accuracy,
lower image loss, earlier epoch. All primary/secondary group aggregation, threshold
0.5/argmax ties to benign, null-rate handling and identity fields remain M1 policies.
No sensitivity-preservation constraint or claimed safe tolerance is introduced.

### Resource accounting

Store shared parent preparation cost once per actual parent; each child records
its parent reference and zero repeated parent-fit cost. Report BOTH incremental
recovery and full lifecycle cost. Scratch/compact receive B+F passes; pruning and
dense continuation incur B parent plus F recovery passes. Equal epochs/data passes
are not equal GPU FLOPs, time or model capacity; disclose actual steps, image
presentations, elapsed train/validation/pruning/report seconds and selected epochs.
Count all recipe trials, failed candidates and replayed work in study spend;
failed jobs never become quality rows. A reused stage is not another fit.

The selected parent may be earlier than B although B epochs were executed. Report
selected epoch separately from executed cost, including discarded fitting work.
Scratch architecture discovery/selection can consume parent/development work;
report it separately and offer both shared-study and full-discovery cost views.
Do not label scratch a free architecture oracle or claim exact time matching.

Use M1 resident-forward-synthetic-v1, seed-zero local input RNG, float32, input
shape, hash, warmup, samples and actual runtime settings. Initially use CPU batch 1,
50 warmups/200 repeats from the template; pilot checks adequacy and freezes any
change for all measured controls. CUDA is a separate profile, synchronized, never
mixed into CPU rows. Multiple devices/profiles do not require retraining: derive
profile-specific report IDs without mutating the scientific fit identity.

Record measured device memory where available: CUDA allocated/reserved peak bytes
with defined reset/synchronization boundaries; CPU RSS only if a supported method
is implemented, otherwise null with a reason. Do not infer memory from parameters
or claim RSS/process memory equals tensor allocation. Log epoch wall time and
peak OOM behavior before expanding from 40X to pooled data. Independent timing
sessions, end-to-end timing, energy and export/runtime claims remain M5.

## 6. Outputs and failure behavior

Study outputs: `artifacts/baselines/<study-id>/` with frozen `study.json`, `jobs.json`,
`cost_estimate.json`, `status.json`, resolved config files and `development/`
`baselines.json`, `baselines.csv`, `frontier.json`, plots and `quality_review.md`.
These are small ordinary records, not another approval/evidence system.

Each fit keeps M1 `artifacts/experiments/<run-id>/` outputs, histories, selection
and complete prediction envelopes. All model weights remain under
`../checkpoint/binary/<magnification>/<architecture>/<run-id>/`; multiclass remains
separate and unsupported for fitting. No weights/data/environments in Git.
Additional timing profiles use `benchmarks/<profile-id>/` inside the study with
checkpoint/config hashes and protocol; no model-selection feedback from test.

M2 comparison must extend, not bypass, M1 invariants. Compare baseline kinds and
inherited/scratch initialization as explicit method factors; retain immutable
manifest, magnification, grouping, class order, preprocessing, threshold and timing
profile. Different declared family recipes appear as contextual strata, not a
matched causal result; produce matched subsets separately. Verify every artifact
and real completion status before reading a row. Never infer or fit in `report`.

The development frontier is multiobjective: maximize malignant sensitivity and
specificity while minimizing CPU median latency, with balanced accuracy displayed.
A dominates B iff it is no worse on all three and strictly better on at least one;
equal points coexist and keep their identities. Save all points and failure/
infeasibility diagnostics, not just the frontier. Retain zero/negative speedup.
No statistical significance or clinical noninferiority is implied by this frontier.

OOM, missing/mismatched parents, corrupt artifacts, partial epochs and blocked
budgets produce failed/incomplete records with actionable logs. Reuse only verified
completed stages; restore original inputs or create a new revision after changes.
Changing the plan/recipe/epoch count never rewrites a finished or resumable job.

## 7. Requirements and ordered ownership

| Requirement | Contract | Owner | Acceptance |
| --- | --- | --- | --- |
| M2-R01 | Strict frozen pilot/main study plan with bounded explicit job matrix | M2.1 | M2-T01, M2-T02 |
| M2-R02 | Saved split/role/policy identity and explicit method/recipe factors | M2.1 | M2-T03 |
| M2-R03 | Parent dependencies, fit-job counts and honest capacity estimates | M2.1 | M2-T04 |
| M2-R04 | Backward-compatible initialization/config/lineage and parent reuse | M2.2 | M2-T05, M2-T06 |
| M2-R05 | Reconstructible compact dense model and fresh narrow topology | M2.2 | M2-T07, M2-T08 |
| M2-R06 | Fresh child optimizer/RNG versus exact resume are distinguished | M2.2 | M2-T09 |
| M2-R07 | L1/random matched-width surgery and mask controls are correct | M2.3 | M2-T10, M2-T11 |
| M2-R08 | Equal declared adaptation and full-lifecycle scratch/continuation controls | M2.3 | M2-T12 |
| M2-R09 | M1 phase selection/references/final compatibility cover new controls | M2.3 | M2-T13 |
| M2-R10 | Public development-only study execution and stage/epoch retries | M2.4 | M2-T14, M2-T15 |
| M2-R11 | Measured pilot timing/memory and bounded core/repeat matrix | M2.4 | M2-T16, M2-T17 |
| M2-R12 | Per-parent, incremental, lifecycle and study-total actual spend | M2.4 | M2-T18 |
| M2-R13 | Saved matched/contextual baseline comparisons and full descriptive frontier | M2.5 | M2-T19, M2-T20 |
| M2-R14 | Real baseline strength/control completeness and M3 handoff | M2.5 | M2-T21 |
| M2-R15 | Public integration/docs/status distinguish code from experiments | M2.5 | M2-T22 |

Path: M2.1 matrix/recipes -> M2.2 initialization/models -> M2.3 conventional
controls -> M2.4 pilot/real matrix -> M2.5 frontier, quality and handoff.
Build M2.4's executor and M2.5's report/integration software with fixtures before
long scientific fits; then execute M2.4 and finish M2.5's real-data review. This
avoids repeatedly changing active fit code. Exact run resume/final remains source-
bound; verified parent-as-new-input lineage is a distinct operation.

## 8. Acceptance matrix

M2-T01-M2-T04 now have focused implementation evidence in M2.1; later cases
remain planned. See each package and tracker for actual regression/review status.

| ID | Requirements | Fixture/experiment and observable outcome | Owner |
| --- | --- | --- | --- |
| M2-T01 | M2-R01 | Invalid/duplicate/nonfinite plans, unsafe IDs and unsupported combinations fail before output/training | M2.1 |
| M2-T02 | M2-R01 | Deterministic expansion produces 22 core/repeat fitting jobs for one recipe per family, stable IDs and no implicit subset | M2.1 |
| M2-T03 | M2-R02 | Changed split/magnification/phase/policy cannot masquerade as a matched trial; test-loader/pixel tripwires hold | M2.1 |
| M2-T04 | M2-R03 | Recipe/pilot/retry costs included, repeated parent charged once; capacity shortfall refuses launch while plan-only remains usable | M2.1 |
| M2-T05 | M2-R04 | Config-1 behavior preserved; proposed config-2 modes validate; legacy artifacts not silently migrated | M2.2 |
| M2-T06 | M2-R04 | Corrupt/different parent selection/weights/classes/seed refuses; compatible parent origins stay attributable across revisions, children never mutate/refit them | M2.2 |
| M2-T07 | M2-R05 | Compact model forward/backward/checkpoint reload preserves predictions/classes; no hidden pretrained download | M2.2 |
| M2-T08 | M2-R05 | Scratch widths match source topology, but weights/BN/optimizer are fresh; same fitting seed reproducible | M2.2 |
| M2-T09 | M2-R06 | New continuation starts fresh optimizer; interrupted child resume matches its uninterrupted fit without pretending to resume parent | M2.2 |
| M2-T10 | M2-R07 | Toy tied-L1 and seeded-random selections retain exact matched counts, propagate channels/BN/classifier and preserve source/RNG | M2.3 |
| M2-T11 | M2-R07 | Save/reload and masked momentum/BN invariants; physical count/MAC reduction for channels, no promised dense MAC reduction for masks | M2.3 |
| M2-T12 | M2-R08 | L1/random/mask/continuation use F; scratch uses B+F; selected versus executed epochs and changed batches cannot be hidden | M2.3 |
| M2-T13 | M2-R09 | Full control development/explicit final binding on generated images, external references verified, no automatic real test evaluation | M2.3 |
| M2-T14 | M2-R10 | Public prepare/pilot/parents/candidates/report on generated fixtures, no test factory or pixels anywhere in study stages | M2.4 |
| M2-T15 | M2-R10 | Real child error/OOM simulation/interrupt/publication retry leaves honest state; completed parents not retrained or mutated | M2.4 |
| M2-T16 | M2-R11 | 40X then pooled execution/cost profile records device, timing, support and defined memory; unknown CPU memory is null | M2.4 |
| M2-T17 | M2-R11 | Required real primary/repeat cells complete or are explicitly incomplete; no discarded slow/low-quality setting | M2.4 |
| M2-T18 | M2-R12 | Hand-counted shared/recovery/replayed steps, image presentations and elapsed costs agree with logs; full discovery cost visible | M2.4 |
| M2-T19 | M2-R13 | Comparison loads no model; wrong hashes/phase/profile/hidden recipe differences reject or separate contextual strata | M2.5 |
| M2-T20 | M2-R13 | Hand-computable dominance/ties/slowdowns, missing support and failed candidates produce all points plus correct frontier/diagnostics | M2.5 |
| M2-T21 | M2-R14 | Real curves, absolute detection rates, compact/random/scratch/continuation controls and seed repeat reviewed; trained parent/prediction/width handoff is usable by M3 | M2.5 |
| M2-T22 | M2-R15 | Public fresh-process compact/VGG/control workflow, M1 regressions, help/examples/links/HTML and trackers agree; no scientific completion from smoke alone | M2.5 |

## 9. Completion and practical decisions

M2.1-M2.3 may be software-reviewed before real fitting. M2.4-M2.5 require measured
runs as well as tests. Keep software status and experiment completeness separate.
The M2 handoff contains dense parent hashes/seeds/selected epochs, validation image/
group probabilities and subtype/component IDs, conventional selected IDs/widths,
matched control tables, costs and known weaknesses. M3 performs new perturbation/
vulnerability analysis from that handoff; it is not hidden in an M2 reporting job.

Proposed defaults here make coding possible without another scientific approval
round. Before main launch, the measured pilot must supply available compute,
per-family epoch estimates, usable batch size, recipe/epoch choices and fixed
benchmark profiles. If resources cannot support all controls or baseline strength,
record the limitation and an explicit revision; do not invent a schedule, clinical
margin, literature superiority or completed novelty. A second dataset and broad
architecture surgery are not prerequisites for this within-BreaKHis M2 study.

### M2.1 concrete planning handoff

`jobs.json` stores a prospective fit graph and explicit recipe selection nodes.
Default counts: 22 required main cells, three conditional second-recipe parents
and six one-epoch pilot cells. Shared parents are actual single nodes, not copied
fit charges. Each `configs/<job-id>.json` is an envelope with document type
`m2-experiment-draft`, `runnable=false`, a proposed config-2 payload and unresolved
parent/recipe/topology bindings. It cannot be passed directly to M1 execution.
Draft data points to the frozen study manifest; no guessed parent is executed.

`cost_estimate.json` reports all missing/short capacity reasons without trimming
jobs. Optional cost input is bound to the canonical plan digest via
`breakhis.baseline_plan.sha`. A cost bucket reused across seeds/recipes requires
a stated conservative basis covering every such job; the planner itself measures
nothing. Unknown costs remain null. Capacity eligibility is distinct from execution
support, which is false until later packages implement it. See the
[implemented input guide](../../guides/EXPERIMENT_CONFIG.md#m21-study-and-cost-inputs).
