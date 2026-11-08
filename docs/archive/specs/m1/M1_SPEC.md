> Historical reference. For the active workflow, use the [experiment guide](../../../guides/EXPERIMENTS.md).

# M1 specification: scientific data and reporting infrastructure

Specification: M1-SPEC-001, version 1.0. Prepared: 7 November 2026.
Status: implementation contract; M1.1 foundation is implemented/self-reviewed with actual CPU acceptance. M1.2-M1.6 and scientific experiments have not started.

Parent: [research roadmap](../../research/RESEARCH_IMPLEMENTATION_ROADMAP.md), M1/P1/G1.
Scientific intent: [research guide](../../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md).
Policies: [draft protocol](../../research/RESEARCH_PROTOCOL.md).
Entry evidence: [M0 acceptance](../m0/M0_6_RUNNER_GATE.md#progress-evidence).
Delivery: [tracker and package checklists](README.md).

MUST is an acceptance requirement; SHOULD allows a justified recorded exception;
MAY is optional. Proposed names/commands below are development targets, not
implemented capabilities. This specification governs M1 development and tests.

## 1. Objective and completion boundary

M1 MUST provide attributable, role-separated research interfaces for component
folds, independent randomness, exact supported resume, development predictions,
group aggregation, metrics and selection. Gate G1 MUST demonstrate an isolated
synthetic development workflow without access to outer-test inference.

Completing M1 means infrastructure acceptance, not a fitted scientific baseline,
approved operating threshold, novel pruning method or confirmatory evaluation.
No hidden real-data training is permitted as an acceptance shortcut.

## 2. Entry conditions, inputs and scope

| Input | Contract |
| --- | --- |
| Repository | Canonical `BreakHisClassification`; preserve existing CLI/model/checkpoint compatibility |
| M0 archive | Read-only verified index, receipt, gate, handoff, inventory, components, exposure and bound protocol/decision versions |
| Current observed data | 7,909 supported images, 7,784 retained, 125 exact duplicate exclusions, 81 filename groups; historical observations, not constants to force |
| Source | `cancer_binary`, retained BreaKHis v1 binary derivative; all available magnifications |
| Identity unit | Full filename/slide group; transitive duplicate-linked component is the indivisible partition unit |
| Hardware | CPU-first synthetic acceptance; GPU optional correctness checks; no timing-performance claim from unit tests |
| Policies | Actual numerical seeds/floors/budgets and scientific approvals remain unresolved in the M0 draft |

The existing M0 run has engineering/GPU ready and exploratory/confirmatory work
blocked. M1 engineering may proceed. It MUST NOT reinterpret that handoff as
permission for real fitting, threshold calibration or untouched test assignment.
V2 restoration/download, multiclass training, compact constructors, new pruning,
real recipe search, vulnerability scoring, bootstrap inference, manuscript claims
and M2-M7 experiment execution are outside this milestone.

### 2.1 Operation permissions

| Mode | Permitted work | Required evidence |
| --- | --- | --- |
| `engineering` | Synthetic folds, tiny CPU fits/resume/export/calibration, synthetic held-out denial fixtures | Verified M0 engineering entry; explicit synthetic identity |
| `diagnostic` | Read frozen real metadata; report support/eligibility and blockers | Verified M0 archive and engineering handoff; no real image iteration or active outer assignment |
| `real_design` | Create an attributable real fold plan, without inference/fitting | Updated G0 confirmatory-design eligibility, reviewed candidate boundary, agreed necessary policies and adequate support |

M1 itself MUST NOT launch real fits even if a later G0 permits them; M2 consumes
the interfaces and permissions. Development routines MUST reject outer-test role
construction/iteration/export. Real outer inference stays M6. A synthetic
pre-generated OOF fixture may test schema/completeness without invoking real
held-out inference. Labels needed internally for stratification do not authorize
using outer labels for selection or metric feedback.

Verify M0 through its verifier and receipt, not a copied Boolean. Preserve its
restrictions, namespace, source hashes and reviewed coverage boundaries. A missing,
failed, altered or incompatible handoff blocks the affected operation. Diagnostic
outputs MUST be unusable as scientific assignments. Permission checks prevent
accidental pipeline misuse; they are not an OS sandbox against arbitrary code.

## 3. Proposed implementation structure and reuse

| Target | Responsibility |
| --- | --- |
| `breakhis/research/m1_records.py` | M1 config, records, lineage and role/permission validation |
| `breakhis/research/splits.py` | Deterministic component assignments and support diagnostics |
| `breakhis/research/seeds.py` | Explicit namespace seeds, generators and resume state |
| `breakhis/research/predictions.py` | Aligned development exports and role-sealed dataset/reporting adapters |
| `breakhis/research/aggregation.py` | Image/group pooling and metric definitions |
| `breakhis/research/selection.py` | Epoch, candidate and calibration policies with distinct roles |
| `breakhis/research/m1.py`, `tools/run_m1.py` | Orchestration, verification, G1 and M2 handoff |
| `configs/research/m1.example.json` | Versioned engineering example; real scientific settings remain unapproved |
| `tests/test_m1_*.py` | Package and isolated integration tests |

Reuse canonical parser/hash, data transforms, training, model reconstruction,
checkpoint/RNG, reporting and M0 verification semantics where they match. Existing
`get_split_loaders` currently sets global seeds; it is not yet independent-loader
randomness. Existing grouped reporting is not proof of research pooling/role
controls. Generalize shared helpers explicitly where M0 paths/types are hardcoded;
do not pass M1 output to M0-only capture/config APIs or copy entire implementations.
M0 archives MUST still verify after shared refactors. Keep M0 and M1 schema/spec
identities distinct, with additive explicit type registries and unknown-type errors.

## 4. Filesystem, lifecycle and evidence

Each new run writes to ignored `artifacts/research/m1/<run_id>/`:

```text
run.json
config.resolved.json
inputs/                         # Frozen M0 references, policies and config bytes
baseline/                       # Current M1 source/environment/check identity
design/
  support.json
  assignments.jsonl
  split_summary.json
randomness/
  seeds.json
training/
  history.jsonl
  checkpoint_refs.json
predictions/development/
  images.jsonl
  groups.jsonl
selection/
  thresholds.json
  epoch_selection.json
  candidate_selection.json
reports/
  metrics.json
  benchmark.json
logs/
acceptance/
  tests.json
  evidence_index.json
  gate_g1.json
  m2_handoff.json
  M1_REPORT.md
  receipt.json
```

Artifact presence depends on mode. Diagnostic mode emits support/restrictions,
not invented successful predictions or selections. The verifier MUST require the
declared mode's complete graph. Optional/not-applicable artifacts need reasons.

Run states: `created|running|completed|blocked|failed`. Completed generation and
G1 readiness are separate. Refuse nonempty destinations, traversal, output/input
overlap and escaping links; preserve partial evidence and never silently resume.
Use atomic finite UTF-8 writes, UTC timestamps, operation IDs, input hashes,
actual argv/cwd/timeouts/durations/return codes and logs. Ctrl+C stops subsequent
work. No dataset/old manifest/M0 archive/checkpoint overwrite.

All weights remain under parent `../checkpoint/binary/`, in separate experiment
namespaces, with hash-backed references in the run. Synthetic checkpoint tests
use an isolated temporary equivalent parent tree. No weights in research artifact
folders or codebase checkpoint directories. No implicit pretrained downloads.

Index stable evidence first, then gate/report/handoff, then receipt; no circular
hashing. Verification is read-only and MUST NOT rerun folds, fits, predictions or
calibration, rewrite statuses or replace invalid artifacts. Current M1 source
identity must be captured; the older M0 source snapshot alone cannot identify new
M1 code. Relocation changes neither stored decisions nor semantic identities.

## 5. Record schemas and scientific semantics

Generated JSON uses v1 common envelopes with `schema_version`, `record_type`,
`run_id`, `created_at_utc`, and exactly `data` or `items`. JSONL declares its row
type/version in the index. M1 types use an explicit M1 namespace. Null means
unavailable/unresolved with a reason. IDs derive from canonical content/identity,
not row order or Python's randomized `hash()`. File and semantic hashes differ.

M1.1 implementation binding: its explicit namespaced registry and flat
`m1.input_row` JSONL foundation are documented in the [contracts API](M1_1_CONTRACTS_API.md).
Later assignment/history/prediction row validators and full G1 remain their
own packages; envelope validation is not scientific approval.

### 5.1 Run/config and lineage

Config declares M1 spec identity/version, mode, repo/data/output roots, M0 archive
and expected index/receipt digests, policy/decision hashes, source instance,
selected component boundary, folds/inner-role policy, seed namespace values,
loader/device/determinism settings, prediction and selection policies, operation
timeouts and predecessor if any. Unknown keys/types/ranges fail before outputs.
Paths resolve against the config directory. Commands are allowlisted, never
arbitrary shell strings. Engineering templates carry explicit synthetic values;
no fixture tolerance becomes an agreed scientific setting.

Diagnostic/archive verification must not require original images or environments
to remain present. Label its counts as frozen observations; current live image
availability/integrity is established only when an authorized consumer rechecks it.

Each derived artifact binds current source/environment, M0 evidence index,
inventory/component/exposure identity, assignment digest and protocol/decision
revision. A mismatched source, policy, checkpoint parent or role is an error,
not a warning followed by use of an unrelated artifact.

### 5.2 Assignment rows and partition invariants

Each assignment row includes:

```text
schema_version, record_type, source_instance_id, image_id, relative_path,
raw_sha256, decoded_pixel_sha256, filename_group_id, component_id,
class_label, subtype, magnification, included_status,
eligibility, design_kind, outer_fold, inner_role, assignment_id,
split_seed, split_policy_version, m0_inventory_sha256, protocol_sha256
```

Outer/inner roles are explicit: `outer_test`, `inner_train`,
`development_selection`, `development_calibration`, optional
`shortlist_confirmation`, plus `excluded`/`development_only` when appropriate.
An image may have one row per declared outer fit, not duplicate rows within a
fit. All images/magnifications in a linked component move together.

Plan in stable sorted component order with a versioned stratification algorithm.
Same inputs/seed/policy yield identical assignments across processes and row
permutations. Binary-class support is required; subtype/magnification support is
reported, not fabricated or guaranteed when counts are inadequate. Check zero
filename-group/component/raw-hash/pixel-hash overlap between disjoint roles.
Conflicting labels or malformed membership block assignments.

Only reviewed untouched candidates may form real outer folds. Exposed components
may appear as declared training/development-only additions, never in the OOF
denominator. Unknown exposure cannot be promoted by a split seed. A proposed
five-fold target must fail clearly if infeasible; no silent fold reduction,
component splitting or reshuffling until the desired answer appears.

Inner calibration and selection are disjoint by default. If small support forces
reuse, require an explicit versioned reviewed policy, enumerate reused groups,
mark the dependence and restrict claims; never hide reuse or include outer test.
M0's legacy 70/15/15 split is historical/pilot metadata, not research outer folds.

### 5.3 Independent seeds and resume

Persist distinct namespaces for `split`, `training`, `initialization`, `loader`,
`augmentation`, `pruning`, `search`; reserve bootstrap/timing namespaces for later
milestones when relevant. Values are explicit uint32 or documented deterministic
derivations from a namespace root. Experiment-name suffixes do not set seeds.

Seed model initialization before construction. Loader/sampler generators MUST
not reset unrelated global/model RNG. Augmentation gets an attributable stream;
workers start at zero. Additional-worker/prefetch behavior needs its own tested
policy before use. Changing one namespace must affect only its intended component.
No method/candidate ordering advantage from consuming a shared uncontrolled RNG.

Research checkpoint references bind topology/class mapping, model/optimizer/
scheduler state, epoch/best selection, Python/NumPy/Torch CPU RNG, selected-device
CUDA RNG where applicable, loader/sampler/augmentation states, effective config,
source/environment and parent hashes. CPU paths must not require usable CUDA or
query GPU-only metadata. Restore state after construction/loading and before the
next stochastic operation; do not re-seed over restored streams.

M1 guarantees controlled CPU **epoch-boundary** resume only. Test uninterrupted
versus interrupted/resumed trajectories, sample order, history and weights under
the same deterministic environment. Mid-epoch and incompatible-device/environment
resume must fail or be explicitly labeled unsupported; a fork is a new run, not
an exact-resume claim. Declare any numerical tolerance; do not promise GPU or
cross-version bitwise equality. Preserve existing checkpoint compatibility without
inventing absent research provenance for legacy weights.

### 5.4 Epoch history and role-sealed development export

Epoch rows include fit/fold/method IDs, epoch, train/development losses and units,
class/group support, fixed-reference selection metric, optimizer/LR state,
checkpoint/config/assignment identities, seeds and elapsed time. Loss weighting
must match actual sample counts; percentages and fractions are named explicitly.
Resume appends exactly the next epoch; duplicate/missing epochs or changed policy
are errors. No outer-test metrics appear in development history.

Per-image prediction rows include:

```text
schema_version, record_type, prediction_id, fit_id, method_id,
source_instance_id, image_id, relative_path, raw_sha256, decoded_pixel_sha256,
filename_group_id, component_id, subtype, magnification, true_label,
role, outer_fold, inner_role, seed_namespace_id,
checkpoint_sha256, config_sha256, assignment_sha256, protocol_sha256,
class_names, logits, probabilities, threshold_id, prediction_scope
```

Declare benign index 0 and malignant index 1, validating stored class mappings.
Require finite logits and normalized binary probabilities in [0,1]. Carry actual
row identities through samplers/batches; never zip outputs with an unrelated
sorted manifest after shuffling. Reject duplicate/missing rows, inconsistent
labels, stale input/checkpoint hashes and mixed fits/roles/policies.

Construct only the requested role's dataset/loader. Deny outer-test dataset
construction and iteration in development paths, not just metric printing.
Validation-only reporting selects declared development roles. Benchmark-only
accepts a supplied synthetic/resident tensor and uses no dataset labels or test
loader; distinguish resident forward from end-to-end timing. Preserve legacy
test-only `evaluate`/`compare`, and do not reuse them as development shortcuts.
In M1 engineering runs the resident input must be synthetic, not a cached real
held-out image disguised as a label-free tensor.

Reserve a separate future outer-test output namespace. OOF validators must check
one held-out prediction per eligible image per method/seed, no training-only
groups, complete expected support and correct fold provenance using synthetic
stored rows. Real held-out inference is still prohibited in M1.

### 5.5 Group aggregation and metrics

Proposed primary malignant probability for group g is
`mean_m(mean_i(p_malignant | g,m))` over available magnifications; each image
within a magnification has equal weight, each available magnification has equal
weight, and each group contributes one metric observation. Secondary pooling is
the ordinary mean of that group's images. Preserve both outputs and the declared
primary policy; this recommendation does not establish better accuracy.

Do not duplicate/impute missing magnifications. Store available/missing support,
per-magnification image counts and pooling policy/hash. Mixed group labels or
duplicate prediction IDs fail; repeated raw/pixel identities must follow the
frozen representative policy before pooling. Group probability and image-mean
loss are distinct quantities. Components remain coupled in later uncertainty
analysis even if they contain multiple filename groups.

Report TP/TN/FP/FN, support, sensitivity, specificity, accuracy and balanced
accuracy with explicit group/image unit. Fractions and legacy percent accuracy
must never be mixed silently. Sensitivity without malignant support, specificity
without benign support, and two-class balanced accuracy/AUROC without both classes
return null with reasons. If used, specify AUROC tie rules, binary Brier mean
definition and average precision separately from trapezoidal PR area; do not call
different definitions the same PR-AUC. No inferential bootstrap is required in M1.

### 5.6 Calibration, epoch selection and candidate selection

Keep fixed-reference evaluation, threshold calibration, epoch selection and
candidate selection as separate artifacts/roles, each with permitted input IDs,
policy hash and evaluated support. Calibration data must belong to the declared
development-calibration role (or explicitly reviewed reuse), never outer test.

Proposed calibration: a frozen finite grid in [0,1], maximizing group sensitivity
subject to a frozen specificity floor. Proposed ties: specificity descending,
distance from 0.5 ascending, then numerical threshold ascending. Record the whole
grid, feasible candidates and reasons. No feasible point means `infeasible`, not
an invented threshold, relaxed floor or test-informed choice. Floors, margins,
grid, tie tolerance and support policy require explicit agreement for real use;
engineering fixtures declare their own values. No clinical margin is assumed.

For new threshold rules, propose malignant when `p >= threshold`, including exact
ties. Existing two-class argmax gives benign on equal logits; report argmax and
0.5-threshold references distinctly and preserve legacy behavior. Freeze the
chosen convention before scientific comparison.

Proposed epoch selection: development group balanced accuracy at the fixed
reference rule, earliest epoch on a declared tie. Candidate selection uses the
same frozen constrained operating-point opportunities across competing methods;
it cannot select compression on outer-test feedback. Calibration reuse across
many candidates must be disclosed and bounded; shortlisted confirmation, if used,
is an independent declared role. No M1 method-ranking/search implementation is
implied by supporting its selection interface.

## 6. Runner, permissions and Gate G1

Design target only (not runnable until M1.6): `tools/run_m1.py` with `--config`,
`--run-id`, `--verify-only`, and help. Mode belongs in validated config. Exit 2 is
invalid input, 1 is failure/blocked requested scope, 0 is valid eligible evidence;
valid blocked archives remain distinguishable from invalid evidence. Verification
must not need live M0 inputs after freezing a verified attributable reference set.

G1 reports `ready|restricted|blocked` separately for engineering, real design and
real-fit entry, with source/M0 index digests, blockers, evidence references, owners,
resolution conditions and claims. Missing or failed mandatory evidence takes
precedence. Engineering G1 may pass with current scientific G0 blockers carried
forward; it cannot authorize real M2 fitting while G0 or necessary policies block
it. G1 never authorizes M6 held-out inference or confirms novelty/clinical safety.

The M2 handoff binds current source, parent M0, inventory/components, assignments,
eligibility, seed/role/metric/selection versions, checkpoint refs and allowed
scope. Synthetic/diagnostic artifacts cannot be consumed as real experiment plans.

## 7. Work packages and dependencies

| Package | Work packages | Deliverable | Dependency |
| --- | --- | --- | --- |
| M1.1 | WP0 | Records, verified entry, role/operation contracts | M0 engineering acceptance |
| M1.2 | WP1 | Component folds, role views, support/eligibility checks | M1.1 reviewed |
| M1.3 | WP2-WP3 | Namespaced seeds, epoch history, supported resume | M1.1-M1.2 reviewed |
| M1.4 | WP4 | Aligned development export and reporting/benchmark boundaries | M1.1-M1.3 reviewed |
| M1.5 | WP5-WP6 | Pooling, metrics, calibration and distinct selection | M1.1-M1.4 reviewed |
| M1.6 | WP7-WP9 | Runner/verifier, G1, compatibility, CI/docs and handoff | M1.1-M1.5 reviewed |

Default order is M1.1 through M1.6. Earlier synthetic scaffolding may be shared,
but a package cannot be reviewed until its dependencies and acceptance evidence
are reviewed. No real scientific operation follows merely from package status.

## 8. Acceptance-test matrix

Tests must assert restrictions/denials as well as positive output contents. Use
temporary repos, images and checkpoint roots. CPU is mandatory; GPU is explicitly
optional. No network/pretrained download or real fitting is needed. Current 167
baseline tests are historical evidence; rerun the applicable current suite.

| ID | Fixture/action | Required result |
| --- | --- | --- |
| M1-T01 | Altered/missing M0 index, receipt, gate or handoff | Entry rejected; no copied Boolean grants access |
| M1-T02 | Unknown schema/config/role, traversal or existing output | Clear refusal without data/old artifact mutation |
| M1-T03 | Reordered/relocated identity inputs | Stable semantic IDs; content/policy changes alter identity |
| M1-T04 | Same and changed split seeds across processes | Reproducible same-seed plan; change only declared partition randomness |
| M1-T05 | One component across groups/magnifications | Entire component remains in one role per fit |
| M1-T06 | Transitive raw/pixel links or mixed labels | Zero disjoint-role leakage; label conflict blocked |
| M1-T07 | Exposed/unknown/candidate eligibility | No exposed OOF or unknown untouched assignment; no reseeding cure |
| M1-T08 | Too few class components or rare subtype/magnification | Infeasible design rejected; honest support diagnostics, no forced counts |
| M1-T09 | Separate/reused calibration roles | Disjoint default; only explicit reviewed reuse; outer test always denied |
| M1-T10 | Changed image bytes after freeze | Loader rejects before use; no silent unaudited replacement |
| M1-T11 | Controlled CPU initialization/training seeds | Seed before construction; reproducible supported runs |
| M1-T12 | Change loader/augmentation namespace alone | Intended stream changes without resetting model/split state |
| M1-T13 | Epoch-boundary interrupted/resumed tiny fit | Sample order, RNG, history, optimizer/scheduler and weights match control under declared tolerance |
| M1-T14 | Conflicting resume policy/device/mid-epoch state | Explicit unsupported/conflict failure; no re-seeding over restoration |
| M1-T15 | Uneven batches, repeated epochs, test access trap | Correct history units/weighting; duplicates denied; development only |
| M1-T16 | Shuffled sampler and partial final batch | Image/label/probability alignment survives export/reload |
| M1-T17 | Duplicate rows, NaN/out-of-range probabilities, stale provenance | Prediction validation fails before aggregation |
| M1-T18 | Trap outer dataset constructor/iterator/predictor | Development workflow never constructs/iterates/predicts outer test |
| M1-T19 | Validation-only and resident benchmark-only entry | Only requested development data; benchmark never uses labels/test loader |
| M1-T20 | Hand-computed unbalanced magnification example | Primary/secondary pooling differ exactly as specified; one group one metric vote |
| M1-T21 | Missing magnification, duplicate content or mixed group labels | Available-only support; no imputation/duplicate inflation; conflicts fail |
| M1-T22 | Hand confusion/Brier/rank fixtures and absent classes | Correct units/definitions; unavailable values null with reasons |
| M1-T23 | Finite threshold grid, exact ties and boundary values | Frozen feasibility/tie rules; argmax versus >=0.5 distinction preserved |
| M1-T24 | Undeclared calibration IDs/floor/grid/tolerance | Rejected for real use; no implicit policy agreement |
| M1-T25 | No threshold satisfies floor | Explicit infeasible record; no relaxation/test feedback |
| M1-T26 | Epoch versus candidate selection and role traps | Distinct declared rules and permissions; no outer-test selection |
| M1-T27 | Synthetic development pipeline from verified entry | Complete attributable evidence/G1; no real scientific fit |
| M1-T28 | Repeated verify-only, changed/missing artifact and forged gate | Read-only deterministic validation; invalid evidence nonzero |
| M1-T29 | Existing CLI/models/checkpoint/CPU regressions | Compatibility intact; old data/weights/manifests unchanged |
| M1-T30 | Current real M0 metadata diagnostic handoff | Restrictions preserved, no image iteration/fitting/active untouched assignment; M2 scope honest |

M1.6 MUST retain actual pass/fail/not-run evidence for all 30 IDs and package
coverage. A required skipped case prevents unrestricted G1. A failed hypothesis,
unverified patient mapping or blocked scientific permission is not a unit-test
pass condition to manufacture.

## 9. Definition of done and handoff

- [ ] All package contracts implemented/reviewed or explicitly revised with rationale.
- [ ] Current source/environment and verified M0/record identities attributable.
- [ ] Component folds/role boundaries deterministic and eligibility-safe.
- [ ] Independent seeds and supported epoch-boundary resume demonstrated.
- [ ] Development predictions aligned and provenance/outer-access checks enforced.
- [ ] Hand-computed pooling, metrics, threshold and selection fixtures pass.
- [ ] Synthetic full pipeline and existing compatibility checks pass with actual logs.
- [ ] G1, verifier, human report and M2 handoff agree on permitted scope.
- [ ] Real metadata diagnostic preserves current G0 restrictions without scientific execution.
- [ ] Documentation/CLI/CI reflect implemented behavior and all acceptance IDs are traceable.

Creating this specification checks none of these boxes. M2 may perform real fits
only after the applicable real-data permissions and policy/compute gates pass.
