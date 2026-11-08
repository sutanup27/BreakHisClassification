# Milestone 1: Reproducible experiment foundation

Specification: **M1-FOUNDATION-SPEC-001**, version 1.0. Date: 8 November 2026.
Document status: Reviewed for implementation. Implementation: In progress;
M1.1-M1.4 are reviewed; M1.5 remains pending. See the package tracker
for current evidence; this master describes the full milestone target.

Plan: [current roadmap](../../research/EXPERIMENT_ROADMAP.md).
Science: [research guide](../../research/Breast_Histopathology_Pruning_End_to_End_Research_Guide.md),
sections 8-10, 16-18. Progress: [package tracker](README.md).
Assessment: [alignment review](ALIGNMENT_REVIEW.md).

## 1. Objective and completion boundary

Provide a simple, reproducible workflow:

```text
prepare one split -> configure -> develop/train -> inspect development results
                 -> explicitly evaluate fixed checkpoints -> compare saved results
```

Reuse the working implementation. Starting this specification from the beginning
means reassessing it against a complete contract, not deleting working code,
resetting experiments or rebuilding every component.

Completion means configurations, data roles, execution, resume, predictions and
comparisons work together and have actual correctness evidence. It does not mean
a converged baseline, malignant-group vulnerability, a novel pruning algorithm,
clinical validity or a generalization claim has been established. This is
infrastructure supporting objectives O1/O2/O3, not an algorithmic contribution.

No approval framework, literature gate, named reviewer requirement, source archive
or separate evidence-management platform is required.

## 2. Scope and existing implementation

Inputs: the supplied v1 binary BreaKHis source, a prepared manifest, an experiment
JSON and installed CPU/CUDA dependencies. Use filename-group terminology. Accept
the researcher's dataset-source declaration; do not demand archive reconciliation.

| Component | Existing behavior to reuse | Gap or verification needed |
| --- | --- | --- |
| `breakhis/experiments.py` | Config validation, snapshots, subprocess stages, resume and comparisons | Default run automatically tests; no phase contract; stage recovery validates only selected primary outputs |
| `breakhis/manifest.py` | Group/duplicate linking, saved stratified split and no overwrite | Validate complete-source grouping consistently when selecting one magnification; test malformed manifests |
| `breakhis/data.py` | Hash-checked image loading, transforms and seeded loaders | Builds all roles eagerly; needs requested-role construction |
| `breakhis/cli.py` | Training, pruning, histories, checkpoint choice and resume | Evaluation is fixed to test; exports need explicit role/schema validation and protected destinations |
| `breakhis/checkpoints.py` | Portable weights, optimizer/scheduler/RNG state | Verify Adam and group-selection resume, including tie-break state and history recovery |
| `breakhis/reporting.py` | Image/group metrics, predictions and resident-forward timing | Development reports, subtype/component attribution, undefined metric behavior and complete output validation |
| Root runners and smoke tools | Simple commands, JSON/CSV exports and generated-image workflows | Phase-aware orchestration and end-to-end acceptance |

Out of scope: new backbones, new ranking/search algorithms, malignant weighting,
threshold fitting, nested/outer-fold generation, bootstrap inference, second-dataset
integration, multi-session benchmarking and manuscript conclusions. Existing
pruning is exercised for integration, not redesigned here.

These belong to milestones 2-5 in the roadmap. The single current grouped
train/validation/test split remains supported; extra roles are not invented.

## 3. Interfaces and configuration

### 3.1 Existing commands

At specification creation, runner/comparison commands lacked phase execution.
M1.3 now executes development by default or explicit `--phase development`;
M1.4 now also executes bound final inference and phase-aware saved comparisons. The existing
six-section experiment JSON is documented in the [configuration guide](../../guides/EXPERIMENT_CONFIG.md).

Keep config schema version 1 and its exact top-level fields:
`schema_version`, `name`, `data`, `training`, `pruning`, `evaluation`.
Keep existing field names, supported models, sparsity validation and examples.
No YAML dependency is needed. Unknown fields, duplicate JSON keys, nonfinite
numbers and invalid scalar types must fail with actionable errors.

Paths resolve from the config directory. Save effective values after seed/device
overrides and root resolution. The split seed comes from the manifest; the fitting
seed comes from training config. Do not silently resplit when fitting seed changes.

Existing defaults remain working defaults, not proven optimal settings: example
VGG, Adam at 0.0001, weight decay 0.0001, no scheduler, 20 epochs, batch 32,
workers zero, seed 102; CPU evaluation at batch one, 50 warmups/200 repeats.
Preserve existing transforms: resize shorter edge 256, crop 224, ImageNet
normalization; random crop/horizontal flip only for training. Save this actual
preprocessing contract, not the different settings in earlier discussions.

### 3.2 Implemented command contract: development, final and saved comparisons

```powershell
# Development is the new default; no test inference.
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102
python run_experiment.py --config configs/experiments/vgg_channel.json --run-id channel-102 --phase development

# Continue an interrupted development run, using the original config/overrides.
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102 --resume

# Explicit final evaluation reads the saved config and selected checkpoint bindings.
python run_experiment.py --run-id baseline-102 --phase final

# Comparisons are phase-specific; development is the default.
python compare_experiments.py --runs baseline-102 channel-102 --phase development --output artifacts/comparisons/dev.json
python compare_experiments.py --runs baseline-102 channel-102 --phase final --output artifacts/comparisons/final.json
```

`--phase` accepts `development` or `final`. Development requires `--config`;
final requires an existing `--run-id`, reads its frozen config and does not need
`--config`. If supplied for final, config must match the frozen effective config.
Seed/training-device overrides are development-only; final uses saved selection
bindings and the fixed evaluation device. `--resume` on a final run can retry an
interrupted final phase but must not resume fitting. It is an error to resume an
already completed requested phase or use an occupied ID as a fresh run.

Low-level `python -m breakhis evaluate` should gain `--split validation|test`.
Its existing default test behavior may remain because this is an explicit
evaluation command; the experiment runner must always pass the intended split.
Standalone explicit test/compare commands remain supported and documented as
final-evaluation operations. This is protection against accidental misuse, not
a sandbox preventing someone from writing arbitrary Python.

## 4. Requirements

| ID | Requirement | Owner | Acceptance |
| --- | --- | --- | --- |
| M1-R01 | Strict, documented config and phase-aware CLI with recorded effective overrides | M1.1 | M1-T01 |
| M1-R02 | Confined, no-overwrite run/checkpoint/report outputs and atomic finite writes | M1.1 | M1-T02, M1-T03 |
| M1-R03 | Frozen manifest/config, source/runtime/seed identity and drift detection | M1.1 | M1-T04 |
| M1-R04 | Reproducible full-source group/raw/pixel/component split integrity | M1.2 | M1-T05, M1-T06 |
| M1-R05 | Validate selected-role paths, metadata, class support and image content | M1.2 | M1-T07 |
| M1-R06 | Construct/read only the roles required by the operation | M1.2 | M1-T08 |
| M1-R07 | Fresh initialization and deterministic fitting setup are attributable and repeatable | M1.3 | M1-T09 |
| M1-R08 | Exact completed-epoch resume with validated state and tie-break recovery | M1.3 | M1-T10, M1-T11 |
| M1-R09 | Durable histories, artifact-complete stage recovery and honest failure/interruption states | M1.3 | M1-T13, M1-T14 |
| M1-R10 | Development defaults to train/validation and never automatically tests | M1.3 | M1-T12 |
| M1-R11 | Final evaluation uses bound development-selected weights/config without fitting | M1.4 | M1-T15, M1-T16 |
| M1-R12 | Correct role-labelled image/group metrics with declared aggregation/threshold and units | M1.4 | M1-T17 |
| M1-R13 | Complete ordered predictions and verifiable report/checkpoint provenance | M1.4 | M1-T18 |
| M1-R14 | Comparable phase-specific JSON/CSV tables distinguish invariants from deliberate factors | M1.4 | M1-T19, M1-T20 |
| M1-R15 | Reproducible bounded resident-forward measurement with explicit scope | M1.4 | M1-T21 |
| M1-R16 | Dense and existing-pruning workflows integrate; old results remain readable without false migration | M1.5 | M1-T22, M1-T23, M1-T24 |
| M1-R17 | Main docs/examples, package tracker and global tracker reflect actual supported behavior | M1.5 | M1-T25 |

### 4.1 Input and role rules

Validate the complete manifest's grouping metadata for the selected source across
all magnifications before filtering images. A complete filename identifier retains
its suffix; raw/pixel-duplicate links and filename groups form connected components.
One component has one split; reject conflicting binary labels, inconsistent stored
component membership, duplicate/case-colliding paths and invalid hashes/metadata.
Do not trust a manipulated `group_id` string instead of the declared linkage rule.
Do not change an existing split to repair an error silently.

Development may audit test-role metadata for split integrity, but must not create
a test dataset/loader, read test pixels, infer, export test probabilities or
compute test quality. Training/selection need both classes within their selected
magnification. Validate test-class support when final evaluation is requested;
missing test files must not prevent a valid development-only run.

At pixel access, verify the audited raw SHA-256, handle decoding errors clearly
and retain the role in failures. A changed image never becomes a newly accepted
input implicitly. Dataset preparation can read all images to build fingerprints;
that audit is distinct from evaluating a fitted model on held-out images.

Provide requested-role dataset/loader construction, reusing current factories
with an optional role list. Train/fine-tune construct train/validation only;
validation reporting constructs validation only; final reporting constructs test
only. Existing callers can retain documented defaults for compatibility, but the
new runner and its CLI adapters must pass roles explicitly.

### 4.2 Selection, fitting and resume

Use existing fresh random initialization and recorded fitting seed, separate from
the split seed. Record Python/NumPy/PyTorch seeding and actual deterministic settings;
do not promise bitwise equality across devices/runtime versions. No hidden global
model, earlier run checkpoint or optimizer is inherited into a fresh experiment.

For `group_balanced_accuracy`, choose the highest validation group balanced
accuracy; break ties by lower image validation loss, then earlier epoch. Preserve
legacy `accuracy` selection behavior and label it explicitly. These are checkpoint
criteria, not sensitivity-preservation constraints or the proposed pruning score.

Persist best selection score, its tie-break loss/epoch and epoch-zero/completed
boundaries consistently. Save optimizer, scheduler and Python/NumPy/CPU/CUDA RNG
state. Resume requires workers zero, unchanged effective training settings,
manifest identity, supported runtime/device and implementation identity. Store a
small source digest map and commit/dirty state; do not create source archives.
Missing Git metadata is recorded as unavailable, not an approval failure.

Histories contain one row per committed epoch: train loss, validation loss,
image accuracy percent, selection score percent/criterion, actual learning rate
used, training and validation seconds, and optimizer-step count. Store next-epoch
learning rate separately if useful. Existing code records rate after scheduler
stepping; do not mislabel that as the rate used by the completed epoch.

On interruption, partial-epoch weights are diagnostic only. Recover history and
best/tie-break state to the same completed-epoch boundary as the latest resumable
checkpoint; never mix a newer best checkpoint/history with an older committed
boundary silently. Validate all outputs before marking/skipping a stage: missing
predictions/history do not become success because a checkpoint/metrics file exists.
Use temporary publication and finish markers or equivalent small records; no
distributed transaction or multi-writer execution service is required.

### 4.3 Development and final behavior

Development runs the fixed dense or dense/prune/fine-tune pipeline, chooses best
checkpoints on validation and exports validation predictions/reports for the
fixed dense/pruned/fine-tuned roles. It produces no test-quality result.

When development succeeds, publish `development/selection.json`: run ID,
effective-config digest, manifest digest, preprocessing/aggregation/threshold
policy IDs, class mapping, and each selected checkpoint's role/path/SHA-256 and
selected epoch/criterion. Include hashes for development reports/predictions.
This is a technical binding to completed outputs, not a human approval record.
The config digest is SHA-256 of the frozen `config.json` bytes, just as the
manifest digest identifies its frozen bytes. Unadapted pruned weights use epoch
zero and `fixed_recipe` attribution rather than an invented best-epoch decision.

Final evaluation requires completed development, unchanged bindings and matching
recorded package source digests. Load the
saved weights; never select another epoch, prune, fit, tune a threshold or use
test metrics to choose a role. Evaluate all bound roles requested by the fixed
pipeline and label their origin. Report a method's designated result role without
choosing whichever test role happened to score best.

CPU final evaluation must work from portable saved weights even when the original
training device was CUDA and is currently unavailable. Only the selected
evaluation device is required; exact fitting resume still uses its stricter runtime
rules. If final evaluation was interrupted, retry only its incomplete outputs.
Repeated final requests cannot overwrite results or silently trigger training.

### 4.4 Metrics, prediction rows and timing

Classes are `['benign','malignant']`; malignant is positive. Image scores use
softmax/argmax. Primary group scores: mean malignant probability within each
available magnification, then equal mean over available magnifications. No
imputation. Threshold is fixed at 0.5, ties to benign. Report ordinary image-mean
group aggregation secondarily. Check group label consistency and report counts.

Report image/group sensitivity, specificity, balanced accuracy, confusion matrix,
precision/recall/F1 and support. Rates are fractions; image accuracy/selection
score fields explicitly marked percent stay percent. Reject nonfinite outputs.
If a reusable metric helper lacks class support, return null with a reason for
undefined rates; normal requested-role checks prevent such a final run. Do not
quietly present unsupported sensitivity as zero evidence of model failure.

Prediction envelopes identify run, phase, split, model role, class mapping,
checkpoint/config/manifest digests and actual evaluation runtime. Rows contain
relative path, raw hash, filename group (`patient_id` legacy field), duplicate-linked
component (`group_id` legacy field), magnification, subtype, target, predicted class
and class-ordered probabilities. Join verified manifest metadata rather than
inventing missing clinical IDs. Require sequential complete coverage: no shuffle,
drop-last, duplicate/missing rows or mixed phase. Per-group records retain group
and component attribution and available magnifications. Future fold IDs can be
added when folds exist; do not fabricate them now.

Retain model-only size, physical/nonzero parameters and scoped Conv/Linear MACs.
Benchmark eval/inference mode, fixed batch/dtype/device/threads, resident inputs,
warmup/repeats, CPU wall-clock and synchronized GPU timing. Use a fixed shape/dtype
input generated from a local RNG with seed zero rather than held-out observations
for search-independent measurement; do not alter global fitting RNG state.
Record protocol `resident-forward-synthetic-v1`, shape, dtype and seed.
Save samples, median/p95 and runtime. One
session is an implementation measurement, not independent session replication or
end-to-end throughput. Extra sessions and statistical inference belong to milestone 5.

### 4.5 Comparison and fair interpretation

`compare_experiments.py` reads already saved outputs and never evaluates models.
Select one phase explicitly (default development); never fall back to the other
phase. Require valid artifacts and matching manifest/group assignments, source,
magnification, class mapping, preprocessing, aggregation/threshold and benchmark
device/runtime/protocol for latency comparisons. Do not permit varying these
identity/measurement invariants to manufacture a matched result.

Model, seed, dense epochs and fixed pruning settings may differ, as currently
supported; enumerate every such difference in the output. Selection criterion is
normally fixed, but a deliberate experiment may declare it through the implemented
`--vary training.selection_metric` option. Reject an undeclared difference; do not
force every future comparator to use identical selection methods or invent
class-weighting/search support here. Those extensions belong to milestone 4.

Rows include phase/split, model/method/seed, role/checkpoint identity, group support,
rates, image accuracy, parameters, MACs, latency, planned/completed dense and
fine-tuning epochs/steps, and measured phase durations where available. Preserve
infeasible/failed-run information as separate diagnostics, not valid scored rows.
Different epochs are disclosed, not automatically declared fair. Dense continuation,
scratch/compact controls and equal search budgets are milestone 2/4 experiments.
Seed replicates are separate rows, not additional independent groups or a CI.

## 5. Storage, identity and compatibility

One run lives under `artifacts/experiments/<run-id>/`; weights remain under
`../checkpoint/binary/<magnification>/<model>/<run-id>/`. No weights inside the
codebase or dataset. IDs follow the existing 1-64 character policy. Reject
traversal, escaping links, dangerous input/output aliases and replacing source
images, manifests, config files or input checkpoints with reports. Writers remain
single-process/single-run; concurrent writers to one ID are not supported.

```text
config.json, manifest.json, inputs.json, run.json
logs/<stage>.log
train.history.json, finetune.history.json       # as applicable
development/
  selection.json
  <role>.metrics.json, <role>.predictions.json
  results.json
final/
  <role>.metrics.json, <role>.predictions.json
  results.json
results.json                                 # phase index, not a mixed scoreboard
```

Manifest bytes are copied exactly; save their SHA-256 and recorded split seed.
Source/runtime/config identities and selected checkpoint digests must match the
phase's requirements. No silent rescan/resplit or scientific-history attestation.

Use run/report schema version 2 for the new phase-aware format; experiment config
and prepared manifest stay version 1. `run.json` records `active_phase`, phase
statuses, stage argv/log/time/return code and complete artifact references/hashes.
Top status distinguishes `development_complete` from `final_complete`, alongside
`prepared`, `running`, `failed`, `interrupted`. Each phase has its own `not_requested`,
`running`, `completed`, `failed`, `interrupted` status. Preparation alone leaves
both phases not_requested; execution starts development running in M1.3.
A failed final phase does not invalidate completed
development outputs for development comparisons.

Old schema-1 runs/reports remain readable as JSON and their supported checkpoints
remain usable by individual commands. They must not be automatically declared
development-only, untouched or eligible for a new final phase. New phase-aware
comparison/resume/final operations reject unsupported schema-1 runs with an
actionable explanation; no automatic migration or extra legacy service is required.
Do not mutate old runs/configs/checkpoints. Document the safer default introduced
in the high-level runner.

## 6. Ordered packages

| Package | Requirements | Dependencies | Handoff |
| --- | --- | --- | --- |
| [M1.1 Config, storage and identity](M1_1_CONFIG_IDENTITY.md) | M1-R01-M1-R03 | None | Validated inputs and stable run contracts |
| [M1.2 Split and role isolation](M1_2_SPLITS_ROLES.md) | M1-R04-M1-R06 | M1.1 | Audited metadata and role-restricted loaders |
| [M1.3 Execution, resume and histories](M1_3_EXECUTION_RESUME.md) | M1-R07-M1-R10 | M1.1, M1.2 | Completed development with recoverable fitting state |
| [M1.4 Evaluation, outputs and comparison](M1_4_EVALUATION_COMPARISON.md) | M1-R11-M1-R15 | M1.1-M1.3 | Bound final inference and phase-specific reports |
| [M1.5 Integration and documentation](M1_5_INTEGRATION.md) | M1-R16, M1-R17 | M1.1-M1.4 | Verified end-to-end workflow and accurate guides |

## 7. Acceptance matrix

These are planned cases, not tests already implemented/passed under these IDs.

| ID | Requirements | Fixture/action | Expected observable outcome | Owner |
| --- | --- | --- | --- | --- |
| M1-T01 | M1-R01 | Valid config/overrides; unknown fields, duplicate keys, bool-as-int, nonfinite values and unsupported phase | Correct resolved settings; invalid inputs fail before run creation | M1.1 |
| M1-T02 | M1-R02 | Occupied ID, traversal, escaping link and report/input aliases | No overwritten run/data/config/checkpoint or escaped output | M1.1 |
| M1-T03 | M1-R02 | Inject publication failure | No truncated successful artifact; failure retained, stage incomplete | M1.1 |
| M1-T04 | M1-R03 | Change manifest/config/source; Git unavailable | Drift refused where required; unavailable Git recorded honestly | M1.1 |
| M1-T05 | M1-R04 | Prepare two small inventories with same seed; alter fitting seed | Same split, no overwrite; fitting seed does not resplit | M1.2 |
| M1-T06 | M1-R04 | Cross-magnification filename, raw/pixel and transitive-component leakage | Rejected even if the chosen magnification hides the collision | M1.2 |
| M1-T07 | M1-R05 | Conflicting labels, malformed hashes, missing class/path or changed bytes | Relevant-role failure before use; no silent repair or missing-class metric | M1.2 |
| M1-T08 | M1-R06 | Spy on role factories and test-pixel reads during development | Test dataset construction/iteration/pixel access never occurs | M1.2 |
| M1-T09 | M1-R07 | Fresh identical tiny CPU fits and a changed seed | Same settings/seed reproduce state; changed seed is attributable | M1.3 |
| M1-T10 | M1-R08 | Interrupt/resume SGD/cosine and Adam/constant, including group-score ties | Weights, optimizer, scheduler, RNG and selected epoch match uninterrupted fits | M1.3 |
| M1-T11 | M1-R08 | Drift settings/runtime, missing best or partial-epoch checkpoint | Actionable refusal; no partial-epoch exact-resume claim | M1.3 |
| M1-T12 | M1-R10 | Run default development pipeline with a test-adapter tripwire | Validation outputs only; no test metrics/probabilities or final execution | M1.3 |
| M1-T13 | M1-R09 | Ctrl+C/child failure and interrupted publication boundaries | Non-success state with logs; resume only completed boundaries | M1.3 |
| M1-T14 | M1-R09 | Stale/duplicate history, changed predictions or missing secondary output | Deterministic history recovery; stage not silently skipped as complete | M1.3 |
| M1-T15 | M1-R11 | Final without development; changed checkpoint/config/selection binding | Refused before test iteration; no replacement selection | M1.4 |
| M1-T16 | M1-R11 | Valid CPU final from CUDA-origin weights; tripwires on fit/prune/optimizer | Test-only inference succeeds without CUDA training or fitting | M1.4 |
| M1-T17 | M1-R12 | Hand-computable image/group/magnification examples, ties and missing support | Correct scores, denominators, units, null/reason where undefined | M1.4 |
| M1-T18 | M1-R13 | Ordered uneven batches; shuffle/drop/duplicate/missing rows; unsafe export path | Complete attributed predictions or clear refusal; existing inputs protected | M1.4 |
| M1-T19 | M1-R14 | Compare phases, missing/corrupt reports, differing split or timing protocol | No cross-phase fallback, inference or invalid matched-latency table | M1.4 |
| M1-T20 | M1-R14 | Different seeds/models/epochs/pruning; declared versus undeclared selection difference | Differences and budgets listed; invariants cannot be varied away | M1.4 |
| M1-T21 | M1-R15 | CPU timing and conditional GPU synchronization; zero gain/slowdown | Scoped valid measurements; negative results retained, no unmeasured speedup claim | M1.4 |
| M1-T22 | M1-R16 | Fresh-process generated-image dense development -> explicit final -> comparison | All outputs validate; no test access before explicit final | M1.5 |
| M1-T23 | M1-R16 | Existing mask/channel pipelines through development/final and reload | Topology/masks preserved; all roles, histories and phases attributable | M1.5 |
| M1-T24 | M1-R16 | Read old run/report/checkpoint; request phase promotion or incompatible resume | Compatible reading or actionable refusal; no historical mutation | M1.5 |
| M1-T25 | M1-R17 | CLI help, example dry checks, documentation links/build and trackers | Examples match implemented options; completion reflects actual evidence | M1.5 |

## 8. Milestone acceptance and next scientific work

M1.1 implementation binding: `--prepare-only` creates schema-2 records with status
`prepared` and no fitting/inference. `--phase` is accepted for preparation/validation
but phase execution remains M1.3/M1.4. The existing no-phase runner still uses its
legacy schema-1 workflow, including automatic test evaluation, until M1.3 replaces
that default. It cannot resume a schema-2 prepared run. This staged compatibility
does not mark M1-R06/M1-R10/M1-R11 complete.

Complete the 25 cases, relevant existing regressions and generated-image
fresh-process dense/mask/channel workflows. CPU is the required correctness
platform; CUDA-only checks are conditional and explicitly reported. A GPU skip
does not invalidate CPU correctness, but cannot establish GPU behavior. Do not
require another costly real-data fit just to approve this infrastructure;
existing real-data execution is context, not proof of the new phase contract.

Each package must be reviewed against its outputs and tests; milestone completion
requires integration and accurate docs, not just five finished specification files.
Keep ordinary test logs/run outputs and a concise completion note. No synthetic
accuracy, timing noise or test count establishes novelty or competitive quality.

Milestone 2 trains competitive controls and budgets; milestone 3 measures
vulnerability; milestone 4 implements the mechanism, constrained selection and
decisive comparisons; milestone 5 adds final-study folds, uncertainty, repeated
timing and feasible replication. No material scientific decision blocks starting
M1.1. Numerical scientific margins, method weights and study-wide fold counts are
deliberately not invented or made prerequisites for this foundation.
