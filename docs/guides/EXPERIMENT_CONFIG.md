# Experiment configuration reference

Use JSON; no additional YAML dependency or approval files are needed.
Start from `configs/experiments/vgg_baseline.json` or `vgg_channel.json`.
Paths in the config are relative to that config's directory, not the terminal.
Unknown/duplicate fields, invalid values and nonfinite numbers are rejected.

| Field | Meaning and allowed values |
| --- | --- |
| `schema_version` | Integer `1` for existing configs, or `2` with the explicit `baseline` object below. |
| `name` | Short label, 1-64 letters/digits/underscores/hyphens, beginning with a letter/digit. |
| `data.manifest` | Path to a prepared binary manifest. Copied exactly into each run. |
| `data.source` | `cancer_binary`; multiclass training is not yet supported. |
| `data.magnification` | `40X`, `100X`, `200X`, `400X`, or `all`. Same split across magnifications. |
| `data.data_root` | Source image folder, or `null` to use the manifest's root; permits relocation without changing assignments. |
| `training.model` | `vgg`, `resnet`, `resnet18`; config 2 also supports `mobilenet_v3_small`. Initialization is explicit in config 2. |
| `training.epochs` | Positive total planned dense-training epochs, not extra epochs on resume. |
| `training.batch_size` | Positive training and image-evaluation loader batch size. |
| `training.num_workers` | `0` for reproducible epoch-boundary resume. |
| `training.device` | `auto`, `cpu` or `cuda`; resolution saved in `inputs.json`. |
| `training.seed` | Integer 0 through 4294967295; seeds Python, NumPy and PyTorch, including model initialization and data transforms. Split seed is separate. |
| `training.deterministic` | Boolean; enables deterministic PyTorch algorithms and cuDNN settings, with cuBLAS workspace configuration when needed. Device/runtime differences may still change numerical results. |
| `training.learning_rate` | Finite positive initial optimizer rate. |
| `training.optimizer` | `adam` or `sgd`; SGD uses momentum 0.9. |
| `training.weight_decay` | Finite nonnegative optimizer weight decay. |
| `training.scheduler` | `none` (constant rate) or `cosine` over the planned epochs. |
| `training.selection_metric` | `group_balanced_accuracy` (recommended) or legacy image `accuracy`. Selection uses validation only. |
| `training.cpu_threads` | Positive CPU thread setting for training subprocesses through OMP/MKL. |
| `pruning` | `null` for dense baseline, otherwise the object below. |
| `pruning.method` | `channel` (VGG only) or `unstructured` (weight masking). |
| `pruning.sparsity` | Fraction strictly between 0 and 1; fixed before running. |
| `pruning.finetune_epochs` | Positive fine-tuning epoch count after pruning; optimizer/scheduler start fresh using the training recipe. |
| `evaluation.device` | Fixed `cpu` or `cuda`, independent of training. |
| `evaluation.cpu_threads` | Positive PyTorch CPU thread count during evaluation. |
| `evaluation.batch_size` | Positive resident-forward benchmark batch size, not image-loader size; use 1 for deployment latency. |
| `evaluation.warmup` | Nonnegative warmup forwards. Examples use 50. |
| `evaluation.repeats` | Positive measured forwards. Examples use 200. |

| Runner option | Use |
| --- | --- |
| `--config FILE` | Required for development; optional when validating a saved final-phase request. |
| `--run-id ID` | Explicit unique run ID; generated from the config label if omitted. Required for resume. |
| `--seed N` | Override and save the fitting seed; does not resplit data. |
| `--device cpu/cuda/auto` | Override and save training device; evaluation keeps its fixed config device. |
| `--resume` | Continue the same interrupted run with identical inputs/code/runtime; never implicitly initializes a new experiment. |
| `--phase development/final` | Development is the default, with validation reports only. Final explicitly evaluates bound selections on test, using the frozen config and evaluation device without fitting. |
| `--prepare-only` | Save schema-2 frozen inputs and expected-output contracts without training/inference. With `--resume`, revalidate an existing preparation read-only. Final preparation validation requires already completed development. |

Preparation saves exact manifest bytes and SHA-256, a frozen effective config
and its SHA-256, split/fitting seeds, actual runtime, commit/dirty availability
and package source hashes. Missing Git is recorded with null commit/dirty values,
not a fabricated clean state. Pending stage contracts list histories, predictions
and reports as well as primary weights. Prepared records are not completed runs
and cannot be compared as scientific results.

Run records save resolved settings, a frozen manifest, commit/dirty state,
package source hashes, device/runtime, actual stage commands, checkpoint/report
hashes and per-stage elapsed seconds. They do not create mandatory source archives
or claim independent provenance verification.

For split seeds, use `prepare-data --seed N` with a new output directory. For
new training seeds, use `run_experiment.py --seed N` and a new run ID. For
additional sparsities, copy a config and change `pruning.sparsity`; pick a
prespecified set or perform selection on development data, never on test metrics.

Comparison options: `--runs` (at least two completed requested phases), `--output`
(new JSON path; CSV sibling is also saved), `--phase development|final` (default
`development`) and repeatable `--vary training.selection_metric` for a deliberate
selection-policy study. No data, split or benchmark invariant can be overridden.
Comparison reads saved outputs only; it does not evaluate models.

## M2.1 study and cost inputs

`configs/baselines/m2_study.json` is a separate baseline-study schema, not an
experiment config. Its complete fields/defaults are documented in
[M2 master section 4](../specs/m2/M2_SPEC.md#4-study-matrix-and-configuration-contract).
`run_baselines.py --stage prepare` implements this plan schema now; experiment
config-2 standard/continuation/scratch fits are implemented in M2.2; study job
study resolution/execution remains M2.4 work. Config-1 parsing has not changed.

The sample declares a 40X pilot, pooled main data, fitting seeds 102/103, ranking
seed 202, ratios 0.25/0.50/0.75, B=20/F=5, two bounded Adam recipes and separate CPU/
CUDA profile descriptions. Declaring CUDA does not require CUDA merely to prepare.
It does not establish that 20 epochs converge or that 31 jobs fit available time.

Cost input (optional `--cost-estimates FILE`) has exactly:

| Field | Type / contract |
| --- | --- |
| `schema_version` | Integer 1 |
| `plan_sha256` | Result of `breakhis.baseline_plan.sha(plan)` for the exact `jobs.json` plan |
| `estimates` | Object keyed by the plan's `cost_key` values; unknown keys refuse |
| Each entry's `epoch_seconds` | Finite positive supplied estimate for that model/operation/shape |
| Each entry's `fixed_seconds` | Finite nonnegative preparation/pruning/reporting allowance |
| Each entry's `basis` | Nonempty explanation of measurement or conservative bound; the planner does not measure/verify hardware timings |

Partial estimates are retained as incomplete, never interpreted as zero costs.
Complete maximum-plan time sums `fit_epochs * epoch_seconds + fixed_seconds`
for every unique fitting job: required matrix, optional recipes and pilots. Shared
parents occur once. Cost buckets reused across seeds/recipes need a conservative
basis covering all those jobs, not an unmarked extrapolation from another model. Unknown resource limits or estimates keep capacity unready;
limits cannot truncate the plan. Capacity is checked against both max fit jobs
and available wall hours.

Developer helpers `estimate_cost(plan, estimates, retries)` and
`require_capacity(plan, estimates, retries)` accept optional replay overhead entries
with exactly `job_id`, `extra_fit_jobs`, `extra_seconds`, `basis`. Restarting from
epoch zero may charge another fit; replaying a partial epoch can charge seconds
without inventing a new parent fit. Actual execution/accounting is M2.4 work.
The helpers report `execution_supported=false` even when capacity is sufficient.
Use the exact current plan hash; changing study inputs requires fresh estimates/
revision. Example cost numbers must not be treated as measured facts.

## M2.2 explicit initialization (experiment config 2)

Keep the six existing top-level fields and add exactly `baseline`. All six baseline
fields must be present, using null where inapplicable. Paths resolve from the
experiment config directory. Existing config-1 semantics remain unchanged.

| Field | Values and constraints |
| --- | --- |
| `baseline.kind` | `standard`, `dense_continue`, `narrow_scratch`, `prune_from_parent` |
| `baseline.parent_selection` | Completed phase-aware parent's `development/selection.json`, or null for standard |
| `baseline.channels` | Eight positive VGG widths for scratch; otherwise null |
| `baseline.topology_checkpoint` | Bound L1-pruned source defining those widths for scratch; otherwise null |
| `baseline.ranking` | null outside parent pruning; `l1` or `random` within it |
| `baseline.ranking_seed` | uint32 for random parent pruning; otherwise null |

- `standard`: pruning and all other baseline fields null. Random VGG/ResNet18 or
  MobileNetV3-Small. Compact example: `configs/experiments/mobilenet_baseline.json`.
- `dense_continue`: parent required, pruning and other overrides null. Runs a
  fresh optimizer/schedule for `training.epochs` (F), starting at the parent's
  selected dense weights, not its optimizer state. No child `train.history.json`
  or invented parent last checkpoint is created; child history is `finetune.history.json`.
- `narrow_scratch`: VGG, parent, exact widths and topology required; pruning and
  ranking fields null. Starts all numerical weights/classifier/BN state fresh.
  Set `training.epochs` to the declared B+F budget. Parent lineage is topology
  discovery provenance; no parent tensor/optimizer/RNG is inherited.
- `prune_from_parent`: reuses the verified selected dense parent, applies uniform
  L1 masks or L1/random VGG channel pruning, then fits a fresh optimizer for
  `pruning.finetune_epochs` (F). Keep `training.epochs` equal to the parent
  allowance B. Random ranking requires channel pruning and a separate uint32 seed.
  Per-layer overrides are excluded from this matched uniform control.

Parent fitting seed, optimizer/LR/decay/schedule, batch size, deterministic and
selection settings, model, data/magnification/manifest/classes must match. Epoch
allowances differ intentionally for continuation/scratch. Parent must designate
an unpruned dense fit in a supported phase-aware run. Legacy schema-1 runs cannot
be promoted. Origin code/runtime is retained; a compatible older parent may
initialize a new fit, while exact child resume/final still requires the child's
unchanged source/runtime rules. A new fit may run on a different device; that is
not a claim of cross-device exact resume or equal compute.

Scratch topology must carry `parent_checkpoint_sha256` and point to that exact
parent. Generate it with the current low-level L1 channel prune command; old
unbound topologies refuse with a regeneration message. Changing its file, parent
weights/selection, declared widths, recipe or frozen initialization is an error.

Preparation freezes `initialization.json` and declared external parent/topology
roots/references in child inputs. `parent.json` records acquisition with zero
child parent-fitting steps; the original histories and selected/executed epochs
remain attributable by immutable reference. Do not move/change these inputs while
using child resume/final. The runner passes advanced `--initialization FILE` to
train/prune/fine-tune automatically; manual use must match its complete fit contract.
