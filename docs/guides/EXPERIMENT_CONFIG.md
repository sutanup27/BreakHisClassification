# Experiment configuration reference

Use JSON; no additional YAML dependency or approval files are needed.
Start from `configs/experiments/vgg_baseline.json` or `vgg_channel.json`.
Paths in the config are relative to that config's directory, not the terminal.
Unknown/duplicate fields, invalid values and nonfinite numbers are rejected.

| Field | Meaning and allowed values |
| --- | --- |
| `schema_version` | Integer `1`. |
| `name` | Short label, 1-64 letters/digits/underscores/hyphens, beginning with a letter/digit. |
| `data.manifest` | Path to a prepared binary manifest. Copied exactly into each run. |
| `data.source` | `cancer_binary`; multiclass training is not yet supported. |
| `data.magnification` | `40X`, `100X`, `200X`, `400X`, or `all`. Same split across magnifications. |
| `data.data_root` | Source image folder, or `null` to use the manifest's root; permits relocation without changing assignments. |
| `training.model` | `vgg`, `resnet`, or `resnet18`; fresh random initialization. |
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
config-2 execution remains M2.2-M2.4 work. Config-1 parsing has not changed.

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
