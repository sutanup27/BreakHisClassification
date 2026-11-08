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
| `--phase development/final` | Development is the execution default, with validation reports only. Final execution remains M1.4; only final preparation validation is currently available. |
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
