# BreaKHis classification and pruning

This project trains PyTorch models to classify breast histology images as **benign** or **malignant**, then evaluates, prunes, and fine-tunes the saved models. Model commands support binary classification. The eight-subtype dataset is audited, but multiclass training is not implemented yet.

## Start here

Read this guide from setup through training before running an experiment. An
[offline HTML guide](docs/index.html) presents this README and the detailed
documentation with navigation, search, and copy buttons. Open `docs/index.html`
in a browser; no web server, installation, or internet connection is required.

| Goal | Read next |
| --- | --- |
| Run the project for the first time | [First-run checklist](#first-run-checklist), then [environment setup](#environment-setup) |
| Train, prune, and evaluate | [Complete binary workflow](#complete-binary-workflow) |
| Understand an argument | [Shared parameters](#shared-parameters) and the command-specific tables below |
| Stop and continue a run | [Resume training and fine-tuning](#resume-training-and-fine-tuning) |
| Find or interpret saved files | [Checkpoint paths](#checkpoint-paths-and-loading-rules), [format details](docs/CHECKPOINTS.md) |
| Modify the implementation | [Working on the codebase](#working-on-the-codebase) |
| Diagnose an error | [Troubleshooting](#troubleshooting-and-remaining-work) |

## First-run checklist

1. Open PowerShell in `D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification`.
2. Activate `.venv313` or use its Python executable directly. If starting from a
   fresh checkout, follow [environment creation](docs/ENVIRONMENT.md) first.
3. Check the interpreter, dependencies, binary images, and saved manifest:

```powershell
python --version
python tools/check_environment.py --device cuda
Test-Path ../Dataset/dataset_cancer_v1/classificacao_binaria
Test-Path artifacts/data/manifest.json
```

Both `Test-Path` results must be `True`. Use `--device cpu` in the environment
checker for a CPU installation. The checker runs a forward pass, not training.
The dataset and manifest are local artifacts and are not included in Git. A fresh
clone therefore needs those files copied into the layout below. If images moved,
use `--data-root` while keeping their contents and manifest assignments unchanged.

**Current setup limitation:** after removing v2, the existing v1 manifest works,
but the audit command cannot yet rebuild using v1 alone. Restore the existing
manifest from your local backup; do not assume `prepare-data` works in a clean
v1-only workspace. See [prepare-data limitation](#prepare-data-parameters-and-current-limitation).

4. Run the tests before a long experiment:

```powershell
python -m unittest discover -s tests -v
```

5. Start with a separate one-epoch run to confirm the complete selected dataset
   loads and produces finite training/validation results:

```powershell
python -m breakhis train --dataset breakhis --model vgg --magnification 40X --epochs 1 --batch-size 16 --num-workers 0 --experiment-id first-check
```

This still processes all selected training/validation images. It is not a small
sample run. Expect `train.config.json`, `train.pth`, and `train-last.pth` in
`../checkpoint/binary/40X/vgg/first-check/`. Use a new ID if that directory already
contains output files. A completed one-epoch run cannot resume for extra epochs;
start a new baseline or fine-tune its weights.

## Workspace and checkpoint layout

```text
BreakHis/
  Dataset/
    dataset_cancer_v1/
      classificacao_binaria/       # Supported training source
      classificacao_multiclasse/   # Audit only for now
  BreakHisClassification/          # Run commands from this codebase directory
    breakhis/                     # Canonical package
    artifacts/data/manifest.json  # Saved audited image records and splits
  checkpoint/
    binary/
      40X/vgg/baseline/train.pth
      40X/vgg/baseline/prune.pth
      40X/vgg/baseline/finetune.pth
      legacy/                     # Previously saved checkpoints
    multiclass/                   # Reserved for future multiclass training
```

The local `BreaKHis_v2` dataset has been removed. The default model-command source is now `cancer_binary`. Moving old files into `binary/legacy/` does not migrate their format or establish their experiment provenance.

## Environment setup

The tested baseline is Windows x64 with Python 3.13. See [environment setup](docs/ENVIRONMENT.md) for environment creation, pinned dependencies, CPU installation, and the verified Python executable.

Activate the existing environment in PowerShell:

```powershell
cd D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification
.\.venv313\Scripts\Activate.ps1
python --version
```

If activation is blocked, allow scripts for the current session and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv313\Scripts\Activate.ps1
```

Activation is optional. Replace `python` in any command with `.\.venv313\Scripts\python.exe` to use the environment directly.

If dependencies are missing, install the NVIDIA CUDA 12.8 wheel selection first, then the common requirements:

```powershell
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements.txt
python tools/check_environment.py --device cuda
```

For CPU execution, use `requirements-torch-cpu.txt` instead of the CUDA wheel file in a separate CPU environment. Installing CPU wheels into the GPU environment replaces its GPU build.

## Complete binary workflow

These commands use the existing saved manifest and the remaining v1 binary dataset. Keep the manifest, source, and magnification consistent throughout the experiment.

```powershell
# Train and select the best weights using validation accuracy.
python -m breakhis train --dataset breakhis --source cancer_binary --magnification 40X --model vgg --epochs 20 --device auto --experiment-id baseline

# Prune weights and check validation performance.
python -m breakhis prune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/train.pth --sparsity 0.5 --experiment-id baseline

# Fine-tune with the pruning masks still active.
python -m breakhis finetune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/prune.pth --epochs 5 --learning-rate 0.0001 --experiment-id baseline

# Evaluate the selected model on the reserved test split.
python -m breakhis evaluate --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/finetune.pth --output artifacts/final-test.json
```

Training and fine-tuning use the training split and choose weights using validation accuracy. Pruning reports validation performance. Evaluation uses the reserved test split; run it after choosing your model and pruning settings. Reported accuracy is a percentage on a 0-100 scale: `90.0` means 90%.

## Shared parameters

These arguments apply to **train, evaluate, prune, and finetune**. Omitted optional arguments use their defaults. Flags such as `--deterministic` take no value.

| Parameter | Default / allowed values | Purpose and usage |
| --- | --- | --- |
| `--dataset` | **Required**; `breakhis` only | Always supply `--dataset breakhis`. CIFAR-10 is a separate legacy workflow. |
| `--source` | `cancer_binary`; also `breakhis_v2` | Selects source records in the manifest. Use `cancer_binary` for v1. The v2 choice requires restoring or relocating its images. `cancer_multiclass` is not accepted by model commands yet. |
| `--manifest` | `<codebase>/artifacts/data/manifest.json` | Saved image records, hashes, roots, and patient split assignments. Example: `--manifest artifacts/data/manifest.json`. |
| `--magnification` | `40X`; `40X`, `100X`, `200X`, `400X`, `all` | Filters images by magnification. `all` combines all four into one run; it does not train four separate models. |
| `--data-root` | Selected source root in the manifest | Overrides the image root after relocation. Example: `--data-root ../Dataset/dataset_cancer_v1/classificacao_binaria`. Use the folder containing magnification folders. Files must still match saved hashes. |
| `--batch-size` | `32`; positive integer | Images processed per batch, for both training and evaluation. Larger batches need more memory; try `16` or `8` if GPU memory is exhausted. |
| `--num-workers` | `0`; nonnegative integer | Background data-loader processes. Zero simplifies interruption on Windows. Positive values enable persistent workers and prefetching. Try `4` after measuring throughput. Different from audit `--workers`. |
| `--device` | `auto`; `auto`, `cpu`, `cuda` | `auto` uses CUDA when available, otherwise CPU. `cuda` fails if unavailable. `cpu` forces CPU execution. |
| `--deterministic` | Off; `--deterministic` / `--no-deterministic` | Enables deterministic cuDNN settings and disables cuDNN benchmarking; may be slower. Does not guarantee identical results across hardware/software versions. Loader RNGs use seed 0; model commands have no `--seed` argument. |
| `--experiment-id` | Generated UTC timestamp and random suffix | Groups outputs under a distinct experiment directory. Use 1-64 letters, digits, underscores or hyphens, starting with a letter or digit. An explicit existing checkpoint output is rejected to avoid overwriting an experiment. |
| `--output` | Command-dependent | Checkpoint output for train/prune/finetune; optional report file for evaluate. See checkpoint rules below. |

## Train parameters

`python -m breakhis train` creates a new model unless `--resume` is supplied.

| Parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--model` | `vgg`; `vgg`, `resnet`, `resnet18` | Architecture. `resnet` is an alias for torchvision ResNet18. The chosen name is used in the default checkpoint directory. |
| `--epochs` | `20`; positive integer | Passes through the training split. Validation runs after every epoch. |
| `--learning-rate` | `0.001` | Initial SGD learning rate. Use a positive value; for example `0.0001` for smaller updates. |
| `--pretrained` | Off; flag | Loads torchvision pretrained ResNet18 weights, potentially downloading them. The binary classifier head is replaced. Has no effect for VGG, which does not load pretrained weights. |

SGD uses momentum `0.9`, weight decay `0.0005`, and cosine learning-rate scheduling over the requested epochs. These settings are currently fixed in code. Weights are saved when validation accuracy improves. A new run refuses an existing checkpoint output. Choose a new experiment ID or output filename.

```powershell
python -m breakhis train --dataset breakhis --model resnet18 --pretrained --epochs 10 --batch-size 16 --output resnet18-baseline.pth
```

This custom filename saves to `../checkpoint/binary/resnet18-baseline.pth`.

## Prune parameters

`python -m breakhis prune` loads weights, applies masks, checks validation, and saves weights. It performs no training.

| Parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--checkpoint` | **Required**; file path | Supported checkpoint to prune. Architecture is reconstructed automatically. |
| `--sparsity` | `0.5`; number from `0` to `1` inclusive | Fraction selected for L1 unstructured pruning in each convolution and linear layer. `0.5` requests 50%; `0` requests none; `1` selects all. This does not remove channels or guarantee faster inference. |

Masks are stored in the checkpoint and reconstructed on loading. Dense tensor shapes remain unchanged.

## Fine-tune parameters

`python -m breakhis finetune` starts a new optimization run from saved weights. Pruned checkpoints retain their masks.

| Parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--checkpoint` | Required for a new run; replaced by `--resume` when continuing | Starting weights; architecture is read from the checkpoint. Do not supply `--model`. |
| `--epochs` | `20`; positive integer | Additional training epochs. Set explicitly, for example `--epochs 5`. |
| `--learning-rate` | `0.001` | Initial rate for the new SGD optimizer. A smaller value can be useful for fine-tuning. |

Uses the same optimizer, scheduler, and validation selection as training. Without `--resume`, optimizer, scheduler, and RNG state start a new run. With `--resume`, their saved states are restored.

## Evaluate parameters

`python -m breakhis evaluate` evaluates saved weights on the test split.

| Parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--checkpoint` | **Required**; file path | Model weights; architecture and pruning masks are reconstructed. |
| `--output` | Experiment-directory `evaluate.json` | JSON report path. Example: `--output artifacts/final-test.json`. Parent folders are created. Evaluation saves no model weights. |

The JSON report is printed even without `--output`. It includes test accuracy, cross-entropy loss, and dataset metadata.

## Checkpoint paths and loading rules

| Command | Default output relative to the workspace parent |
| --- | --- |
| `train` | `checkpoint/binary/<magnification>/<architecture>/<experiment-id>/train.pth` |
| `prune` | `checkpoint/binary/<magnification>/<architecture>/<experiment-id>/prune.pth` |
| `finetune` | `checkpoint/binary/<magnification>/<architecture>/<experiment-id>/finetune.pth` |

Defaults are resolved using the codebase location, independently of your working directory. For checkpoint commands, a bare output filename such as `--output baseline.pth` goes directly into `checkpoint/binary/`. An explicit relative or absolute path must resolve inside that task directory and end in `.pth`. From the codebase, `--output ../checkpoint/binary/40X/vgg/baseline.pth` is valid. Binary outputs cannot go into the multiclass directory. Checkpoint input paths can be anywhere.

Other explicit relative paths, including checkpoint inputs and evaluation reports, are relative to your current working directory. Quote paths containing spaces.

Each completed training/fine-tuning epoch also saves `<name>-last.pth`, including optimizer, scheduler, and random state. The main checkpoint remains the best validation model. Each command saves its resolved arguments as `<name>.config.json`; evaluation saves a JSON report automatically when no output is supplied.

Ctrl+C attempts to save partial-epoch weights as `<name>-interrupted.pth`. Resume from the completed-epoch `-last.pth`, not the partial weights. Work from an interrupted epoch is replayed. A second interruption or stalled operation can prevent saving.

### Resume training and fine-tuning

| Parameter | Commands | Purpose |
| --- | --- | --- |
| `--resume` | `train`, `finetune` | Completed-epoch checkpoint containing optimizer, scheduler, and RNG state. Cannot be combined with `--checkpoint`; training cannot combine it with `--pretrained`. |

```powershell
python -m breakhis train --dataset breakhis --resume ../checkpoint/binary/40X/vgg/baseline/train-last.pth
python -m breakhis finetune --dataset breakhis --resume ../checkpoint/binary/40X/vgg/baseline/finetune-last.pth
```

Omitted options inherit saved settings. `--epochs` is the original **total planned epochs**, not additional epochs on resume. Changing the planned total changes the cosine schedule and is rejected. Batch size, learning rate, worker count, deterministic mode, device, output, runtime versions, experiment ID, and dataset provenance must match. A fully completed run cannot be resumed for extra epochs; start `finetune --checkpoint ...` instead.

Exact epoch-boundary resume requires `--num-workers 0`. Background-worker RNG/prefetch states are not stored. The full original experiment directory, including its best checkpoint, must remain available. Reproducibility across different software/hardware or nondeterministic GPU operations is not guaranteed. Version-1 weights remain loadable for evaluation/pruning/fine-tuning, but do not contain exact resume state. See [checkpoint format and resume](docs/CHECKPOINTS.md).

Supported checkpoints contain architecture, binary labels, state dictionaries, epoch, and dataset metadata. Loading requires matching manifest hash, source, magnification, and preprocessing. Regenerating a manifest may invalidate checkpoint metadata. Old whole-model pickle files require explicit migration and are not automatically loaded.

## Prepare-data parameters and current limitation

`prepare-data` decodes and hashes images, checks folder/filename labels, links duplicate content, and saves approximately 70%/15%/15% patient-group splits for train/validation/test. Group fractions do not guarantee identical image fractions. See [dataset details](docs/DATASETS.md) for the grouping policy.

| Parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--dataset-root` | `<workspace>/Dataset` | Parent containing `BreaKHis_v2` and `dataset_cancer_v1`, not a magnification folder. |
| `--output` | `<codebase>/artifacts/data` | Directory receiving `manifest.json` and `audit.json`. This is a directory, unlike checkpoint/report outputs. |
| `--seed` | `0`; integer | Reproducible patient-group split seed. Does not configure training seed. |
| `--workers` | `8`; positive integer | Image-audit threads. Separate from loader `--num-workers`. |

**After deleting v2, rebuilding is currently blocked:** this command still requires all three configured sources, including `BreaKHis_v2`. Continue using the existing manifest with `--source cancer_binary`. A v1-only rebuild needs a further code change. The following example requires restoring v2 or implementing that change first:

```powershell
python -m breakhis prepare-data --dataset-root ../Dataset --output artifacts/data --seed 0 --workers 8
```

## Preprocessing and runtime

Images are converted to RGB and resized to a 256-pixel short edge with aspect ratio preserved. Training uses random 224 x 224 crops (padding 4) and horizontal flips. Validation/test use center crops. Tensors use ImageNet mean/std normalization. These are fixed in code; there is no image-size option. Older 460 x 700 evaluation accuracy is not directly comparable.

Each loaded image is checked against its manifest hash. Epoch logs show training and validation time, losses, and validation accuracy. `--epochs 1` checks execution but still processes all selected images; there is no quick-subset argument. CPU execution and `--magnification all` can take longer. Measure throughput before increasing workers. Unstructured pruning does not automatically speed up dense computation.

Training and validation losses are averaged over samples, so a smaller final
batch does not receive the same weight as a full batch. Invalid learning rates,
epoch counts, and empty splits are rejected. Nonfinite inputs, outputs, losses,
or gradient norms stop training; invalid gradients are rejected before the
optimizer updates weights. This fail-fast validation can add device checks per batch.

## Help, environment checking, and wrappers

```powershell
python -m breakhis --help
python -m breakhis train --help
python -m breakhis evaluate --help
python -m breakhis prune --help
python -m breakhis finetune --help
python -m breakhis prepare-data --help
python tools/check_environment.py --device cuda --output artifacts/environment.json
python -m unittest discover -s tests -v
```

| Environment-checker parameter | Default / allowed values | Purpose |
| --- | --- | --- |
| `--device` | `cpu`; `cpu`, `cuda` | Device for tensor and VGG forward checks. Unlike model commands, `auto` is not supported. |
| `--output` | No file by default | Optional JSON evidence path; the report is also printed. |

The checker verifies imports, tensor computation, image processing, a two-class VGG forward pass, and dependency consistency. It does not train or download pretrained weights.

| Compatibility script | Equivalent command |
| --- | --- |
| `Train.py` | `train` |
| `Main.py` | `train`, defaulting to model `resnet` |
| `Test.py`, `PruneEvaluator.py`, `PruneViewer.py` | `evaluate` |
| `PruneTrain.py` | `prune` |
| `PruneReTrain.py` | `finetune` |

Example: `python Train.py --dataset breakhis --magnification 40X --model vgg`. Wrappers accept their corresponding command's parameters. Optional editable installation (`python -m pip install -e . --no-deps --no-build-isolation`) also exposes `breakhis train ...` without `python -m`.

## Troubleshooting and remaining work

| Symptom | Action |
| --- | --- |
| `python` is not recognized | Activate the environment or use its executable directly. |
| `No module named torch` | Check the interpreter and install pinned requirements into that environment. |
| CUDA unavailable | Check the CUDA wheel selection and NVIDIA driver with the environment checker; use `--device cpu` if needed. |
| Manifest missing | Restore the saved manifest; rebuilding currently needs the removed v2 source as described above. |
| Image changed since audit | Restore the audited image or rebuild after resolving the v1-only audit limitation. |
| Checkpoint metadata mismatch | Match source, magnification, and manifest to the checkpoint's original run. |
| Checkpoint output rejected | Use a bare `.pth` filename or a path inside `../checkpoint/binary/`. |
| Legacy checkpoint rejected | Train a new supported checkpoint or implement explicit migration. |
| Output already exists | Choose a new experiment ID for a new run, or use `--resume` to continue an incomplete run. |
| Resume configuration differs | Keep the saved total epochs, learning rate, batch size, workers, deterministic mode, runtime/device, and original output. Omit options to inherit them. |
| Run already completed | Start a new fine-tuning run from its weights; resume does not extend the saved schedule. |

Stages 1-6 are implemented and verified for the supported binary workflow. Full baseline quality, multiclass training, structured pruning correctness, efficiency reports, and CI remain future work. See [fix roadmap](docs/CODEBASE_FIX_ROADMAP.md) and [command reference](docs/COMMANDS.md).

CIFAR-10 experiments and mixed tutorials live under `experiments/cifar10/` as legacy reference material outside the supported command surface. Virtual environments and generated artifacts are excluded from Git; the parent checkpoint directory is outside this repository.

## Working on the codebase

The supported implementation lives in `breakhis/`. Root scripts and the historical
`Models/`, `DataProcess/`, and `Utills/` paths delegate to it. Edit the canonical
module rather than adding another implementation to a compatibility wrapper.

| Module / directory | Responsibility | Change here when you need to... |
| --- | --- | --- |
| `breakhis/cli.py` | Argument parsing, workflow dispatch, configuration, validation selection | Add a command/argument or change how commands connect |
| `breakhis/data.py` | Manifest datasets, transforms, loaders, loader seeds | Change preprocessing or data loading |
| `breakhis/manifest.py` | Filename validation, image hashes, grouping, saved splits | Audit a source or change the split policy |
| `breakhis/models/` | Architecture implementations | Add or modify a model; also register its constructor in checkpoints |
| `breakhis/training.py` | Training/evaluation steps, loss aggregation, finite checks | Change optimizer-step behavior or shared metrics |
| `breakhis/checkpoints.py` | Model reconstruction, format validation, training/RNG state | Change saved metadata or resume behavior |
| `breakhis/paths.py` | Task checkpoint directories and experiment path validation | Change checkpoint placement safely |
| `tests/` | CPU regressions and subprocess reload checks | Verify behavior affected by a code change |
| `tools/check_environment.py` | Dependency/device verification | Diagnose environment setup |
| `experiments/cifar10/` and legacy pruning helpers | Historical experiments | Consult reference code; these are outside the supported workflow |

To change behavior, locate the canonical module, implement the change, run the
relevant existing tests, then run the suite above. Keep training/validation/test
boundaries intact: model selection and pruning settings use validation, with the
test split reserved for final reporting. Changes to preprocessing or split files
must be reflected in checkpoint provenance so incompatible weights are rejected.

The supported pruning command currently applies L1 unstructured masks. Structured
channel pruning helpers are experimental and need Stage 7 validation. Model
commands do not currently support multiclass training, configurable image sizes,
training-seed arguments, or quick-subset flags; avoid inventing CLI options.

## Documentation maintenance

The HTML guide is generated from this README and the Markdown files in `docs/`.
After editing them, rebuild the offline page using any Python 3.13 environment:

```powershell
python tools/build_docs.py
Start-Process docs/index.html
```

The builder uses only the Python standard library. Keep CLI parameter tables and
examples aligned with `python -m breakhis <command> --help`. The page embeds its
styles and scripts, so it can be opened directly or printed to PDF from a browser.
