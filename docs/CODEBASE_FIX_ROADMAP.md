# Codebase repair roadmap

Updated: 11 October 2026

## Objective

Build a reproducible BreaKHis workflow that loads the intended dataset, trains
a classifier, saves and reloads it, applies pruning, fine-tunes the pruned model,
and measures accuracy and efficiency using independent evaluation data.

Complete the stages in order. Each stage's target is the condition for proceeding
to the next stage. Use small smoke runs before committing to full training.
Eight-class classification is a future extension, not part of the initial binary baseline.

## Current state

| Area | Status |
| --- | --- |
| Separate BreaKHis and CIFAR workflows | Implemented locally; full runtime verification pending |
| Explicit primary dataset selection and label-derived class count | Implemented locally |
| Workflow separation checks | Eight checks passed using dataset doubles and CLI/static checks |
| Existing Python files | Syntax checked; this does not prove runtime correctness |
| Python environment | User activated `.venv313` using Python 3.13; dependency installation remains unconfirmed |
| PyTorch | Last check found no `torch` or `torchvision` in `.venv313` |
| Full baseline training and pruning | Not verified |

## Stage 1 — Separate dataset workflows

**Status:** Implemented locally; runtime acceptance pending.

**Issues addressed:** Mixed BreaKHis/CIFAR experiments and incorrect classifier class counts.

### Steps

- [x] Make BreaKHis binary classification the primary training workflow.
- [x] Move CIFAR experiments and mixed tutorial notebooks into `experiments/cifar10/`.
- [x] Require explicit dataset selection in the primary training command.
- [x] Derive VGG and ResNet output counts from loaded labels.
- [x] Validate that the primary dataset has `benign` and `malignant` folders.
- [x] Require explicit CIFAR selection in the isolated CIFAR loader; make downloads opt-in.
- [x] Document supported commands and the limitations of preserved legacy experiments.
- [ ] Verify a real torchvision batch and a two-class forward pass after dependencies are installed.

### Target

The primary command loads BreaKHis images and produces two logits per image.
Selecting CIFAR through the primary command fails clearly. Legacy experiment
files remain preserved, and primary training does not depend on them.

**Files:** `Train.py`, `Main.py`, `DataPreprocessing.py`,
`DataProcess/DataPreprocessing.py`, `VGG.py`, `experiments/cifar10/`, `README.md`.

## Stage 2 — Establish a reproducible environment

**Status:** Pending; virtual environment activation is complete.

**Issues addressed:** Missing dependencies, machine-specific CUDA nightly pins,
and missing declared imports.

### Steps

- [ ] Confirm `.venv313` uses the intended Python executable and pip.
- [ ] Install a compatible stable PyTorch/torchvision pair for the local NVIDIA GPU.
- [ ] Install the dependencies imported by supported primary commands.
- [ ] Remove unnecessary imports; declare any dependencies that remain required.
- [ ] Replace the existing nightly environment snapshot with a tested dependency specification.
- [ ] Document CPU and GPU setup separately, including the tested Python version.
- [ ] Exclude virtual environments and generated files from Git.
- [ ] Verify imports, CUDA availability, and a small tensor operation.

### Target

A fresh environment can install the documented dependencies and run the primary
imports without errors. Record the tested Python, PyTorch, torchvision, CUDA
runtime, and driver versions. Confirm GPU execution with a tensor computation.

**Files:** `requirement.txt` or its documented replacement, `.gitignore`, `README.md`.

## Stage 3 — Repair package imports and command structure

**Status:** Pending.

**Issues addressed:** Missing `PruningNAS` package references, duplicate modules,
import side effects, and inconsistent launch locations.

### Steps

- [ ] Choose one package name and use it consistently in supported modules.
- [ ] Replace stale imports and consolidate duplicate training, model, and utility implementations.
- [ ] Move execution into explicit functions guarded by `if __name__ == "__main__"`.
- [ ] Provide documented entry points for training, evaluation, pruning, and fine-tuning.
- [ ] Define which architectures and legacy experiments are supported.
- [ ] Keep compatibility wrappers only where they preserve a documented interface.

### Target

Supported modules import without starting training, loading checkpoints, opening
plots, or downloading data. Every documented command launches from its documented
working directory without `ModuleNotFoundError`.

**Files:** `__init__.py`, primary command scripts, `Models/`, `Utills/`, `Testing/`.

## Stage 4 — Audit datasets and create patient-separated splits

**Status:** Pending.

**Issues addressed:** Possible patient leakage, class imbalance, uncertain duplicate
content, and confusion between the two local dataset copies.

### Steps

- [ ] Select and record the dataset source for each experiment.
- [ ] Inventory supported image files and check that images decode successfully.
- [ ] Compare the two local dataset copies using filenames and content hashes;
      investigate count differences rather than assuming every file is an image.
- [ ] Create a manifest with path, label, magnification, patient ID, and content hash.
- [ ] Validate patient ID extraction against filenames or dataset metadata.
- [ ] Generate train/validation/test splits grouped by patient with a saved seed.
- [ ] Keep a patient's images together across magnifications when experiments combine them.
- [ ] Report split sizes and class distributions, and check for overlapping patients and duplicates.

### Target

Every accepted image has a traceable label and patient group. Saved splits have
no patient overlap; duplicate content across splits is resolved or explicitly
excluded. Reusing a manifest reproduces the same experiment data.

### Local dataset choices

- `../Dataset/BreaKHis_v2/<magnification>/`: two classes; current default.
- `../Dataset/dataset_cancer_v1/classificacao_binaria/<magnification>/`: two classes;
  selectable through `--data-root`.
- `../Dataset/dataset_cancer_v1/classificacao_multiclasse/<magnification>/`: eight
  classes; currently rejected by the primary binary loader.

Both binary dataset roots have 40X, 100X, 200X, and 400X folders.

**Files:** `DataPreprocessing.py`, future manifest/split utilities, saved split metadata.

## Stage 5 — Standardize preprocessing and baseline training

**Status:** Pending.

**Issues addressed:** Inconsistent image dimensions, test-set model selection,
fragile best-model tracking, and device handling.

### Steps

- [ ] Define an input size and normalization policy appropriate to the chosen model.
- [ ] Separate training augmentation from deterministic validation/test transforms.
- [ ] Use the saved patient splits rather than creating image-level random splits.
- [ ] Select the best checkpoint using validation performance only.
- [ ] Reserve the test split for final evaluation after model decisions are complete.
- [ ] Initialize best-model tracking so the first valid epoch is retained, even at zero accuracy.
- [ ] Reject invalid epoch counts, empty splits, and nonfinite training values clearly.
- [ ] Pass the device consistently and make loss aggregation sample-weighted.
- [ ] Run a small baseline smoke test before a full experiment.

### Target

One VGG or ResNet baseline completes training and validation with consistent
batch shapes, finite losses, and a valid best checkpoint. The test set does not
influence checkpoint selection or training decisions.

**Files:** `DataPreprocessing.py`, `TrainingModules.py`, `Train.py`, `Viewer.py`, model code.

## Stage 6 — Standardize configuration and checkpoints

**Status:** Partially addressed by the primary training command; remaining work pending.

**Issues addressed:** Hardcoded paths, missing output directories, incompatible
checkpoint formats, and insufficient experiment metadata.

### Steps

- [ ] Extend consistent configuration to evaluation, pruning, and fine-tuning commands.
- [ ] Resolve input paths consistently and validate missing files before computation begins.
- [ ] Create output directories and use distinct experiment identifiers.
- [ ] Save architecture identity/configuration, weights, label mapping, preprocessing,
      and the split-manifest reference.
- [ ] Save optimizer, scheduler, epoch, and random state when training resume is supported.
- [ ] Save pruning masks or channel configuration needed to reconstruct pruned architectures.
- [ ] Replace reliance on pickled whole-model objects with a documented reconstruction format.
- [ ] Verify checkpoint loading in a fresh process and inference on the intended device.

### Target

Dense and pruned checkpoints reload without relying on an old directory layout
or class import path. Reloaded predictions match within a declared numerical
tolerance. Supported training resumes restore optimizer and scheduler state.

**Files:** Training/evaluation/pruning entry points, model constructors, future checkpoint utilities.

## Stage 7 — Repair pruning and fine-tuning

**Status:** Pending.

**Issues addressed:** Unused pruning callbacks, incorrect scheduler argument,
undefined pruning variables, and unverified structural pruning.

### Steps

- [ ] Pass mask callbacks to training and reapply masks after optimizer updates.
- [ ] Pass the actual epoch into training and step the supplied scheduler correctly.
- [ ] Define or reconstruct `pruner` and callback selection before fine-tuning.
- [ ] Standardize return values across training implementations and their callers.
- [ ] Validate sparsity boundaries, tiny tensors, tied weights, and missing layer settings.
- [ ] Verify channel dimensions through convolutions, batch normalization, residual
      connections, depthwise operations, and classifiers for each supported architecture.
- [ ] Implement fine-grained pruning on the verified baseline first; add channel pruning next.
- [ ] Select pruning ratios and fine-tuned checkpoints using validation data only.
- [ ] Confirm that masks and channel configuration survive checkpoint reload.

### Target

Masked weights remain zero through fine-tuning, the learning rate follows the
configured schedule, and supported pruned models complete forward/backward passes.
Reloading preserves pruning behavior. Unsupported configurations fail clearly.

**Files:** `TrainingModules.py`, `PruneTrain.py`, `Utill.py`,
`Utills/PrunUtillFGP.py`, `Utills/PrunUtillCP.py`, supported fine-tuning entry point.

## Stage 8 — Correct evaluation and efficiency reporting

**Status:** Pending.

**Issues addressed:** CIFAR-sized profiling inputs, misleading sparse-size
estimates, and inconsistent latency comparisons.

### Steps

- [ ] Use the experiment's actual input dimensions for MAC and latency measurements.
- [ ] Report accuracy, precision, recall, F1, confusion matrix, and relevant patient-level metrics.
- [ ] Preserve class labels and metric averaging conventions in reports.
- [ ] Report dense parameters, nonzero parameters, sparsity, actual checkpoint bytes,
      and theoretical storage estimates as separate quantities.
- [ ] Benchmark using fixed hardware, input shape, batch size, warm-up, and repeated runs.
- [ ] Synchronize GPU operations for timing and report the device/runtime used.
- [ ] Compare dense, pruned-before-fine-tuning, and pruned-after-fine-tuning models.
- [ ] Report measured efficiency changes even when pruning gives no practical speedup.

### Target

A saved comparison report can be reproduced from checkpoints and configuration.
It distinguishes measured file size and latency from theoretical sparse-storage
estimates and documents the accuracy/efficiency tradeoff.

**Files:** `Model_Evaluation.py`, supported evaluation command,
`Utills/EvaluatiorUtills.py`, latency utilities, report generation code.

## Stage 9 — Add regression checks and operational documentation

**Status:** Initial separation tests exist; broader coverage pending.

**Issues addressed:** Manual-only demonstrations, weak setup documentation, and
lack of checks against recurring correctness failures.

### Steps

- [ ] Add regression tests for the repaired data, training, checkpoint, and pruning failures.
- [ ] Add a small actual PyTorch integration test using temporary synthetic images.
- [ ] Check patient split isolation, output shapes, mask persistence, and checkpoint round trips.
- [ ] Add lightweight CPU continuous integration; keep full dataset/GPU experiments separate.
- [ ] Document environment setup, supported datasets, commands, outputs, and troubleshooting.
- [ ] Mark legacy demonstrations and incomplete architectures clearly.
- [ ] Record the limitations of the final experiment and the supported operating environment.

### Target

A new user can reproduce the smoke workflow from the README. Automated checks
catch the repaired correctness failures without requiring the full dataset or a GPU.

**Files:** `tests/`, future CI workflow, `README.md`, dependency documentation.

## Final acceptance run

- [ ] Install dependencies into a fresh virtual environment.
- [ ] Validate the chosen dataset and saved patient splits.
- [ ] Train one binary baseline and select its checkpoint using validation data.
- [ ] Reload the baseline in a fresh process.
- [ ] Prune the baseline and verify its structural or mask invariants.
- [ ] Fine-tune using validation selection and save/reload the result.
- [ ] Evaluate the selected baseline and pruned model on the reserved test data.
- [ ] Generate the accuracy, size, computation, and latency comparison report.
- [ ] Save configuration, split references, checkpoint metadata, and logs.

Do not mark the codebase complete on syntax checks or a single successful import.
Completion requires this reproducible end-to-end run and the relevant regression checks.
