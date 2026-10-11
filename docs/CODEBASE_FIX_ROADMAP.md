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
| Separate BreaKHis and CIFAR workflows | Stage 1 complete; real BreaKHis batch and forward-pass checks passed |
| Explicit primary dataset selection and label-derived class count | Implemented locally |
| Regression checks | Fifteen Stage 3/4 tests passed in CPU and CUDA environments |
| Existing Python files | Syntax checked; this does not prove runtime correctness |
| Python environment | Stage 2 complete; existing CUDA and fresh CPU environments verified with Python 3.13.14 |
| PyTorch | Verified PyTorch 2.11.0+cu128 and CUDA execution in `.venv313` |
| Full baseline training and pruning | Tiny real-data train/prune/fine-tune/test smoke passed; full baseline quality remains unverified |

## Stage 1 — Separate dataset workflows

**Status:** Complete; runtime acceptance verified on 11 October 2026.

**Issues addressed:** Mixed BreaKHis/CIFAR experiments and incorrect classifier class counts.

### Steps

- [x] Make BreaKHis binary classification the primary training workflow.
- [x] Move CIFAR experiments and mixed tutorial notebooks into `experiments/cifar10/`.
- [x] Require explicit dataset selection in the primary training command.
- [x] Derive VGG and ResNet output counts from loaded labels.
- [x] Validate that the primary dataset has `benign` and `malignant` folders.
- [x] Require explicit CIFAR selection in the isolated CIFAR loader; make downloads opt-in.
- [x] Document supported commands and the limitations of preserved legacy experiments.
- [x] Verify a real torchvision batch and a two-class forward pass after dependencies are installed.

### Verification evidence

- Loaded actual `BreaKHis_v2/40X` training and test batches with labels
  `benign` and `malignant`.
- VGG and ResNet18 each returned finite outputs shaped `(2, 2)` for a batch of
  two images, using CUDA in `.venv313`.
- Training inputs were `(2, 3, 224, 224)`; test inputs were `(2, 3, 460, 700)`.
  These were the dimensions at Stage 1 acceptance. The later evaluation
  performance change standardized both outputs to 224×224.
- The primary loader rejected `dataset="cifar10"`.
- All eight workflow separation checks passed again.
- The ResNet smoke check used `weights=None` to avoid downloading pretrained
  weights. This verifies data routing and classifier dimensions, not the full
  pretrained training command or model quality.

### Target

The primary command loads BreaKHis images and produces two logits per image.
Selecting CIFAR through the primary command fails clearly. Legacy experiment
files remain preserved, and primary training does not depend on them.

**Files:** `Train.py`, `Main.py`, `DataPreprocessing.py`,
`DataProcess/DataPreprocessing.py`, `VGG.py`, `experiments/cifar10/`, `README.md`.

## Stage 2 — Establish a reproducible environment

**Status:** Complete; verified on 11 October 2026.

**Issues addressed:** Missing dependencies, machine-specific CUDA nightly pins,
and missing declared imports.

### Steps

- [x] Confirm `.venv313` uses the intended Python executable and pip.
- [x] Install a compatible stable PyTorch/torchvision pair for the local NVIDIA GPU.
- [x] Install the dependencies imported by supported primary commands.
- [x] Remove unnecessary imports; declare any dependencies that remain required.
- [x] Replace the existing nightly environment snapshot with a tested dependency specification.
- [x] Document CPU and GPU setup separately, including the tested Python version.
- [x] Exclude virtual environments and generated files from Git.
- [x] Verify imports, CUDA availability, and a small tensor operation.

### Verification evidence

- Existing `.venv313`: Python 3.13.14, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
  CUDA runtime 12.8, cuDNN 9.19.0, RTX 5070 Ti, driver 610.88.
- Fresh `.venv-stage2-cpu`: created separately and installed successfully from
  `requirements-torch-cpu.txt` followed by `requirements.txt` with `constraints.txt`.
  PyTorch 2.11.0+cpu and torchvision 0.26.0+cpu verified.
- Both environments passed primary-module imports, tensor multiplication, image
  preprocessing, a finite two-class VGG forward pass, and `pip check`.
- All eleven regression tests passed in both environments.
- The fresh environment contains neither pygame nor OpenCV; primary imports
  and regression tests still passed after unused imports were removed.
- A dry run confirmed the compatibility `requirement.txt` matches the existing
  CUDA installation without requesting package changes.
- `git check-ignore` confirmed exclusion of virtual environments, checkpoints,
  and generated verification reports. Reports are saved in ignored
  `artifacts/stage2-cpu.json` and `artifacts/stage2-cuda.json`.
- CPU installation was tested from scratch. CUDA execution was tested in the
  existing user environment; a second CUDA installation was not created.

### Target

A fresh environment can install the documented dependencies and run the primary
imports without errors. Record the tested Python, PyTorch, torchvision, CUDA
runtime, and driver versions. Confirm GPU execution with a tensor computation.

**Files:** `requirements.txt`, `constraints.txt`, `requirements-torch-cpu.txt`,
`requirements-torch-cu128.txt`, compatibility `requirement.txt`, `.gitignore`,
`docs/ENVIRONMENT.md`, `tools/check_environment.py`, `README.md`, cleaned primary imports.

## Stage 3 — Repair package imports and command structure

**Status:** Complete for the supported command surface; verified on 11 October 2026.

**Issues addressed:** Missing `PruningNAS` package references, duplicate modules,
import side effects, and inconsistent launch locations.

### Steps

- [x] Choose one package name and use it consistently in supported modules.
- [x] Replace stale imports and consolidate duplicate training, model, and utility implementations.
- [x] Move execution into explicit functions guarded by `if __name__ == "__main__"`.
- [x] Provide documented entry points for training, evaluation, pruning, and fine-tuning.
- [x] Define which architectures and legacy experiments are supported.
- [x] Keep compatibility wrappers only where they preserve a documented interface.

### Verification evidence

- Canonical implementations reside in `breakhis`; root/`Models`/`Utills` modules
  are compatibility imports. Duplicate VGG and training definitions were consolidated.
- No `PruningNAS` imports remain in the canonical package. CIFAR reference
  notebooks remain explicitly outside the supported workflow.
- Editable installation and the `breakhis` executable succeeded without dependency changes.
- All five commands expose working help: prepare-data, train, evaluate, prune, finetune.
- Import tests prohibit checkpoint loads, image scans, plot display, or output on import.
- A 12-image real-data GPU smoke ran all four model commands and reloaded checkpoints.
- Fifteen regression tests passed in both the existing CUDA and isolated CPU environments.
- Supported architectures: VGG and torchvision ResNet18. Supported pruning:
  built-in L1 unstructured masks. Custom architectures/channel pruning remain experimental.
- Old whole-model checkpoints require explicit migration; new commands do not
  silently load them. See `docs/COMMANDS.md` for compatibility changes.

### Target

Supported modules import without starting training, loading checkpoints, opening
plots, or downloading data. Every documented command launches from its documented
working directory without `ModuleNotFoundError`.

**Files:** `breakhis/`, `pyproject.toml`, primary compatibility scripts, `Models/`,
`Utills/`, `Testing/`, `docs/COMMANDS.md`, import/command regression tests.

## Stage 4 — Audit datasets and create patient-separated splits

**Status:** Complete for the local filename-based patient/slide grouping policy;
verified on 11 October 2026.

**Issues addressed:** Possible patient leakage, class imbalance, uncertain duplicate
content, and confusion between the two local dataset copies.

### Steps

- [x] Select and record the dataset source for each experiment.
- [x] Inventory supported image files and check that images decode successfully.
- [x] Compare the two local dataset copies using filenames and content hashes;
      investigate count differences rather than assuming every file is an image.
- [x] Create a manifest with path, label, magnification, patient ID, and content hash.
- [x] Validate patient ID extraction against filenames or dataset metadata.
- [x] Generate train/validation/test splits grouped by patient with a saved seed.
- [x] Keep a patient's images together across magnifications when experiments combine them.
- [x] Report split sizes and class distributions, and check for overlapping patients and duplicates.

### Verification evidence

- All 7,900 BreaKHis images and both 7,909-image cancer representations decoded;
  no corruption exclusions were needed. Each source has 125 repeated decoded images,
  excluded from its training manifest without modifying originals.
- All 7,900 shared binary filenames differ in decoded pixels; cancer images are
  224×224, while BreaKHis is mostly 700×460. Cancer has nine additional filenames.
- Seed-zero group-stratified allocation gives 57 training, 13 validation, and
  11 test patient/slide groups. All four magnifications share those assignments.
- The identifier rule preserves the full YEAR-SLIDE_ID including suffix letters,
  following the official filename structure. The local census is 81 groups; no
  independent patient-to-slide metadata is available to verify distinct people.
- Exact decoded-content links also keep matching images together across splits.
  Audit reports zero patient overlap and zero exact-content overlap.
- Saved reports include source choice, class/image distributions, patient counts,
  hashes, dimensions, and duplicate exclusions. Loading checks for modified files.
- Regression tests cover reproducible grouping, cross-patient duplicate linkage,
  label conflicts, corrupt images, missing manifests, and file changes.
- Primary training now uses saved training/validation splits. Reserved test
  evaluation is a separate command. See `docs/DATASETS.md`.
- Real batches from all three splits of both binary sources, combining all
  magnifications, produced finite two-class ResNet18 outputs.

### Target

Every accepted image has a traceable label and patient group. Saved splits have
no patient overlap; duplicate content across splits is resolved or explicitly
excluded. Reusing a manifest reproduces the same experiment data.

### Local dataset choices

- `../Dataset/BreaKHis_v2/<magnification>/`: two classes; current default.
- `../Dataset/dataset_cancer_v1/classificacao_binaria/<magnification>/`: two classes;
  selected with `--source cancer_binary`; `--data-root` can relocate that source.
- `../Dataset/dataset_cancer_v1/classificacao_multiclasse/<magnification>/`: eight
  classes; currently rejected by the primary binary loader.

Both binary dataset roots have 40X, 100X, 200X, and 400X folders.

**Files:** `breakhis/manifest.py`, `breakhis/data.py`, `breakhis/cli.py`,
`docs/DATASETS.md`, generated `artifacts/data/manifest.json` and `audit.json`, tests.

## Stage 5 — Standardize preprocessing and baseline training

**Status:** Complete for the supported binary workflow; verified on 11 October 2026.

**Issues addressed:** Inconsistent image dimensions, test-set model selection,
fragile best-model tracking, and device handling.

### Steps

- [x] Define an input size and normalization policy appropriate to the chosen model:
      resize the shorter edge to 256, crop to 224×224, and retain ImageNet normalization.
- [x] Separate training augmentation from deterministic validation/test transforms:
      random training crops/flips and deterministic evaluation center crops.
- [x] Use the saved patient splits rather than creating image-level random splits.
- [x] Select the best checkpoint using validation performance only.
- [x] Reserve the test split for final evaluation after model decisions are complete.
- [x] Initialize best-model tracking so the first valid epoch is retained, even at zero accuracy, in supported commands.
- [x] Reject invalid epoch counts, empty splits, and nonfinite training values clearly.
- [x] Pass the device consistently and make loss aggregation sample-weighted.
- [x] Run a small baseline smoke test before a full experiment.

### Target

One VGG or ResNet baseline completes training and validation with consistent
batch shapes, finite losses, and a valid best checkpoint. The test set does not
influence checkpoint selection or training decisions.

### Verification and limits

- Training and validation aggregate loss by sample count, including partial last
  batches; scalar mean and sum criterion reductions are handled explicitly.
- Inputs, outputs, losses, accumulated loss, and training gradient norms are
  checked for nonfinite values. Invalid gradients fail before an optimizer step.
- Batches, prediction inputs, and criteria follow the model's actual device;
  explicit CPU models do not follow a global CUDA default.
- Invalid epoch/loader settings and nonpositive or nonfinite learning rates are
  rejected. Empty manifest selections name the source, magnification, and split.
- Regression tests verify first-epoch checkpoint selection at zero accuracy and
  prohibit iterating the reserved test split during training.
- An actual v1 40X GPU VGG smoke completed two epochs with uneven batches
  (batch size 3, four training and four validation images), finite losses, and
  a reconstructible checkpoint under the parent binary checkpoint directory.
- Accuracy uses percentages (0-100), consistent with the shared evaluator.

This establishes baseline execution correctness, not full-dataset clinical or
classification quality. Full experiments and comparative reporting remain work
for later stages. V1-only manifest rebuilding after removing v2 is still pending.

**Files:** `breakhis/data.py`, `breakhis/training.py`, `breakhis/cli.py`,
`tests/test_training.py`, `tests/test_manifest_loading.py`, existing preprocessing tests.

## Stage 6 — Standardize configuration and checkpoints

**Status:** Complete for supported dense and unstructured binary models; verified on 11 October 2026.

**Issues addressed:** Hardcoded paths, missing output directories, incompatible
checkpoint formats, and insufficient experiment metadata.

### Steps

- [x] Extend consistent configuration to evaluation, pruning, and fine-tuning commands.
- [x] Resolve input paths consistently and validate missing files before computation begins.
- [x] Create output directories and use distinct experiment identifiers.
- [x] Save architecture identity/configuration, weights, label mapping, preprocessing,
      and the split-manifest reference.
- [x] Save optimizer, scheduler, epoch, and random state when training resume is supported.
- [x] Save pruning masks or channel configuration needed to reconstruct pruned architectures.
- [x] Replace reliance on pickled whole-model objects with a documented reconstruction format.
- [x] Verify checkpoint loading in a fresh process and inference on the intended device.

### Target

Dense and pruned checkpoints reload without relying on an old directory layout
or class import path. Reloaded predictions match within a declared numerical
tolerance. Supported training resumes restore optimizer and scheduler state.

### Verification and limits

- All model commands save resolved `.config.json` records and use distinct default
  experiment IDs under the parent binary checkpoint tree. Explicit existing
  new-run checkpoint outputs are rejected.
- Version 2 records architecture, class mapping, preprocessing, manifest hash/path,
  source/root, split policy, pruning masks, runtime and effective configuration.
  Version 1 remains readable for weights workflows; whole-model pickles remain rejected.
- Completed-epoch last checkpoints contain optimizer, scheduler, best accuracy,
  Python/NumPy/PyTorch CPU/CUDA RNG state and backend determinism settings.
- Dense and pruned stochastic CPU interruption/resume regressions produce exactly
  matching weights, optimizer, scheduler and RNG states against uninterrupted runs.
- Fresh-process VGG dense/pruned predictions match at atol 1e-6, rtol 1e-5.
- A real v1 GPU smoke interrupted after epoch one, resumed epoch two in a fresh
  process, then completed pruning, fine-tuning, and reserved-test evaluation with
  the new format and saved command configurations.

Exact resume uses completed epochs, workers zero, the original planned schedule,
configuration, runtime and device. Partial epochs are replayed; their interrupted
weights files cannot resume exactly. Background worker/prefetch state is not saved.
Full-dataset quality and structured channel architecture reconstruction are outside
this stage. V1-only manifest rebuilding remains pending after local v2 deletion.

**Files:** `breakhis/cli.py`, `breakhis/checkpoints.py`, `breakhis/paths.py`,
checkpoint/resume tests, README and `docs/CHECKPOINTS.md`.

## Stage 7 — Repair pruning and fine-tuning

**Status:** Complete for supported unstructured VGG/ResNet18 and sequential VGG channel pruning; verified on 11 October 2026.

**Issues addressed:** Unused pruning callbacks, incorrect scheduler argument,
undefined pruning variables, and unverified structural pruning.

### Steps

- [x] Pass mask callbacks to training and reapply masks after optimizer updates.
- [x] Pass the actual epoch into training and step the supplied scheduler correctly.
- [x] Define or reconstruct `pruner` and callback selection before fine-tuning.
- [x] Use shared train/evaluate return contracts and preserve documented legacy tuple
      adapters without changing their callers.
- [x] Validate sparsity boundaries, tiny tensors, tied weights, and missing layer settings.
- [x] Verify VGG channel dimensions through convolutions, batch normalization and
      classifiers; reject unsupported residual/depthwise channel configurations explicitly.
- [x] Implement fine-grained pruning on the verified baseline first; add channel pruning next.
- [x] Select pruning ratios and fine-tuned checkpoints using validation data only.
- [x] Confirm that masks and channel configuration survive checkpoint reload.

### Target

Masked weights remain zero through fine-tuning, the learning rate follows the
configured schedule, and supported pruned models complete forward/backward passes.
Reloading preserves pruning behavior. Unsupported configurations fail clearly.

### Verification and limits

- Canonical weight pruning uses stable exact mask counts, rejects invalid/empty
  tensors and tied storage, validates complete per-layer maps before mutation,
  and treats repeated ratios as total target sparsity.
- Training reapplies masks before batches and after optimizer updates, including
  matching optimizer buffers. Legacy callbacks, epochs, supplied schedulers,
  best-first-epoch selection and cloning of mask-hook models are repaired.
- VGG channel surgery preserves retained convolution weights, BN affine/running
  state/counters, downstream inputs and classifier columns. At least one channel
  remains. Zero-ratio predictions match and narrower models pass backward checks.
- Checkpoints record and validate widths and retained indices. Dense/masked/channel
  reload is supported; fresh-process channel predictions match at atol 1e-6, rtol 1e-5.
- Pruning writes a separate validation report for ratio selection. Training and
  fine-tuning continue selecting checkpoints on validation; test evaluation is separate.
- CPU tests cover pruning boundaries/ties, tiny layers, missing maps, tied weights,
  masks/momentum, callbacks/schedulers, ResNet masked forward/backward and VGG surgery.
- Actual v1 GPU VGG channel pruning, fine-tuning and fresh-process interruption/resume
  completed successfully using the existing 12-image smoke manifest.

Residual/depthwise structured pruning, mixed channel/mask models, and historical
architecture-specific channel helpers remain unsupported and fail on the supported
command surface. Automatic ratio search and measured efficiency are not implemented.
These checks do not establish full-dataset accuracy or a latency benefit.

**Files:** `breakhis/pruning.py`, `breakhis/training.py`, `breakhis/models/vgg.py`,
`breakhis/checkpoints.py`, `breakhis/cli.py`, compatibility helpers, tests and
`docs/PRUNING.md`.

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
