# Run and compare experiments

This is the main workflow for the university research project. Setup, prepare
one split, choose a config, run, inspect results and repeat.
Technical checks for invalid inputs, duplicate/group leakage and missing classes
remain active. Existing individual package commands remain supported.

**Development is the default:** the runner trains, selects weights and reports on
validation. It never automatically evaluates test images. M1.4 will add the bound
final-inference and phase-specific comparison commands; those are not available
for the new schema-2 runs yet. Explicit low-level `evaluate --split test` remains
available, but test reports must not guide development choices.

Optional preparation without fitting:

```powershell
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id prepared-baseline --prepare-only
# Execute that prepared run with its identical config:
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id prepared-baseline --resume
```

Preparation checks full-source metadata and train/validation files only; test
pixels are unnecessary. `--prepare-only --resume` revalidates without execution.
Previously prepared runs require unchanged package source hashes; use a fresh ID
after updating code. Old schema-1 runs remain readable and are never migrated or
resumed through the new development runner.

## 1. Set up and prepare a split once

Open PowerShell at the repository root and activate your existing environment:

```powershell
.\.venv313\Scripts\Activate.ps1
python tools/check_environment.py --device cuda
```

For a fresh installation use the [environment guide](ENVIRONMENT.md). A CPU
installation can use `--device cpu` in the checker and runner. No model weights
are downloaded by the experiment examples; initialization is from scratch.

If `artifacts/data/manifest.json` already exists, reuse it for comparable runs.
Otherwise:

```powershell
python -m breakhis prepare-data --dataset-root ../Dataset --sources cancer_binary --output artifacts/data --seed 101
```

The manifest records filename groups, exact-image fingerprints and split
assignments. Linked duplicates and filename groups stay together. Preparation
refuses to overwrite an existing manifest/audit: choose another output folder
for a new split and update the config. Different split seeds create different
experiments, not directly matched method comparisons.

## 2. Run a baseline and a pruning experiment

```powershell
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102
python run_experiment.py --config configs/experiments/vgg_channel.json --run-id channel-102
```

Both examples use the same saved split, initialization seed and dense-training
recipe. The channel experiment trains its own reproducible dense reference,
removes channels and fine-tunes the pruned network. It evaluates dense, pruned
and fine-tuned checkpoints on validation after fitting finishes. It does not
search candidate sparsities using test results. Unstructured pruning can be
selected by changing `pruning.method` to `unstructured` in a copied config.

To start small, copy the baseline config, change `epochs` to 1 and give it a new
name/run ID. One epoch still processes the selected real split; the generated
smoke test below is the quick check that avoids processing the full dataset.

For another fresh seed, use a new run ID:

```powershell
python run_experiment.py --config configs/experiments/vgg_baseline.json --seed 103 --run-id baseline-103
python run_experiment.py --config configs/experiments/vgg_channel.json --seed 103 --run-id channel-103
```

`--seed` changes fitting/initialization randomness, not the saved split. The
override is stored in the resolved config. `--device cpu` is another supported
recorded override. Evaluation device is an independent fixed config setting;
the examples train on the available device and benchmark on CPU.

## 3. Inspect saved results

```text
artifacts/experiments/<run-id>/
  config.json                    resolved experiment settings
  manifest.json                  exact saved split bytes
  inputs.json                    split hash, counts, runtime, commit/source hashes
  run.json                       stage commands, status, timings, output hashes
  logs/                          individual train/prune/fine-tune/evaluate logs
  train.history.json             epoch losses and validation selection scores
  finetune.history.json          present for pruning runs
  development/
    dense.metrics.json           validation metrics, size and resident-forward latency
    dense.predictions.json       ordered validation image labels/probabilities
    pruned.* / finetuned.*        present for pruning runs
    selection.json               selected weight hashes/epochs and report bindings
    results.json                 validation results and designated pipeline role
  results.json                   phase references; final stays null
```

Checkpoints stay outside the codebase:
`../checkpoint/binary/<magnification>/<model>/<run-id>/`.
That directory contains best checkpoints, completed-epoch `*-last.pth` files,
command configs and histories. Multiclass checkpoint storage remains separate;
this experiment runner currently supports binary training only.

Follow a running stage's log in another PowerShell terminal:

```powershell
Get-Content artifacts/experiments/baseline-102/logs/train.log -Wait -Tail 10
```

## 4. Resume versus rerun

An interrupted training/fine-tuning stage resumes from its last completed epoch:

```powershell
python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102 --resume
```

Use the same config and overrides as the original invocation. Config, manifest,
source hashes and runtime/device must match; completed stage outputs must remain
unchanged. Every required output, including histories, predictions and configs,
is checked by hash before a completed stage is skipped. Workers are fixed at zero to preserve the existing exact-resume
contract. Partial-epoch weights are not used for exact resume. Prune/report stages may be retried when not marked completed. Training retries
must have a committed epoch-zero or completed-epoch boundary. A small epoch
journal permits repair of interrupted best/history publication to that boundary;
unexplained missing/corrupt files fail. Inspect the stage log for child errors.
Histories record the learning rate used, next rate, optimizer steps, training and
validation seconds, losses and selection score for each committed epoch.

A **fresh rerun** uses a new ID without `--resume`. Completed runs are never
silently overwritten or extended by changing their planned epoch count.

## 5. Final evaluation and comparison

M1.4 implements final inference from saved development selections and phase-aware
comparison tables. Until then, inspect each run's `development/results.json` and
validation reports. The existing `compare_experiments.py` handles old schema-1
completed runs only; it does not compare these new schema-2 development runs.

Fine-tuning adds epochs: include an equal-total-budget dense control before
attributing improvements to pruning. Seed repeats are not independent patient
samples. Match hardware and training budgets; masked zeros do not automatically
speed up dense operations. Final test metrics must never select the best recipe.

## Minimal evaluation protocol

- Use one saved group-separated train/validation/test split for the first study.
- Training uses only train; checkpoint choice uses validation. Default criterion:
  group balanced accuracy, ties by lower validation loss and then earlier epoch.
- Primary group probability averages within each available magnification, then
  averages the magnification means equally. Report image-weighted grouping and
  image metrics secondarily. Group identity comes from filenames.
- The fixed binary operating point is argmax (0.5 reference; ties to benign).
  Separate threshold calibration, constrained candidate search and nested
  cross-validation are not part of this minimal runner.
- Preprocessing uses the existing aspect-preserving resize-256/crop-224 and
  ImageNet normalization. Training adds random crop and horizontal flip; all
  evaluation preprocessing is deterministic. This deliberately reuses tested
  code rather than adding another transform pipeline.
- Example training is Adam, learning rate 0.0001, weight decay 0.0001, no
  scheduler. Full parameters are in the [config reference](EXPERIMENT_CONFIG.md).
- CPU resident-forward latency is primary; a config can instead select GPU.
  Timing is one session per evaluation; repeat sessions before performance claims.

These implemented settings supersede conflicting earlier pilot defaults for
the simple workflow. Prior dataset use is acknowledged; results are held-out
within this study, without historical-untouched or verified-patient claims.
Literature review proceeds alongside the experiments and supports later claims.

## Quick implementation check

```powershell
python tools/smoke_experiments.py
python -m unittest discover -s tests -v
```

The smoke workflow generates 12 images, trains one-epoch dense/channel-pruned
runs, saves validation predictions, selection bindings and histories, and checks
that matching seeds reproduce dense weights. Its scores and timing are not
research results. Real experiments still need adequate training and replicated
evaluation. See the [five-milestone plan](../research/EXPERIMENT_ROADMAP.md).
