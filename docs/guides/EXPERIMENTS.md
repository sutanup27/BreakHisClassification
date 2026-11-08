# Run and compare experiments

This is the main workflow for the university research project. Setup, prepare
one split, choose a config, run, inspect results and repeat.
Technical checks for invalid inputs, duplicate/group leakage and missing classes
remain active. Existing individual package commands remain supported.

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
and fine-tuned checkpoints only after development phases finish. It does not
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
  dense.metrics.json             image/group metrics, model size and latency
  dense.predictions.json         ordered image identities, labels and probabilities
  pruned.* / finetuned.*          present for pruning runs
  results.json                   final result and all report references
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
unchanged. Workers are fixed at zero to preserve the existing exact-resume
contract. Partial-epoch weights are not used for exact resume. If pruning
publication fails with a partially occupied destination, the runner refuses
overwrite; preserve that run and start a new ID after inspecting the log.

A **fresh rerun** uses a new ID without `--resume`. Completed runs are never
silently overwritten or extended by changing their planned epoch count.

## 5. Compare completed experiments

```powershell
python compare_experiments.py --runs baseline-102 channel-102 --output artifacts/comparisons/baseline-vs-channel.json
```

This writes JSON and a CSV table. Split hash, preprocessing, optimizer/selection
settings, evaluation device/protocol and runtime must match. The table includes
all dense/pruned/fine-tuned roles, seed, model, training/fine-tuning epochs,
group sensitivity/specificity/balanced accuracy, image accuracy, parameter count
and median latency. Seeds, models and epoch budgets may differ and are displayed
explicitly; this is descriptive comparison, not an automatic fairness verdict.

Fine-tuning uses additional training time. Include an equal-total-budget dense
control before attributing improvements to the pruning method. Report runtime,
epoch budgets and seed variation. Repeated seeds are not independent patient
samples. Compare speed on the same hardware; masked zeros do not automatically
speed up dense operations. Do not choose settings from these final test tables.

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
runs, saves predictions and histories, exports comparison JSON/CSV and checks
that matching seeds reproduce dense weights. Its scores and timing are not
research results. Real experiments still need adequate training and replicated
evaluation. See the [five-milestone plan](../research/EXPERIMENT_ROADMAP.md).
