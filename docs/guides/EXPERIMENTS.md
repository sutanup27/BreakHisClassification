# Run and compare experiments

This is the main workflow for the university research project. Setup, prepare
one split, choose a config, run, inspect results and repeat.
Technical checks for invalid inputs, duplicate/group leakage and missing classes
remain active. Existing individual package commands remain supported.

**Development is the default:** the runner trains, selects weights and reports on
validation. It never automatically evaluates test images. Use `--phase final`
only after development choices are fixed. Final inference verifies the saved
selection and evaluates its fixed checkpoint roles; it does not fit or select
weights. Saved comparisons are phase-specific and run no inference.

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
  final/                         created only by explicit --phase final
    dense.* / pruned.* / finetuned.*  test reports and ordered predictions
    results.json                 test results tied to development selection
  results.json                   separate development/final references
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

## 5. Compare development results, then explicitly evaluate final weights

```powershell
# Development comparison is the default; reads saved validation outputs only.
python compare_experiments.py --runs baseline-102 channel-102 --output artifacts/comparisons/development.json

# After fixing experimental choices, explicitly evaluate the bound checkpoints.
python run_experiment.py --run-id baseline-102 --phase final
python run_experiment.py --run-id channel-102 --phase final
python compare_experiments.py --runs baseline-102 channel-102 --phase final --output artifacts/comparisons/final.json
```

Final reads the frozen config, manifest, selected checkpoint hashes/epochs and
policy IDs. It requests test only, needs only the configured evaluation device,
and can evaluate portable CUDA-trained weights on CPU without the training GPU
or training images. It evaluates all fixed roles, using the recipe's designated
role (`dense` or `finetuned`); it never picks the best test score. Final outputs
live under `final/`, separately from `development/`. A failed final phase leaves
development results usable. Retry an incomplete final with:

```powershell
python run_experiment.py --run-id baseline-102 --phase final --resume
```

Completed final stages are verified and skipped; completed final phases cannot
be overwritten. Missing/corrupt completed artifacts fail rather than being
silently replaced. Current package source hashes must match the saved run.
Old schema-1 records remain readable and their weights remain usable with
individual commands, but phase-aware final/comparison does not migrate them.

Comparison exports JSON and CSV without loading a model or opening images. It
requires two completed requested phases and verifies reports, predictions,
checkpoint hashes, manifests, policy IDs, training recipe and timing invariants.
It never falls back to another phase. With at least two valid runs, extra
incomplete runs appear as diagnostics, never as scored rows. Model/seed/epoch/
pruning differences are enumerated. An intentional checkpoint-selection study
must declare the selection difference explicitly:

```powershell
python compare_experiments.py --runs selection-a selection-b --vary training.selection_metric --output artifacts/comparisons/selection-study.json
```

That declaration cannot override data or measurement invariants. Rows include
planned/completed epochs, optimizer steps, measured fitting/validation/stage
seconds, selected checkpoint identity, group support, rates, MACs and latency.
Fine-tuning adds work: include equal-budget controls before attributing gains to
pruning. Disclosing budgets does not prove fairness. Seeds are not independent
patient samples; one timing session does not establish a significant speedup.
Masked zeros do not automatically speed up dense operations.

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

The smoke workflow generates 12 images and uses fresh public CLI processes for
dense, masked and channel-pruned/fine-tuned runs. Test images are unavailable
during development; fitting images are unavailable during final inference. Saved
comparison also succeeds with all pixel files unavailable. Matching seeds
reproduce dense weights, and masks/channel selections survive fine-tuning/reload.
The tool checks real training/final failures followed by valid `--resume` calls.
`development-failure`, `final-failure` and `completed-final-refusal` are expected
nonzero checks, printed as PASS when the refusal works. It retains commands,
logs, phase reports, histories, selections and development/final JSON/CSV tables. Its scores and timing are not
research results. Real experiments still need adequate training and replicated
evaluation. See the [five-milestone plan](../research/EXPERIMENT_ROADMAP.md).

## Daily operation reference

| Action | Command pattern | Result |
| --- | --- | --- |
| Prepare once | `python -m breakhis prepare-data --dataset-root ../Dataset --sources cancer_binary --output artifacts/data --seed 101` | Frozen group-separated manifest; existing output refuses overwrite |
| Fresh experiment | `python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102` | Training and validation-only development outputs |
| Fresh seed/rerun | `python run_experiment.py --config configs/experiments/vgg_baseline.json --seed 103 --run-id baseline-103` | New fitting randomness; unchanged split |
| Resume development | `python run_experiment.py --config configs/experiments/vgg_baseline.json --run-id baseline-102 --resume` | Same inputs/code/runtime; last committed epoch or verified completed stage |
| Compare development | `python compare_experiments.py --runs baseline-102 baseline-103 --output artifacts/comparisons/dev-seeds.json` | Saved validation results only; no inference |
| Explicit final | `python run_experiment.py --run-id baseline-102 --phase final` | Test-only inference using bound development selections |
| Resume final | `python run_experiment.py --run-id baseline-102 --phase final --resume` | Retry incomplete final outputs; never fitting |
| Compare final | `python compare_experiments.py --runs baseline-102 baseline-103 --phase final --output artifacts/comparisons/final-seeds.json` | Both runs must already have completed final phases |

Give every fresh experiment and comparison output a new ID/path. For final
seed comparison, explicitly finalize each listed run first. Copy configs within
`configs/experiments/` to keep their existing relative paths valid; paths resolve
from the config directory. Completed phases cannot be overwritten. An integrity
failure requires restoring the original bytes or starting an intentional new
manifest/run, never silently accepting changed images. Logs identify the failed
child under `artifacts/experiments/<run-id>/logs/`.

## M2.1: prepare a baseline study without fitting

```powershell
python run_baselines.py --study configs/baselines/m2_study.json --stage prepare
```

Copy the study JSON and choose a new `study_id` for another preparation. Paths in
that JSON resolve from its own directory; keep `experiment_template` valid if
moving the file. Existing study output refuses overwrite, including an interrupted
preparation. `--resume` is not a preparation operation.

The current command only compiles a plan and checks the full saved source metadata
and train/validation paths. It opens no image pixels, creates no model/loaders,
queries no CUDA availability, fits nothing and reserves no checkpoint directory.
Missing test pixel files do not prevent preparation. It does not resplit data.

Default counts: **22 required fits**, **3 optional alternate-recipe fits** and
**6 one-epoch pilot fits** = **31 prospective fitting jobs** before retry overhead.
Every candidate refers to the same declared parent's selection; parents are not
counted again for each ratio. Repeated VGG recipe choice and scratch topology are
unresolved dependencies, never guessed from a future result.

Outputs under `artifacts/baselines/<study-id>/`:

| File | Meaning |
| --- | --- |
| `study.json` | Effective study inputs and resource limits |
| `jobs.json` | Stable job IDs, pilot/main roles, dependencies, counts, policies and input/source hashes |
| `manifest.json`, `template.json` | Exact original input bytes; template source path is in `jobs.json` |
| `configs/<job-id>.json` | Non-runnable draft envelope, config-2 proposal and unresolved bindings |
| `cost_estimate.json` | Unknown costs/capacity reasons, or supplied plan-bound estimates |
| `cost_inputs.json` | Optional copied cost inputs when supplied |
| `status.json` | Preparing/prepared/failed state, no-fitting flag and complete output hashes |

Drafts are deliberately not accepted by `run_experiment.py`. M2.2-M2.4 will
implement config-2 model/init support, resolve parent/topology/recipe bindings and
execute jobs. `pilot`, `parents`, `candidates`, `report` currently refuse clearly;
preparation is the only implemented study stage. Existing M1 experiment commands
remain supported and unchanged.

Resource limits may remain null while planning. No hardware throughput or fit time
is measured by this command. Optional `--cost-estimates FILE` accepts the strict
plan-bound JSON documented in the [config reference](EXPERIMENT_CONFIG.md#m21-study-and-cost-inputs).
Even a sufficient capacity estimate does not enable unimplemented execution.
A shortfall never silently removes jobs. M2.4 owns actual cost profiling/launch.
