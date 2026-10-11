# Package and command reference

## Canonical package

`breakhis` is the canonical package. Implementations live in:

```text
breakhis/
  cli.py               # Command dispatch
  data.py              # Manifest-backed loading
  manifest.py          # Audit, hashing, grouping, splitting
  training.py          # Shared training/evaluation routines
  checkpoints.py       # Model construction and checkpoint reconstruction
  models/              # Canonical VGG and architecture definitions
  plots.py             # Shared plots
  metrics.py           # Shared metric utilities
  pruning_legacy.py    # Historical VGG pruning helpers
  legacy_utils/        # Historical architecture-specific utilities
```

Root modules, `Models/`, `DataProcess/`, and `Utills/` preserve compatibility imports.
The duplicate VGG and training modules delegate to shared implementations.
Historical architecture-specific utilities remain available as importable helpers;
that does not certify their pruning algorithms or make them supported commands.

From the repository directory, `python -m breakhis` works without installation.
After installing dependencies, optionally install the local package in editable mode:

```powershell
python -m pip install -e . --no-deps --no-build-isolation
breakhis --help
```

The declared tested Python range is 3.13. Module commands can also be launched
outside the repository after editable installation. Explicit relative argument
paths are relative to the current working directory; default artifact paths are
resolved against this repository. Default checkpoints are resolved against its parent directory.

## Supported commands

Use the existing patient manifest with the remaining v1 binary source; see
[DATASETS.md](DATASETS.md). Rebuilding currently requires the removed v2 source.
The following supported pipeline uses a shared experiment ID:

```powershell
python -m breakhis train --dataset breakhis --source cancer_binary --magnification 40X --model vgg --epochs 20 --experiment-id baseline

python -m breakhis prune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/train.pth --sparsity 0.5 --experiment-id baseline

python -m breakhis finetune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/prune.pth --epochs 5 --experiment-id baseline

python -m breakhis evaluate --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline/finetune.pth --experiment-id baseline
```

Supported baseline architectures are VGG and torchvision ResNet18 (`resnet` is
an alias). ResNet pretrained weights download only when `--pretrained` is supplied.
Other custom architectures remain experimental and are not CLI model choices.

The supported `prune` command uses PyTorch L1 unstructured weight pruning on
convolution and linear layers. Masks are saved and reconstructed, and remain
effective during fine-tuning. `--method channel` physically removes VGG channels
and adjusts batch normalization, downstream inputs and classifier columns.
Residual/depthwise channel models remain unsupported. Complete per-layer ratios
can be supplied with `--layer-sparsities`. See [PRUNING.md](PRUNING.md).

Common options include `--manifest`, `--source`, `--magnification`, `--data-root`,
`--batch-size`, `--num-workers`, `--device`, and `--deterministic`.
Workers default to zero for simpler interruption handling on Windows. Use
`--num-workers 4` only after checking throughput on your machine.

Each improvement in validation accuracy saves the selected checkpoint. Ctrl+C
also attempts to save partial weights to an `-interrupted.pth` file. Every completed
epoch saves an independent `-last.pth` with optimizer, scheduler, and RNG state.
Use `--resume` with that file; see [CHECKPOINTS.md](CHECKPOINTS.md).

## Compatibility entry points

| Existing command | Delegates to |
| --- | --- |
| `python Train.py ...` | `python -m breakhis train ...` |
| `python Main.py ...` | Training with default `--model resnet` |
| `python Test.py ...` | `python -m breakhis evaluate ...` |
| `python PruneEvaluator.py ...` | Evaluation |
| `python PruneTrain.py ...` | Unstructured pruning |
| `python PruneReTrain.py ...` | Fine-tuning |
| `python PruneViewer.py ...` | Evaluation; old sensitivity plotting is available only as a helper |

Command scripts have main guards and perform no experiments on import. Legacy
checkpoint paths, magnifications, and pruning dictionaries are no longer hardcoded.
Manual preview/pruning demonstrations run only when explicitly invoked.

The compatibility two-loader API returns **training and validation**, not the
reserved test set. Use `breakhis.data.get_split_loaders` for explicit access to
all three named splits. Changing split fractions requires rebuilding the manifest.

## Checkpoints and limits

Checkpoint storage is alongside the dataset and codebase directories:

```text
BreakHis/
  Dataset/
  BreakHisClassification/
  checkpoint/
    binary/
      40X/vgg/<experiment-id>/train.pth
      40X/vgg/<experiment-id>/prune.pth
      40X/vgg/<experiment-id>/finetune.pth
      legacy/             # Existing files; provenance is not inferred
    multiclass/           # Reserved for future multiclass training
```

Training, pruning, fine-tuning, and interrupted saves use this layout. A bare
`--output baseline.pth` saves inside `checkpoint/binary/`; an explicit path must
stay inside the task directory. Outputs outside that directory are rejected.
Evaluation `--output` remains a report path. Checkpoint inputs can be anywhere.
The default source is now `cancer_binary`; the local v2 copy was removed.
Multiclass training is not yet supported, but its checkpoint directory is separate.

New commands save state dictionaries with architecture, labels, dataset source,
magnification, preprocessing, manifest hash, and epoch metadata. Loading rejects
a different source, magnification, preprocessing policy, or manifest.

Existing whole-model pickle checkpoints are not automatically migrated or loaded.
Their original experiment split/provenance is unknown; keep them as historical
artifacts until an explicit migration is implemented. Re-train with the patient
manifest to obtain checkpoints suitable for the new workflow.

The preserved `experiments/cifar10/` scripts and notebooks are legacy reference
material. They are outside the supported command surface and may retain old
`PruningNAS` names and unresolved logic.

## Stage 6 configuration

All model commands accept `--experiment-id`, generate distinct default run directories,
and save resolved arguments beside their output as `.config.json`. Existing new-run
checkpoint outputs are rejected. Evaluation writes a report by default.
Train/fine-tune accept `--resume` with saved effective defaults; completed epochs,
optimizer, scheduler and RNG state are restored. Exact resume requires workers zero
and the original schedule/runtime. See [CHECKPOINTS.md](CHECKPOINTS.md) and the
[README parameter tables](../README.md).
