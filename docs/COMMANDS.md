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

Prepare the patient manifest first; see [DATASETS.md](DATASETS.md).

```powershell
python -m breakhis prepare-data --seed 0

python -m breakhis train --dataset breakhis --source cancer_binary --magnification 40X --model vgg --epochs 20 --output ../checkpoint/binary/40X/vgg/baseline.pth

python -m breakhis prune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/baseline.pth --sparsity 0.5 --output ../checkpoint/binary/40X/vgg/pruned.pth

python -m breakhis finetune --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/pruned.pth --epochs 5 --output ../checkpoint/binary/40X/vgg/finetuned.pth

python -m breakhis evaluate --dataset breakhis --source cancer_binary --magnification 40X --checkpoint ../checkpoint/binary/40X/vgg/finetuned.pth --output artifacts/final-test.json
```

Supported baseline architectures are VGG and torchvision ResNet18 (`resnet` is
an alias). ResNet pretrained weights download only when `--pretrained` is supplied.
Other custom architectures remain experimental and are not CLI model choices.

The supported `prune` command uses PyTorch L1 unstructured weight pruning on
convolution and linear layers. Masks are saved and reconstructed, and remain
effective during fine-tuning. Structured channel pruning, architecture-specific
pruning validation, and systematic ratio selection remain Stage 7 work.

Common options include `--manifest`, `--source`, `--magnification`, `--data-root`,
`--batch-size`, `--num-workers`, `--device`, and `--deterministic`.
Workers default to zero for simpler interruption handling on Windows. Use
`--num-workers 4` only after checking throughput on your machine.

Each improvement in validation accuracy saves the selected checkpoint. Ctrl+C
also attempts to save current weights to an `-interrupted.pth` file. Optimizer,
scheduler, and RNG-state resume are not yet implemented; these files are weights
checkpoints rather than exact training-resume checkpoints.

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
      40X/vgg/train.pth
      40X/vgg/prune.pth
      40X/vgg/finetune.pth
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
