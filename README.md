# BreaKHis classification and pruning

The primary workflow classifies BreaKHis histology images as **benign** or
**malignant**, separately at 40X, 100X, 200X, or 400X magnification. Training
derives the classifier output count from the dataset labels. Primary commands
load only audited binary sources from a saved patient manifest.

## Environment setup

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the tested Python 3.13 environment,
virtual environment setup, pinned dependencies, and CPU/GPU installation commands.
For the existing NVIDIA environment:

```powershell
.\.venv313\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe tools/check_environment.py --device cuda
```

## Primary workflow

The canonical package is `breakhis`. Run from this repository directory after
[environment setup](docs/ENVIRONMENT.md):

```powershell
python -m breakhis prepare-data --seed 0
python -m breakhis train --dataset breakhis --source breakhis_v2 --magnification 40X --model vgg
python -m breakhis train --dataset breakhis --source cancer_binary --magnification 40X --model resnet
```

The existing `python Train.py --dataset breakhis ...` command remains a wrapper.
See [docs/COMMANDS.md](docs/COMMANDS.md) for training, evaluation, pruning,
fine-tuning, editable installation, checkpoint requirements, and common options.

Data loading now requires a saved patient manifest. Training selects checkpoints
using validation; the separate evaluation command uses the reserved test split.
See [docs/DATASETS.md](docs/DATASETS.md) for audit findings and split policy.
The two binary roots differ in resolution and content; choose a source explicitly.
Eight-class folders are audited but remain unsupported by primary commands.

Images are resized with aspect ratio preserved, then cropped to 224×224. Training
uses random crops/flips; evaluation uses deterministic center crops. This differs
from older 460×700 evaluation, so old accuracy figures are not directly comparable.

Workers default to zero for easier Ctrl+C handling. `--num-workers 4` enables
persistent background loading; `--deterministic` prefers deterministic cuDNN
algorithms. Timings distinguish training from validation. Best weights are saved
per epoch, and Ctrl+C attempts an interrupted weights checkpoint.

## Separate legacy experiments

CIFAR-10 experiments and the mixed pruning tutorial are preserved under
`experiments/cifar10/`. See its README before using them. Shared architecture
definitions live in `breakhis/models/` with `Models/` compatibility imports; their historical default class counts are not
used by primary training.

## Verification and remaining work

```powershell
python -m unittest discover -s tests -v
```

Manifest and CLI checks cover patient isolation, duplicate linkage, corruption,
source comparisons, and command dispatch. PyTorch checks also cover import safety,
actual image transforms, manifest loading, file-change detection, and pruned
checkpoint reconstruction.

Stages 1–4 are implemented and verified. Full baseline quality, exact training
resume, structured pruning correctness, efficiency reports, and CI remain future
work; see [docs/CODEBASE_FIX_ROADMAP.md](docs/CODEBASE_FIX_ROADMAP.md).
