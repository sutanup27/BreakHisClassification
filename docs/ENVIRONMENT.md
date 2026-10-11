# Environment setup

The supported Stage 2 baseline is **Windows x64 with CPython 3.13.14**.
Dependency versions are pinned in `requirements.txt`; `constraints.txt` pins
the tested dependency versions. This is version pinning, not a wheel-hash lock.
Other platforms or Python versions require their own installation verification.

## Create a virtual environment

From PowerShell, enter the repository directory. Use your Python 3.13 executable;
the following AppData path was verified on this machine:

```powershell
cd D:\Sutanu_WorkSpace\BreakHis\BreakHisClassification
$python313 = "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python3.13.exe"
& $python313 -m venv .venv313
```

For an existing `.venv313`, skip creation. Use its executable directly to avoid
PowerShell activation-policy problems:

```powershell
.\.venv313\Scripts\python.exe --version
.\.venv313\Scripts\python.exe -m pip --version
```

Optional activation applies to the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv313\Scripts\Activate.ps1
```

Virtual environments require a base Python installation; activation does not
install Python. See the [Python venv documentation](https://docs.python.org/3/library/venv.html).

## NVIDIA GPU installation

Install the tested CUDA 12.8 wheels first, then the common dependencies:

```powershell
.\.venv313\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe tools/check_environment.py --device cuda
```

The installed wheels contain the CUDA runtime needed for normal training;
a compatible NVIDIA driver is required. The verified local GPU is an RTX 5070 Ti
with driver 610.88. The checked combination is PyTorch `2.11.0+cu128`, torchvision
`0.26.0+cu128`, CUDA runtime 12.8, and cuDNN 9.19.0.
Wheel selection follows [PyTorch's official 2.11.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v2110).

## CPU installation

Use a separate environment so CPU wheel installation does not replace your GPU build:

```powershell
& $python313 -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-cpu\Scripts\python.exe tools/check_environment.py --device cpu
```

CPU training will usually take longer. CPU and CUDA use the same pinned upstream
PyTorch/torchvision versions with different build variants. Install the wheel
selection and common requirements in separate pip commands; the PyTorch index
is not the package source for the remaining scientific Python dependencies.

## What the files cover

| File | Purpose |
| --- | --- |
| `requirements.txt` | Direct dependencies for primary BreaKHis training, plotting, and root evaluation/pruning utilities |
| `constraints.txt` | Tested versions for the dependency graph |
| `requirements-torch-cu128.txt` | Stable CUDA 12.8 torch/torchvision wheels from the official PyTorch index |
| `requirements-torch-cpu.txt` | Stable CPU torch/torchvision wheels from the official PyTorch index |
| `requirement.txt` | Compatibility alias for `requirements.txt`; select torch wheels first |
| `tools/check_environment.py` | Import, tensor, preprocessing, classifier, and pip consistency checks |

OpenCV and pygame were unused imports in the primary VGG and preview code and
have been removed. They are not required by primary training. The existing user
environment may still contain them; no packages were uninstalled from it.
Legacy notebooks may need additional dependencies and still contain unresolved
imports and logic. Stage 2 does not make those legacy workflows supported.

The requirements replace the old nightly CUDA snapshot. Do not use that old
snapshot to reconstruct an environment. For the reasoning behind version pinning,
see [pip's repeatable installation documentation](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

## Verify and run

```powershell
.\.venv313\Scripts\python.exe tools/check_environment.py --device cuda --output artifacts/environment.json
.\.venv313\Scripts\python.exe -m unittest discover -s tests -v
.\.venv313\Scripts\python.exe -m breakhis prepare-data --seed 0
.\.venv313\Scripts\python.exe Train.py --dataset breakhis --magnification 40X --model vgg --num-workers 0
```

The environment check imports supported primary modules without launching
training or downloading weights. It checks a tensor computation and a finite
two-class VGG forward pass on a transformed synthetic image, then runs `pip check`.
It does not verify patient splits, pruning correctness, or model quality.

Use `--num-workers 0` for simpler Ctrl+C handling; use background workers after
checking their throughput on your machine. Supported commands now save best
weights after validation and attempt an interrupted weights checkpoint on Ctrl+C.
Exact optimizer/scheduler resume remains future work. See [COMMANDS.md](COMMANDS.md).

Virtual environments, checkpoints, and generated environment reports are ignored
by Git. The disposable `.venv-stage2-cpu` directory is a separate installation
verification environment; it does not replace `.venv313`.
