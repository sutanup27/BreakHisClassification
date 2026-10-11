# BreaKHis classification and pruning

The primary workflow classifies BreaKHis histology images as **benign** or
**malignant**, separately at 40X, 100X, 200X, or 400X magnification. Training
derives the classifier output count from the dataset labels. The loader rejects
folders that do not contain exactly these two classes.

## Primary workflow

Run commands from this repository directory with the project dependencies installed:

```powershell
python Train.py --dataset breakhis --magnification 40X --model vgg
python Train.py --dataset breakhis --magnification 400X --model resnet
python Main.py --dataset breakhis --magnification 400X
```

`Main.py` uses the same training workflow and defaults to ResNet. Dataset selection
is required at the command line; CIFAR-10 is not a primary training option.
Use `--epochs`, `--batch-size`, or `--data-root` to override the training settings.
The default data location is the sibling `../Dataset/BreaKHis_v2` directory:

```text
Dataset/BreaKHis_v2/
  40X/benign/...
  40X/malignant/...
  100X/benign/...
  100X/malignant/...
  200X/benign/...
  200X/malignant/...
  400X/benign/...
  400X/malignant/...
```

Both `DataPreprocessing` and the compatibility path
`DataProcess.DataPreprocessing` now load BreaKHis. Python callers can explicitly
pass `dataset="breakhis"`; older BreaKHis callers retain that default.
Checkpoints go to `checkpoint/<magnification>/<model>/`.

## Separate legacy experiments

CIFAR-10 experiments and the mixed pruning tutorial are preserved under
`experiments/cifar10/`. See its README before using them. Shared architecture
definitions remain in `Models/`; their historical default class counts are not
used by primary training.

## Verification and remaining work

```powershell
python -m unittest discover -s tests -v
```

The workflow separation tests run without PyTorch. They check dataset routing,
folder-label validation, split-label discovery, and CLI choices. They are not
a replacement for a PyTorch data-loading or training smoke test.

This change separates datasets only. Patient-separated splitting, independent
validation, consistent preprocessing, pruning-loop repairs, package-import
cleanup, dependency setup, and checkpoint format standardization remain pending.
Existing training still selects models using the test split, so its reported
accuracy should not yet be treated as an independent final evaluation.
