# Legacy CIFAR-10 experiments

This directory preserves the exploratory CIFAR-10 code separately from the
primary BreaKHis workflow. It is not a supported end-to-end training command.

Moved here:

- The former `DataProcess/DataPreprocessing.py` and `DataProcess/DataPreview.py`.
- `DoThings.py`, `DoThings4.py`, `DoThingsy.py`, and `DoThings.ipynb`.
- `Evaluator.py`, `PruneReTrain.py`, and `DoPruneReTrain.ipynb`.
- The former `Testing/dotest.py` architecture demonstration.
- `legacy_mixed_pruning_tutorial.ipynb`, formerly
  `PruningSparsing_BreakHis_NAS(JU).ipynb`. Despite its former name, it mixes
  CIFAR tutorial code with BreaKHis experiments; preserve it as reference only.

The isolated CIFAR loader requires an explicit dataset argument and does not
download data unless requested. From the repository directory:

```python
from experiments.cifar10.DataPreprocessing import get_dataloaders

train_loader, test_loader = get_dataloaders(
    "./dataset/cifar10", dataset="cifar10", download=False,
)
num_classes = len(train_loader.dataset.classes)
```

Set `download=True` only when you intend to download CIFAR-10. Models created for
new experiments should use the labels' class count rather than a literal ten.

Legacy scripts and notebooks retain other unresolved issues, including
`PruningNAS` imports, old relative paths, missing checkpoints, and pruning bugs.
Moving them does not repair those issues. Do not load CIFAR checkpoints into the
two-class BreaKHis workflow.
