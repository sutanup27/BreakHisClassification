# Experiment configuration and checkpoint format

All supported model commands normalize input paths and check manifest/checkpoint
files and the selected dataset root before image loading or model construction.
Dataset provenance is checked before loading weights or downloading pretrained
weights. Relative explicit paths use the current working directory; default
outputs use the workspace's parent `checkpoint/binary/` directory.

## Experiment outputs

Without `--output`, model commands use:

```text
checkpoint/binary/<magnification>/<architecture>/<experiment-id>/
  train.config.json
  train.pth                   # Best validation model
  train-last.pth              # Last completed epoch; resume state
  train-interrupted.pth       # Optional partial-epoch weights; not resumable
  prune.config.json
  prune.pth
  finetune.config.json
  finetune.pth
  finetune-last.pth
  evaluate.config.json
  evaluate.json
```

Each invocation generates a UTC timestamp plus random suffix as its ID unless
`--experiment-id` is supplied. Supply the same ID across train/prune/fine-tune to
group a pipeline, with explicit checkpoint input paths. Existing new-run output
checkpoints are rejected. IDs cannot contain path separators or traversal.
Custom checkpoint outputs remain confined to the binary task directory. A bare
filename is placed directly in that directory; configuration and last/interrupted
files are placed beside it. Evaluation reports can be placed elsewhere.

Configuration JSON records resolved CLI arguments, absolute input/output paths,
actual device, architecture, experiment ID, parent checkpoint, and fixed optimizer
and scheduler settings where applicable. It is an execution record, not a separate
configuration-file input API. Command-line arguments and checkpoint resume state
remain the configuration sources.

## Format version 2

Checkpoints use tensors and standard Python values, loaded with
`torch.load(..., weights_only=True)`. They do not pickle model instances.

| Field | Contents |
| --- | --- |
| `format_version` | `2`; format `1` remains readable for weights-only workflows |
| `architecture`, `architecture_config` | Supported constructor identifier, class count, and optional VGG channel widths |
| `classes`, `class_to_idx` | Ordered class names and corresponding numeric labels |
| `state_dict` | CPU tensors, including weights, batch-normalization buffers, pruning originals and masks |
| `metadata` | Source, magnification, preprocessing description/configuration, manifest hash/path, dataset root, and split-selection policy |
| `epoch` | Completed epoch number; partial interrupted files use the last completed epoch number |
| `config` | Resolved command configuration |
| `pruning` | Dense/unstructured/channel designation, masks or validated retained-channel configuration |
| `runtime` | Python, PyTorch, and CUDA runtime versions |
| `training_state` | Optimizer, scheduler, best accuracy, planned epochs, RNGs, device, and resume-boundary policy; absent for pruning and partial weights |

Writes use a temporary sibling file and atomic replacement. Dense and supported
unstructured models reconstruct from their declared architecture; mask hooks are
reinstalled before loading state. VGG channel models reconstruct from saved widths
and retained-channel metadata. Residual/depthwise channel models and mixed
channel/mask checkpoints remain unsupported.

Version 1 checkpoints can be evaluated, pruned, or used to start a new fine-tuning
run. They cannot resume exactly. Legacy whole-model pickles are rejected rather
than loaded with unrestricted unpickling. Missing fields, inconsistent labels,
invalid format versions, and nonfinite weights are rejected.

## Resume semantics

```powershell
python -m breakhis train --dataset breakhis --experiment-id baseline --epochs 20 --num-workers 0 --deterministic
# If interrupted before the planned run completes:
python -m breakhis train --dataset breakhis --resume ../checkpoint/binary/40X/vgg/baseline/train-last.pth
```

Resume uses the same command (`train` or `finetune`) as the saved run. Omitted
options inherit the saved effective settings. The scheduler is stepped before
each completed-epoch checkpoint is saved. Restore starts at `epoch + 1`, with
optimizer momentum, learning-rate schedule, best-validation tracking, Python RNG,
NumPy RNG, PyTorch CPU RNG, CUDA RNGs, and recorded backend determinism settings.
An epoch-zero last checkpoint allows restarting if the first epoch is interrupted.

The original planned epoch count and learning rate, batch size, worker count,
deterministic mode, resolved device, output path, experiment ID, runtime versions,
and dataset provenance must match. A changed total would define a different cosine
schedule and is rejected. Completed runs require a new fine-tuning run rather
than extending resume. Restore the original best checkpoint if it was moved;
the experiment's best and last files belong together.

Exact resume is supported at completed-epoch boundaries with **num-workers 0**.
Positive-worker runs still save optimizer/scheduler information but are marked
nonresumable because worker RNGs and prefetch queues are not captured. Partial
interrupted weights are nonresumable; restart from the last completed epoch and
replay the partial work. Software/hardware changes and nondeterministic GPU kernels
can change numerical results even when random states are restored.

## Verification

Regression tests compare an uninterrupted stochastic CPU run with interrupted and
resumed runs, for both dense and masked models. Final weights, optimizer,
scheduler, and RNG states must match exactly. Separate actual VGG subprocess
checks compare dense and pruned predictions using `atol=1e-6`, `rtol=1e-5`.
GPU smoke checks validate actual checkpoint loading and finite inference; they
do not establish full-dataset model quality or cross-device numerical equivalence.
