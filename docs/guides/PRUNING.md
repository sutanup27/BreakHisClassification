# Pruning and fine-tuning

The supported workflow offers unstructured weight masks for VGG/ResNet18 and physical output-channel removal for sequential VGG. Pruning decisions and fine-tuned checkpoint selection use validation. Test evaluation remains a separate final step.

## Supported methods

| Method | Models | Effect | Limits |
| --- | --- | --- | --- |
| `--method unstructured` (default) | VGG, ResNet18 | Masks convolution/linear weights | Dense tensor shapes remain; no guaranteed speedup |
| `--method channel` | VGG only | Removes convolution output channels and corresponding BN/input/classifier dimensions | Ratios below 1; at least one channel remains |

Residual/depthwise channel surgery, tied/shared weight storage, and mixing channel-pruned models with weight masks are rejected. Historical custom ResNet, MobileNet, DenseNet, and Inception channel helpers remain experimental, outside supported commands. Unsupported channel operations do not fall back to masking.

## Channel-pruning workflow

Starting from a supported dense VGG with the same manifest/source/magnification:

```powershell
python -m breakhis prune --dataset breakhis --checkpoint ../checkpoint/binary/40X/vgg/baseline/train.pth --method channel --sparsity 0.5 --experiment-id channel-50
python -m breakhis finetune --dataset breakhis --checkpoint ../checkpoint/binary/40X/vgg/channel-50/prune.pth --epochs 5 --learning-rate 0.0001 --experiment-id channel-50
python -m breakhis evaluate --dataset breakhis --checkpoint ../checkpoint/binary/40X/vgg/channel-50/finetune.pth --experiment-id channel-50
```

For nondefault data settings, supply the same manifest/source/magnification on every command. The narrower model is reconstructed from saved channel widths automatically.

Importance is the sum of absolute filter weights. Keep the highest-scoring outputs, with stable original-index ordering for ties. Copy corresponding batch-normalization affine values, running means/variances, and batch counters. Select the same indices in downstream convolution inputs and final classifier columns. RGB inputs and class labels remain unchanged. Channel pruning returns a new model without modifying the source.

The kept count is `max(1, original - round(original * ratio))`. Small layers may retain more than the nominal fraction. Zero-ratio channel pruning preserves predictions within numerical tolerance.

## Per-layer settings

Optional `--layer-sparsities layers.json` replaces the uniform `--sparsity` with a complete JSON layer-name-to-ratio object. Specify every eligible layer exactly. Missing/unknown names, invalid ratios, NaN/infinity, and tied weights fail before any layer changes.

VGG channel settings require `backbone.conv0` through `backbone.conv7`. VGG unstructured settings also require `fc2`. ResNet18 unstructured settings require all named Conv2d/Linear modules, including downsample convolutions and `fc`. Generate a complete unstructured template from the actual checkpoint:

```python
import json
from pathlib import Path
from torch import nn
from breakhis.checkpoints import load_checkpoint

model, _ = load_checkpoint('../checkpoint/binary/40X/vgg/baseline/train.pth')
settings = {name: 0.5 for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))}
Path('layers.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
```

For channel settings, filter only `nn.Conv2d` and start from unmasked VGG. Edit values and run:

```powershell
python -m breakhis prune --dataset breakhis --checkpoint ../checkpoint/binary/40X/vgg/baseline/train.pth --method unstructured --layer-sparsities layers.json --experiment-id layer-settings
```

The resolved map is saved in configuration and the validation report. The uniform ratio is still validated. JSON may use UTF-8 with or without a byte-order mark.

## Mask invariants and edge cases

Unstructured pruning selects exactly `round(numel * ratio)` mask entries per layer. Stable sorting handles magnitude ties, one-element tensors, and zero-count rounding. Naturally zero weights can make numerical sparsity exceed mask sparsity. Empty tensors and shared weight storage are rejected.

Re-pruning targets the requested total mask fraction rather than pruning another fraction of remaining weights. Lowering existing sparsity is rejected; use dense weights to compare smaller ratios.

Training enforces masks before batches and after optimizer steps. Masked original weights and matching optimizer buffers, including SGD momentum, stay zero. Forward hooks retain differentiable mask multiplication. Saved masks reconstruct automatically; fine-tuning needs no external pruner object.

## Validation, saved files, and resume

Pruning writes `prune.pth`, `prune.config.json`, and `prune.validation.json`. The report contains validation accuracy/loss, method, ratio/map, source, magnification, and input checkpoint. Compare candidate ratios using validation files; then fine-tune with validation-based selection. Do not choose settings through repeated test evaluations. Automatic ratio search is not implemented.

Fine-tuning passes the actual epoch and steps its supplied/configured scheduler. Best weights and last-epoch optimizer/scheduler/RNG state follow [checkpoint rules](CHECKPOINTS.md), for both masked and channel-pruned models. Channel checkpoints save validated widths and retained indices. Parameter reduction does not establish measured latency gains; efficiency reports remain Stage 8 work.

## Compatibility and verification

Canonical `prune_weights` and `prune_vgg_channels` return the resulting model. Shared `train` returns sample-averaged loss; `evaluate` returns percent accuracy and loss. Compatibility fine-grained helpers delegate to validated masks. `TrainingPrunned` retains `(best_accuracy, best_model, accuracies)` while correctly passing callbacks/epochs, stepping the supplied scheduler, and retaining the first valid epoch even at zero accuracy. Its second loader must be validation despite the historical `test_dataloader` name. `Training` retains its four-item tuple.

Tests cover ties/boundaries, masks and momentum, missing layer settings, tied-weight rejection, legacy callbacks/schedulers, VGG BN/classifier surgery, ResNet masked forward/backward, and fresh-process channel reload (`atol=1e-6`, `rtol=1e-5`). Real v1 GPU smoke runs confirm channel pruning, fine-tuning and fresh-process resume. These checks establish execution correctness, not full-dataset quality or guaranteed speed improvements.
