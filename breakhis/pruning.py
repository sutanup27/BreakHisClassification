"""Validated weight masks and sequential VGG output-channel pruning."""
import math
import torch
from torch import nn
from torch.nn.utils import prune

from .models.vgg import VGG


def validate_sparsity(value, channel=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Sparsity must be a finite number')
    if value < 0 or value > 1 or (channel and value == 1):
        raise ValueError('Channel sparsity must be in [0, 1)' if channel else 'Weight sparsity must be in [0, 1]')
    return value


def layer_settings(layers, sparsity, settings=None, channel=False):
    validate_sparsity(sparsity, channel)
    if settings is None:
        return {name: sparsity for name, _ in layers}
    if not isinstance(settings, dict) or set(settings) != {name for name, _ in layers}:
        raise ValueError('Layer sparsities must specify every prunable layer exactly; missing or unknown layer names')
    return {name: validate_sparsity(value, channel) for name, value in settings.items()}


def validate_independent_weights(layers):
    storages = set()
    for name, module in layers:
        weight = getattr(module, 'weight_orig', module.weight)
        if weight.numel() == 0:
            raise ValueError(f'Cannot prune empty weights in layer {name}')
        storage = (str(weight.device), weight.untyped_storage().data_ptr())
        if storage in storages:
            raise ValueError(f'Tied/shared weight storage is unsupported: {name}')
        storages.add(storage)
        if not torch.isfinite(weight).all():
            raise ValueError(f'Nonfinite weights in layer {name}')


def validate_channel_config(configuration, channels):
    if not isinstance(configuration, dict) or configuration.get('method') != 'l1-output-channel' or configuration.get('channels') != channels:
        raise ValueError('Invalid channel pruning configuration')
    layers = configuration.get('layers', {})
    if not isinstance(layers, dict) or set(layers) != {f'backbone.conv{index}' for index in range(8)}:
        raise ValueError('Channel configuration must specify all eight VGG convolutions')
    for index, width in enumerate(channels):
        settings = layers[f'backbone.conv{index}']
        if not isinstance(settings, dict):
            raise ValueError('Invalid per-layer channel configuration')
        original, retained = settings.get('original_channels'), settings.get('retained_indices')
        if isinstance(original, bool) or not isinstance(original, int) or original < width or not isinstance(retained, list):
            raise ValueError('Invalid original or retained channel count')
        if len(retained) != width or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < original for value in retained) or retained != sorted(set(retained)):
            raise ValueError('Invalid retained channel indices')
        ratio = validate_sparsity(settings.get('requested_sparsity'), channel=True)
        if max(1, original - round(original * ratio)) != width:
            raise ValueError('Channel count differs from the recorded pruning ratio')


@torch.no_grad()
def magnitude_mask(tensor, sparsity):
    """Prune exactly round(numel*sparsity); stable ordering resolves tied values."""
    validate_sparsity(sparsity)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError('Cannot prune an empty or nonfinite tensor')
    mask = torch.ones(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    count = round(tensor.numel() * sparsity)
    if count:
        order = torch.argsort(tensor.detach().abs().reshape(-1), stable=True)
        mask.reshape(-1)[order[:count]] = 0
    return mask


@torch.no_grad()
def enforce_masks(model, optimizer=None):
    """Keep original masked weights and matching optimizer buffers zero."""
    for module in model.modules():
        if not hasattr(module, 'weight_mask'):
            continue
        mask, original = module.weight_mask, module.weight_orig
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Pruning masks must contain only zero or one')
        original.mul_(mask)
        if optimizer is not None:
            for value in optimizer.state.get(original, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == original.shape:
                    value.mul_(mask)
        # Refresh the cached effective tensor; forward hooks recompute it with autograd.
        module.weight = original * mask


@torch.no_grad()
def prune_weights(model, sparsity=.5, settings=None):
    if hasattr(model, 'channel_pruning'):
        raise ValueError('Combining channel pruning with weight masks is not supported')
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, (nn.Conv2d, nn.Linear))]
    if not layers:
        raise ValueError('No convolution or linear weights to prune')
    ratios = layer_settings(layers, sparsity, settings)
    validate_independent_weights(layers)
    masks = {}
    # Validate every layer before modifying any layer.
    for name, module in layers:
        existing = getattr(module, 'weight_mask', torch.ones_like(module.weight))
        total = existing.numel()
        desired = round(total * ratios[name])
        removed = int((existing == 0).sum())
        if desired < removed:
            raise ValueError(f'Cannot reduce existing sparsity in {name}; use a dense checkpoint')
        retained = (existing.reshape(-1) != 0).nonzero().flatten()
        effective = getattr(module, 'weight_orig', module.weight).reshape(-1)
        order = torch.argsort(effective[retained].abs(), stable=True)
        mask = existing.contiguous().clone()
        mask.reshape(-1)[retained[order[:desired - removed]]] = 0
        masks[name] = mask
    for name, module in layers:
        if hasattr(module, 'weight_mask'):
            module.weight_mask.copy_(masks[name])
        else:
            prune.custom_from_mask(module, 'weight', masks[name])
    enforce_masks(model)
    return model


@torch.no_grad()
def prune_vgg_channels(model, sparsity=.5, settings=None):
    if type(model) is not VGG:
        raise ValueError('Channel pruning supports sequential VGG only; residual/depthwise architectures are unsupported')
    if any(hasattr(module, 'weight_mask') for module in model.modules()):
        raise ValueError('Channel pruning requires an unmasked VGG checkpoint')
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    ratios = layer_settings(layers, sparsity, settings, channel=True)
    validate_independent_weights(layers)
    selections = []
    for name, module in layers:
        if module.groups != 1:
            raise ValueError('Grouped/depthwise convolutions are unsupported for channel pruning')
        keep = max(1, module.out_channels - round(module.out_channels * ratios[name]))
        importance = module.weight.abs().flatten(1).sum(1)
        indices = torch.argsort(importance, descending=True, stable=True)[:keep].sort().values
        selections.append(indices)
    channels = [len(indices) for indices in selections]
    reference = next(model.parameters())
    rebuilt = VGG(classes=model.fc2.out_features, channels=channels).to(device=reference.device, dtype=reference.dtype)
    previous = torch.arange(3, device=reference.device)
    configuration = {}
    for index, ((name, old), retained) in enumerate(zip(layers, selections)):
        new = rebuilt.get_submodule(name)
        new.weight.copy_(old.weight[retained][:, previous])
        old_bn = model.backbone.get_submodule(f'bn{index}')
        new_bn = rebuilt.backbone.get_submodule(f'bn{index}')
        new_bn.weight.copy_(old_bn.weight[retained])
        new_bn.bias.copy_(old_bn.bias[retained])
        new_bn.running_mean.copy_(old_bn.running_mean[retained])
        new_bn.running_var.copy_(old_bn.running_var[retained])
        new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
        new_bn.eps, new_bn.momentum = old_bn.eps, old_bn.momentum
        configuration[name] = dict(original_channels=old.out_channels, retained_indices=retained.cpu().tolist(),
                                   requested_sparsity=ratios[name])
        previous = retained
    rebuilt.fc2.weight.copy_(model.fc2.weight[:, previous])
    rebuilt.fc2.bias.copy_(model.fc2.bias)
    rebuilt.train(model.training)
    rebuilt.channel_pruning = dict(method='l1-output-channel', channels=channels, layers=configuration)
    return rebuilt
