"""Compatibility imports for the validated canonical weight-mask pruner."""
import torch
from torch import nn
from ..pruning import enforce_masks, magnitude_mask, prune_weights, validate_sparsity


@torch.no_grad()
def magnitude_based_prune(tensor, sparsity):
    mask = magnitude_mask(tensor, sparsity)
    tensor.mul_(mask)
    return mask


def fine_grained_prune(tensor, sparsity, prune_type='magnitude_based'):
    if prune_type != 'magnitude_based':
        raise ValueError(f'Unsupported pruning type: {prune_type}')
    return magnitude_based_prune(tensor, sparsity)


def get_sparsity_dict_template_FGP(model, value=.9):
    validate_sparsity(value)
    return {name: value for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))}


class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = self.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        enforce_masks(model)

    @staticmethod
    def prune(model, sparsity_dict):
        prune_weights(model, settings=sparsity_dict)
        return {name: module.weight_mask.detach().clone() for name, module in model.named_modules()
                if hasattr(module, 'weight_mask')}
