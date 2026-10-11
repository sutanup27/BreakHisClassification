from torch import nn
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def magnitude_based_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements*sparsity)
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    # Step 3: calculate the pruning threshold
    threshold,ind=  torch.kthvalue(torch.abs(tensor.view(-1)), num_zeros)
    # Step 4: calculate the pruning mask
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance.gt(threshold)
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


def fine_grained_prune(tensor: torch.Tensor, sparsity : float, prune_type="magnitude_based"):
    if prune_type=="magnitude_based" :
        return magnitude_based_prune(tensor,sparsity)
    else:
        print("Wrong prune type is passed")
        return tensor
        
def get_sparsity_dict_template_FGP(model,value=0.9):
    sparsity_dict = dict()
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
            sparsity_dict[name] = value   # Example: set all layers to 90% sparsity, you can customize this based on layer importance or other criteria
    return sparsity_dict

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_modules():
            if name in self.masks:
                param.weight *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param.weight, sparsity_dict[name],prune_type="magnitude_based")
        return masks
 

 