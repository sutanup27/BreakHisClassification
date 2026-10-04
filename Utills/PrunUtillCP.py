import copy
from torch import nn
from typing import Union
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from PruningNAS.Models.Inceptionv1 import InceptionBlock
from PruningNAS.Models.MobileNetV1 import *
from PruningNAS.Models.DenseNet import DenseBlock, TransitionLayer
from PruningNAS.Models.MobileNetV2 import InvertedResidual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    ##################### YOUR CODE STARTS HERE #####################
    return round(channels*(1-prune_ratio))
    ##################### YOUR CODE ENDS HERE #####################


# function to sort the channels from important to non-important
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        ##################### YOUR CODE STARTS HERE #####################
        importance = torch.linalg.norm(channel_weight)
        ##################### YOUR CODE ENDS HERE #####################
        importances.append(importance.view(1))
    return torch.cat(importances)


def count_resnet_blocks(model):
    return len(model.layer1)+len(model.layer2)+len(model.layer3)+len(model.layer4)

def count_densenet_prunable_layers(model):
    return 1+sum(model.block_config)*2

def count_mobilenet_blocks(model):
    count=0
    if model.__class__.__name__.lower()=='mobilenetv1':
        for layer in model.model:
            if isinstance(layer, DepthwiseSeparableConv):
                count=count+1
        return count
    else:
        for layer in model.features:
            if isinstance(layer, InvertedResidual):
                count=count+1
        return count

def count_inception_prunable_layers(model):
    count = 0
    for m in model.stem:
        if isinstance(m, nn.Conv2d):
            count += 1
    for module in model.modules():
        if isinstance(module, InceptionBlock):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    count += 1
    return count


def count_prunable_layers(model,model_type='Vgg-16'):
    if model_type=='Vgg-16':
        return len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    

@torch.no_grad()
def apply_channel_sorting_on_vgg(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(
            prev_conv.weight.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(torch.index_select(
            next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################

    return model

@torch.no_grad()
def apply_channel_sorting_on(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [ layer for layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    all_bns = [ layer for layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))

        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(torch.index_select(
            next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################
    return model



@torch.no_grad()
def apply_channel_sorting_on_inception(model):
    """
    Sort channels in InceptionNet for structured pruning.
    Stem: sequential conv chain, sort by next conv input importance.
    InceptionBlock branches: sort internal reduce conv by next conv's importance.
    """
    model = copy.deepcopy(model)

    # ── Stem: 3 Conv2d layers (no BN), sort each by next conv's importance ──
    stem_convs = [m for m in model.stem if isinstance(m, nn.Conv2d)]
    for i in range(len(stem_convs) - 1):
        importance = get_input_channel_importance(stem_convs[i + 1].weight)
        sort_idx   = torch.argsort(importance, descending=True)
        # re-order output channels of conv[i]
        stem_convs[i].weight.copy_(
            torch.index_select(stem_convs[i].weight.detach(), 0, sort_idx))
        # re-order input channels of conv[i+1]
        stem_convs[i + 1].weight.copy_(
            torch.index_select(stem_convs[i + 1].weight.detach(), 1, sort_idx))

    # ── InceptionBlocks ──
    for module in model.modules():
        if not isinstance(module, InceptionBlock):
            continue

        # branch2: [Conv2d(reduce), ReLU, Conv2d(3x3)]  -> sort reduce output
        b2_convs = [m for m in module.branch2 if isinstance(m, nn.Conv2d)]
        if len(b2_convs) == 2:
            importance = get_input_channel_importance(b2_convs[1].weight)
            sort_idx   = torch.argsort(importance, descending=True)
            b2_convs[0].weight.copy_(
                torch.index_select(b2_convs[0].weight.detach(), 0, sort_idx))
            b2_convs[1].weight.copy_(
                torch.index_select(b2_convs[1].weight.detach(), 1, sort_idx))

        # branch3: [Conv2d(reduce), ReLU, Conv2d(5x5)]  -> sort reduce output
        b3_convs = [m for m in module.branch3 if isinstance(m, nn.Conv2d)]
        if len(b3_convs) == 2:
            importance = get_input_channel_importance(b3_convs[1].weight)
            sort_idx   = torch.argsort(importance, descending=True)
            b3_convs[0].weight.copy_(
                torch.index_select(b3_convs[0].weight.detach(), 0, sort_idx))
            b3_convs[1].weight.copy_(
                torch.index_select(b3_convs[1].weight.detach(), 1, sort_idx))

        # branch1 & branch4 are single Conv2d -- no internal sort needed

    return model


@torch.no_grad()
def channel_prune_vgg(model,
                  prune_ratio: Union[float, dict,list]) :
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, dict,list))
    n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...

    if isinstance(prune_ratio, dict):
        prune_ratio=list(prune_ratio.values())
        assert len(prune_ratio) == n_conv 
    elif isinstance(prune_ratio, float):  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv )
    else:
        assert len(prune_ratio) == n_conv 
    # we prune the convs in the backbone with a uniform ratio
    # we only apply pruning to the backbone features
    all_params = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    assert len(all_params) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_params[i_ratio]
        prev_bn = all_bns[i_ratio]
        if i_ratio!=n_conv-1:
            next_conv = all_params[i_ratio + 1]
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels

        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
        if i_ratio!=n_conv-1:
            next_conv.weight.set_(next_conv.weight.detach()[:,:n_keep])
    model.fc2.weight.set_(model.fc2.weight.detach()[:,:n_keep])
    return model

@torch.no_grad()
def channel_prune_resnet(model, prune_ratios: Union[float, dict,list]):    
    def prune_block(block, prune_ratios,prev_n_keep):
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change

        block.conv1.weight.set_(block.conv1.weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel chang

        original_channels=block.conv1.out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratios)
        block.conv1.weight.set_(block.conv1.weight.detach()[:n_keep])
        block.bn1.weight.set_(block.bn1.weight.detach()[:n_keep])
        block.bn1.bias.set_(block.bn1.bias.detach()[:n_keep])
        block.bn1.running_mean.set_(block.bn1.running_mean.detach()[:n_keep])
        block.bn1.running_var.set_(block.bn1.running_var.detach()[:n_keep])

        block.conv2.weight.set_(block.conv2.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change

        block.conv2.weight.set_(block.conv2.weight.detach()[:n_keep])
        block.bn2.weight.set_(block.bn2.weight.detach()[:n_keep])
        block.bn2.bias.set_(block.bn2.bias.detach()[:n_keep])
        block.bn2.running_mean.set_(block.bn2.running_mean.detach()[:n_keep])
        block.bn2.running_var.set_(block.bn2.running_var.detach()[:n_keep])
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:n_keep])
            block.shortcut[1].weight.set_(block.shortcut[1].weight.detach()[:n_keep])
            block.shortcut[1].bias.set_(block.shortcut[1].bias.detach()[:n_keep])
            block.shortcut[1].running_mean.set_(block.shortcut[1].running_mean.detach()[:n_keep])
            block.shortcut[1].running_var.set_(block.shortcut[1].running_var.detach()[:n_keep])
        return n_keep #we will need n_keep to fix next conv' inchannels fixing

    def prune_bn_block(block, prune_ratios, prev_n_keep):
        
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change

        block.conv1.weight.set_(block.conv1.weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel chang

        original_channels=block.conv1.out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratios)
        block.conv1.weight.set_(block.conv1.weight.detach()[:n_keep])
        block.bn1.weight.set_(block.bn1.weight.detach()[:n_keep])
        block.bn1.bias.set_(block.bn1.bias.detach()[:n_keep])
        block.bn1.running_mean.set_(block.bn1.running_mean.detach()[:n_keep])
        block.bn1.running_var.set_(block.bn1.running_var.detach()[:n_keep])

        block.conv2.weight.set_(block.conv2.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change

        block.conv2.weight.set_(block.conv2.weight.detach()[:n_keep])
        block.bn2.weight.set_(block.bn2.weight.detach()[:n_keep])
        block.bn2.bias.set_(block.bn2.bias.detach()[:n_keep])
        block.bn2.running_mean.set_(block.bn2.running_mean.detach()[:n_keep])
        block.bn2.running_var.set_(block.bn2.running_var.detach()[:n_keep])

        output_channels=n_keep*block.expansion
        block.conv3.weight.set_(block.conv3.weight.detach()[:output_channels,:n_keep]) #fixing number of inchannels due to previous channel change
        
        block.bn3.weight.set_(block.bn3.weight.detach()[:output_channels])
        block.bn3.bias.set_(block.bn3.bias.detach()[:output_channels])
        block.bn3.running_mean.set_(block.bn3.running_mean.detach()[:output_channels])
        block.bn3.running_var.set_(block.bn3.running_var.detach()[:output_channels])
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:output_channels]) #fixing number of inchannels due to previous channel change
            block.shortcut[1].weight.set_(block.shortcut[1].weight.detach()[:output_channels])
            block.shortcut[1].bias.set_(block.shortcut[1].bias.detach()[:output_channels])
            block.shortcut[1].running_mean.set_(block.shortcut[1].running_mean.detach()[:output_channels])
            block.shortcut[1].running_var.set_(block.shortcut[1].running_var.detach()[:output_channels])
        return output_channels #we will need n_keep to fix next conv' inchannels fixing

    assert isinstance(prune_ratios, (float, dict,list))
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratios, dict):
        prune_ratios=list(prune_ratios.values())
        assert len(prune_ratios) == count_resnet_blocks(model)+1
    elif isinstance(prune_ratios, float):  # convert float to list
        prune_ratios = [prune_ratios] * (count_resnet_blocks(model)+1)
    else:
        pass
    original_channels=model.conv1.out_channels
    n_keep = get_num_channels_to_keep(original_channels, prune_ratios[0])
    model.conv1.weight.set_(model.conv1.weight.detach()[:n_keep])
    model.bn1.weight.set_(model.bn1.weight.detach()[:n_keep])
    model.bn1.bias.set_(model.bn1.bias.detach()[:n_keep])
    model.bn1.running_mean.set_(model.bn1.running_mean.detach()[:n_keep])
    model.bn1.running_var.set_(model.bn1.running_var.detach()[:n_keep])
    i=1
    for name, layer in model.named_children():
        if isinstance(layer,nn.Sequential):
            for b in layer:
                p_ratios=prune_ratios[i]
                if hasattr(b, 'conv3'):
                    n_keep=prune_bn_block(b,p_ratios,n_keep)
                else:
                    n_keep=prune_block(b,p_ratios,n_keep)
                i=i+1
    
    model.fc.weight.set_(model.fc.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
    return model

@torch.no_grad()
def channel_prune_densenet(model, prune_ratios: Union[float, dict, list]):
    """Prune DenseNet model by channel pruning with specified ratios."""
    
    def prune_dense_layer(layer, prune_ratio1, prune_ratio2, prev_n_keep):
        """Prune a single DenseLayer in the DenseBlock."""
        # Prune norm1 and conv1 based on input channels
        layer.norm1.weight.set_(layer.norm1.weight.detach()[:prev_n_keep])
        layer.norm1.bias.set_(layer.norm1.bias.detach()[:prev_n_keep])
        layer.norm1.running_mean.set_(layer.norm1.running_mean.detach()[:prev_n_keep])
        layer.norm1.running_var.set_(layer.norm1.running_var.detach()[:prev_n_keep])
        layer.conv1.weight.set_(layer.conv1.weight.detach()[:, :prev_n_keep])

        # Calculate kept channels after conv1
        original_channels = layer.conv1.out_channels
        n_keep_conv1 = get_num_channels_to_keep(original_channels, prune_ratio1)
        layer.conv1.weight.set_(layer.conv1.weight.detach()[:n_keep_conv1])

        # Prune norm2 and conv2 based on conv1 output
        layer.norm2.weight.set_(layer.norm2.weight.detach()[:n_keep_conv1])
        layer.norm2.bias.set_(layer.norm2.bias.detach()[:n_keep_conv1])
        layer.norm2.running_mean.set_(layer.norm2.running_mean.detach()[:n_keep_conv1])
        layer.norm2.running_var.set_(layer.norm2.running_var.detach()[:n_keep_conv1])
        layer.conv2.weight.set_(layer.conv2.weight.detach()[:, :n_keep_conv1])

        # Calculate kept channels after conv2 (output channels)
        original_channels = layer.conv2.out_channels
        n_keep_conv2 = get_num_channels_to_keep(original_channels, prune_ratio2)
        layer.conv2.weight.set_(layer.conv2.weight.detach()[:n_keep_conv2])
        
        return prev_n_keep+n_keep_conv2

    def prune_transition_layer(transition, prev_n_keep):
        """Prune a transition layer."""
        transition.norm.weight.set_(transition.norm.weight.detach()[:prev_n_keep])
        transition.norm.bias.set_(transition.norm.bias.detach()[:prev_n_keep])
        transition.norm.running_mean.set_(transition.norm.running_mean.detach()[:prev_n_keep])
        transition.norm.running_var.set_(transition.norm.running_var.detach()[:prev_n_keep])
        transition.conv.weight.set_(transition.conv.weight.detach()[:, :prev_n_keep])

        # Calculate kept channels after transition
        n_keep = prev_n_keep//2
        transition.conv.weight.set_(transition.conv.weight.detach()[:n_keep])

        return n_keep

    # Validate and normalize prune_ratios
    assert isinstance(prune_ratios, (float, dict, list))
    
    if isinstance(prune_ratios, dict):
        prune_ratios = list(prune_ratios.values())
    elif isinstance(prune_ratios, float):
        prune_ratios = [prune_ratios] * count_densenet_prunable_layers(model)
    
    assert len(prune_ratios) == count_densenet_prunable_layers(model), \
        f"Expected {count_densenet_prunable_layers(model)} prune ratios, got {len(prune_ratios)}"

    # Prune initial convolution
    original_channels = model.init_conv[0].out_channels
    n_keep = get_num_channels_to_keep(original_channels, prune_ratios[0])
    model.init_conv[0].weight.set_(model.init_conv[0].weight.detach()[:n_keep])
    model.init_conv[1].weight.set_(model.init_conv[1].weight.detach()[:n_keep])
    model.init_conv[1].bias.set_(model.init_conv[1].bias.detach()[:n_keep])
    model.init_conv[1].running_mean.set_(model.init_conv[1].running_mean.detach()[:n_keep])
    model.init_conv[1].running_var.set_(model.init_conv[1].running_var.detach()[:n_keep])

    # Prune DenseBlocks and TransitionLayers
    ratio_idx = 1  # Start from index 1 (init_conv uses index 0)
    
    for block in model.blocks:
        if isinstance(block, DenseBlock):
            for layer in block.block:
                if ratio_idx + 1 < len(prune_ratios):
                    n_keep = prune_dense_layer(layer, prune_ratios[ratio_idx], 
                                               prune_ratios[ratio_idx + 1], n_keep)
                    ratio_idx += 2
        elif isinstance(block, TransitionLayer):
            if ratio_idx < len(prune_ratios):
                n_keep = prune_transition_layer(block, n_keep)

    # Prune final fully connected layer
    model.final_bn.weight.set_(model.final_bn.weight.detach()[:n_keep])
    model.final_bn.bias.set_(model.final_bn.bias.detach()[:n_keep])
    model.final_bn.running_mean.set_(model.final_bn.running_mean.detach()[:n_keep])
    model.final_bn.running_var.set_(model.final_bn.running_var.detach()[:n_keep])
    model.fc.weight.set_(model.fc.weight.detach()[:, :n_keep])

    return model

def channel_prune_mobilenetv1(model, prune_ratios: Union[float, dict, list]):
    def prune_dws_block(block, prune_ratios, prev_n_keep):
        block.depthwise.weight.set_(block.depthwise.weight.detach()[:prev_n_keep]) #fixing number of inchannels due to previous channel change

        block.depthwise.groups=prev_n_keep  # to handle depthwwise conv pruning
        block.depthwise.weight.set_(block.depthwise.weight.detach()[:prev_n_keep])
        block.bn1.weight.set_(block.bn1.weight.detach()[:prev_n_keep])
        block.bn1.bias.set_(block.bn1.bias.detach()[:prev_n_keep])
        block.bn1.running_mean.set_(block.bn1.running_mean.detach()[:prev_n_keep])
        block.bn1.running_var.set_(block.bn1.running_var.detach()[:prev_n_keep])

        block.pointwise.weight.set_(block.pointwise.weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change
        original_channels = block.pointwise.out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratios)
        block.pointwise.weight.set_(block.pointwise.weight.detach()[:n_keep])
        block.bn2.weight.set_(block.bn2.weight.detach()[:n_keep])
        block.bn2.bias.set_(block.bn2.bias.detach()[:n_keep])
        block.bn2.running_mean.set_(block.bn2.running_mean.detach()[:n_keep])
        block.bn2.running_var.set_(block.bn2.running_var.detach()[:n_keep])
        
        return n_keep #we will need n_keep to fix next conv' inchannels fixing
    
    assert isinstance(prune_ratios, (float, dict,list))
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratios, dict):
        prune_ratios=list(prune_ratios.values())
        assert len(prune_ratios) == count_mobilenet_blocks(model)+1
    elif isinstance(prune_ratios, float):  # convert float to list
        prune_ratios = [prune_ratios] * (count_mobilenet_blocks(model)+1)
    else:
        pass
    original_channels=model.model[0].out_channels
    n_keep = get_num_channels_to_keep(original_channels, prune_ratios[0])
    model.model[0].weight.set_(model.model[0].weight.detach()[:n_keep])
    model.model[1].weight.set_(model.model[1].weight.detach()[:n_keep])
    model.model[1].bias.set_(model.model[1].bias.detach()[:n_keep])
    model.model[1].running_mean.set_(model.model[1].running_mean.detach()[:n_keep])
    model.model[1].running_var.set_(model.model[1].running_var.detach()[:n_keep])
    i=1
    for name, module in model.model.named_children():
        if isinstance(module, DepthwiseSeparableConv):
            n_keep=prune_dws_block(module, prune_ratios[i], n_keep)
            i=i+1
    
    model.fc.weight.set_(model.fc.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
    return model


def channel_prune_mobilenetv2(model, prune_ratios: Union[float, dict, list]):
    def prune_ir_block(block, prune_ratio1,prune_ratio2, prev_n_keep):
        i=0
        if len(block)!=5:
            block[i].weight.set_(block[i].weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change
            original_channels = block[i].out_channels
            n_keep = get_num_channels_to_keep(original_channels, prune_ratio1)
            block[i].weight.set_(block[i].weight.detach()[:n_keep])

            i=i+1
            block[i].weight.set_(block[i].weight.detach()[:n_keep]) #fixing number of inchannels due to previous channel change
            block[i].bias.set_(block[i].bias.detach()[:n_keep])
            block[i].running_mean.set_(block[i].running_mean.detach()[:n_keep])
            block[i].running_var.set_(block[i].running_var.detach()[:n_keep])
            i=i+2
        ###################################################
        if prune_ratio1 is not None:
            original_channels = block[i].out_channels
            n_keep = get_num_channels_to_keep(original_channels, prune_ratio1)
        else:
            n_keep=prev_n_keep
        block[i].weight.set_(block[i].weight.detach()[:n_keep]) #fixing number of inchannels due to previous channel change
        block[i].groups=n_keep  # to handle depthwwise conv pruning

        i= i+1
        block[i].weight.set_(block[i].weight.detach()[:n_keep])
        block[i].bias.set_(block[i].bias.detach()[:n_keep])
        block[i].running_mean.set_(block[i].running_mean.detach()[:n_keep])
        block[i].running_var.set_(block[i].running_var.detach()[:n_keep])
        ###################################################
        i=i+2
        block[i].weight.set_(block[i].weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
        original_channels = block[i].out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratio2)
        block[i].weight.set_(block[i].weight.detach()[:n_keep])
        i=i+1
        block[i].weight.set_(block[i].weight.detach()[:n_keep])
        block[i].bias.set_(block[i].bias.detach()[:n_keep])
        block[i].running_mean.set_(block[i].running_mean.detach()[:n_keep])
        block[i].running_var.set_(block[i].running_var.detach()[:n_keep])
        
        return n_keep #we will need n_keep to fix next conv' inchannels fixing
    
    assert isinstance(prune_ratios, (float, dict,list))
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratios, dict):
        prune_ratios=list(prune_ratios.values())
        assert len(prune_ratios) == count_mobilenet_blocks(model)*2+1
    elif isinstance(prune_ratios, float):  # convert float to list
        prune_ratios = [prune_ratios] * (count_mobilenet_blocks(model)*2+1)
    else:
        pass
    original_channels=model.features[0][0].out_channels
    n_keep = get_num_channels_to_keep(original_channels, prune_ratios[0])
    model.features[0][0].weight.set_(model.features[0][0].weight.detach()[:n_keep])
    model.features[0][1].weight.set_(model.features[0][1].weight.detach()[:n_keep])
    model.features[0][1].bias.set_(model.features[0][1].bias.detach()[:n_keep])
    model.features[0][1].running_mean.set_(model.features[0][1].running_mean.detach()[:n_keep])
    model.features[0][1].running_var.set_(model.features[0][1].running_var.detach()[:n_keep])
    i=1
    for name, module in model.named_modules():
        if isinstance(module, InvertedResidual):
            if  len(module.block)==5: #if it is the first block, we only prune the output of the first conv and input of the second conv, but not the depthwise conv
                n_keep=prune_ir_block(module.block,None, prune_ratios[i], n_keep)
                i=i+1
            else:
                n_keep=prune_ir_block(module.block, prune_ratios[i],prune_ratios[i+1], n_keep)
                i=i+2
    
    model.features[18].weight.set_(model.features[18].weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
    n_keep = get_num_channels_to_keep(model.features[18].out_channels, prune_ratios[i])
    model.features[18].weight.set_(model.features[18].weight.detach()[:n_keep]) #fixing number of inchannels due to previous channel change
    
    model.features[19].weight.set_(model.features[19].weight.detach()[:n_keep])
    model.features[19].bias.set_(model.features[19].bias.detach()[:n_keep])
    model.features[19].running_mean.set_(model.features[19].running_mean.detach()[:n_keep])
    model.features[19].running_var.set_(model.features[19].running_var.detach()[:n_keep])

    model.fc.weight.set_(model.fc.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
    return model



@torch.no_grad()
def channel_prune_inception(model, prune_ratios: Union[float, dict, list]):

    # ────────────────────────────────────────────────
    #  Helper: prune a single Conv2d (no BN in this model)
    # ────────────────────────────────────────────────
    def prune_conv(conv, n_keep_in, n_keep_out):
        """Prune conv input channels then output channels, update bias if exists."""
        # 1) Fix input channels from previous layer's pruning
        conv.weight.set_(conv.weight.detach()[:, :n_keep_in])
        # 2) Prune output channels
        conv.weight.set_(conv.weight.detach()[:n_keep_out])
        # 3) Prune bias if it exists
        if conv.bias is not None:
            conv.bias.set_(conv.bias.detach()[:n_keep_out])

    # ── collect ordered inception blocks ──
    inception_block_names = [
        'inception3a', 'inception3b',
        'inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e',
        'inception5a', 'inception5b'
    ]
    inception_blocks = [getattr(model, n) for n in inception_block_names
                        if hasattr(model, n)]

    n_stem_convs    = len([m for m in model.stem if isinstance(m, nn.Conv2d)])
    n_ratios_needed = n_stem_convs + 6 * len(inception_blocks)

    # ── normalise prune_ratios ──
    assert isinstance(prune_ratios, (float, dict, list))
    if isinstance(prune_ratios, float):
        prune_ratios = [prune_ratios] * n_ratios_needed
    elif isinstance(prune_ratios, dict):
        prune_ratios = list(prune_ratios.values())

    assert len(prune_ratios) == n_ratios_needed, \
        f"Expected {n_ratios_needed} ratios, got {len(prune_ratios)}"

    ratio_idx = 0

    # ────────────────────────────────────────────────
    #  STEM  (3 Conv2d, no BN, no bias by default)
    #  stem structure:
    #    Conv2d(3,  64, 7x7)  -> index 0
    #    Conv2d(64, 64, 1x1)  -> index 1
    #    Conv2d(64, 192, 3x3) -> index 2
    # ────────────────────────────────────────────────
    stem_convs = [m for m in model.stem if isinstance(m, nn.Conv2d)]
    n_keep = 3  # RGB — never prune input channels of first conv
    for conv in stem_convs:
        n_keep_out = get_num_channels_to_keep(conv.out_channels, prune_ratios[ratio_idx])
        prune_conv(conv, n_keep, n_keep_out)
        n_keep = n_keep_out
        ratio_idx += 1

    # ────────────────────────────────────────────────
    #  InceptionBlocks
    #  branch1 : Conv2d(in, c1, 1x1)                  -- bare Conv2d (may have bias)
    #  branch2 : Sequential[Conv2d(in,c3r,1), ReLU, Conv2d(c3r,c3,3)]
    #  branch3 : Sequential[Conv2d(in,c5r,1), ReLU, Conv2d(c5r,c5,5)]
    #  branch4 : Sequential[MaxPool2d, Conv2d(in,pool_proj,1)]
    # ────────────────────────────────────────────────
    for block in inception_blocks:
        prev_n_keep = n_keep  # all 4 branches share same input channel count

        r_b1, r_b2r, r_b2, r_b3r, r_b3, r_b4 = prune_ratios[ratio_idx: ratio_idx + 6]
        ratio_idx += 6

        # ── branch1: bare Conv2d ──
        b1_conv   = block.branch1
        n_keep_b1 = get_num_channels_to_keep(b1_conv.out_channels, r_b1)
        prune_conv(b1_conv, prev_n_keep, n_keep_b1)

        # ── branch2: [Conv2d reduce, ReLU, Conv2d 3x3] ──
        b2_convs  = [m for m in block.branch2 if isinstance(m, nn.Conv2d)]
        n_keep_b2r = get_num_channels_to_keep(b2_convs[0].out_channels, r_b2r)
        n_keep_b2  = get_num_channels_to_keep(b2_convs[1].out_channels, r_b2)
        prune_conv(b2_convs[0], prev_n_keep, n_keep_b2r)
        prune_conv(b2_convs[1], n_keep_b2r,  n_keep_b2)

        # ── branch3: [Conv2d reduce, ReLU, Conv2d 5x5] ──
        b3_convs  = [m for m in block.branch3 if isinstance(m, nn.Conv2d)]
        n_keep_b3r = get_num_channels_to_keep(b3_convs[0].out_channels, r_b3r)
        n_keep_b3  = get_num_channels_to_keep(b3_convs[1].out_channels, r_b3)
        prune_conv(b3_convs[0], prev_n_keep, n_keep_b3r)
        prune_conv(b3_convs[1], n_keep_b3r,  n_keep_b3)

        # ── branch4: [MaxPool2d, Conv2d proj] ──
        b4_convs  = [m for m in block.branch4 if isinstance(m, nn.Conv2d)]
        n_keep_b4 = get_num_channels_to_keep(b4_convs[0].out_channels, r_b4)
        prune_conv(b4_convs[0], prev_n_keep, n_keep_b4)

        # concatenated output -> input to next block
        n_keep = n_keep_b1 + n_keep_b2 + n_keep_b3 + n_keep_b4

    # ────────────────────────────────────────────────
    #  Fix FC layer input features
    # ────────────────────────────────────────────────
    model.fc.weight.set_(model.fc.weight.detach()[:, :n_keep])

    return model



def apply_channel_sorting(model):
    model_name = model.__class__.__name__.lower()
    if model_name[:3]=='vgg':
        return apply_channel_sorting_on_vgg(model)
    elif model_name[:9]=='inception':                         
        return apply_channel_sorting_on_inception(model)
    elif model_name[:6]=='resnet' or model_name[:8]=='densenet' or model_name[:9]=='mobilenet':
        return apply_channel_sorting_on(model)
    else:
        print(f'model_type doesn\'t exists 1:{model_name}')
        exit(0)


def channel_prune(model, prune_ratio: Union[dict, float],model_type):
    if model_type[:3].lower()=='vgg':
        return channel_prune_vgg(model, prune_ratio)
    elif model_type[:6].lower()=='resnet':
        return channel_prune_resnet(model, prune_ratio)
    elif model_type[:8].lower()=='densenet':
        return channel_prune_densenet(model, prune_ratio)        
    elif model_type[:11].lower()=='mobilenetv1':
        return channel_prune_mobilenetv1(model, prune_ratio)
    elif model_type[:11].lower()=='mobilenetv2':
        return channel_prune_mobilenetv2(model, prune_ratio)
    elif model_type[:9].lower()=='inception':                
        return channel_prune_inception(model, prune_ratio)
    else:
        print(f'model_type doesn\'t exists 2:{model_type}')
        exit(0)

def ChannelPrunner(model,channel_pruning_ratio,model_type):
    sorted_model = apply_channel_sorting(model)
    pruned_model = channel_prune(sorted_model, channel_pruning_ratio,model_type)
    return pruned_model