import os

import torch
from PruningNAS.Models.DenseNet import *
from PruningNAS.Models.ResNetBasic import *
from PruningNAS.Models.ResNetBottleNeck import *
from PruningNAS.Utills.PrunUtillCP import ChannelPrunner, channel_prune_resnet

# Initialize DenseNet-121 model
model =DenseNet121(classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#       Load pre-trained weights (if available)
# model_path = r'PruningNAS\checkpoint\Resnet-34\Resnet-34_cifar_95.69000244140625.pth'
# model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary

model.to(device)
model.eval()

# P_model=channel_prune_resnet(model, 0.7)
# P_model.eval()
# P_model.to(device)


print("Pruned Model Architecture:")
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
        params = list(module.parameters())
        if params:
            print(f"{name}: {params[0].shape}")

count=1



# for blocks in model.blocks:
#     DenseBlock
#     TransitionLayer


print("Total prunable layers in DenseNet-121:", count)


model_path=r'PruningNAS\checkpoint\Resnet-34\FGP'
model_dir = os.path.dirname(model_path)



print("dir", model_dir)
# # Example: create dummy input data (batch size 1, 3 channels, 32x32 for CIFAR-10)
# dummy_input = torch.randn(1, 3, 32, 32)

# # Forward pass through the pruned model
# with torch.no_grad():
#     dummy_output = P_model(dummy_input.to(device))      
# print("\nOutput shape:", dummy_output.shape)
# print("Output:", dummy_output)

# # Calculate pruning ratio count for DenseNet: conv0 + num_denseblocks + num_transitions
# num_denseblocks = 4  # DenseNet121 has 4 dense blocks
# num_transitions = 3  # transitions between blocks (not after last block)
# prune_ratio_count = 1 + num_denseblocks + num_transitions

# # Create uniform pruning ratios (0.7 = 30% pruning)
# sparsity_dict = [0.7] * prune_ratio_count

# # Prune the model using ChannelPrunner
# pruned_model = ChannelPrunner(model, sparsity_dict, 'Densenet')
# pruned_model.eval()

# # Test the pruned model with dummy input
# print("Pruned Model Architecture:")
# for name, module in pruned_model.named_modules():
#     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
#         params = list(module.parameters())
#         if params:
#             print(f"{name}: {params[0].shape}")

# # Forward pass
# dummy_output = pruned_model(dummy_input)
# print("\nOutput shape:", dummy_output.shape)
# print("Output:", dummy_output)