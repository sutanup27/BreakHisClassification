import copy
from multiprocessing.dummy import freeze_support
import pickle

import torch

from PruningNAS.DataProcess.DataPreprocessing import get_dataloaders
from PruningNAS.Models.DenseNet import DenseNet121
from PruningNAS.Models.MobileNetV1 import MobileNetV1
from PruningNAS.Models.ResNetBasic import *
from PruningNAS.Models.ResNetBottleNeck import *
from PruningNAS.PruneEvaluator import load_and_print_accuracies
from PruningNAS.Utills.PrunUtillCP import apply_channel_sorting_on, channel_prune_densenet, channel_prune_resnet
from PruningNAS.Utills.TrainingModulesUtills import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize the model
    basedir='PruningNAS'
    path='./dataset/cifar10'
    model=MobileNetV1(classes=10)
    # model_path=r'PruningNAS\checkpoint\Densenet-121\Densenet-121_cifar_95.769997.pth'
    # model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary

    model.to(device)

    train_dataloader,test_dataloader=get_dataloaders(path )
    dense_model_accuracy,_=evaluate(model,test_dataloader)
    print(f"Original model accuracy: {dense_model_accuracy:.4f}")

    sorted_model=copy.deepcopy(model)
    sorted_model = apply_channel_sorting_on(sorted_model)
    # pr=[0.00,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    sorted_model = channel_prune_densenet(sorted_model, pr)

    sorted_model.eval()  # Set to evaluation mode
    model.eval()  # Set to evaluation modeS

    dummy_input = torch.randn(20, 3, 32, 32).to(device)

    print("Original Model Architecture:")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = list(module.parameters())
            if params:
                print(f"{name}: {params[0].shape}")

    print("Sorted Model Architecture:")
    for name, module in sorted_model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = list(module.parameters())
            if params:
                print(f"{name}: {params[0].shape}")

    model.to(device)
    sorted_model.to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)    
    print("\nOutput shape:", dummy_output.shape)
    print("Output:", dummy_output)


    with torch.no_grad():
        dummy_output = sorted_model(dummy_input)    
    print("\nOutput shape:", dummy_output.shape)
    print("Output:", dummy_output)

    model_accuracy,_ =evaluate(model,test_dataloader)
    print("Original Model Accuracy:",model_accuracy)

    sorted_model_accuracy,_ =evaluate(sorted_model,test_dataloader)  
    print("Sorted Model Accuracy:",sorted_model_accuracy)



if __name__ == '__main__':
    freeze_support()
    main() 
