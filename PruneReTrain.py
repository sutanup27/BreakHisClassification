import copy
import os
import random
from multiprocessing import freeze_support


import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from .Models.ResNetBasic import *
from .Models.ResNetBottleNeck import *
from .Models.DenseNet import *
from .Utills.TrainingModulesUtills import TrainingPrunned, evaluate
from .DataProcess.DataPreprocessing import get_dataloaders
from .Models.VGG import *
from .Utills.TrainingModulesUtills import Training
from .Utills.ViewerUtills import plot_accuracy, plot_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seed=0
    random.seed(seed)
    basedir='PruningNAS'

    path='./dataset/cifar10'
    classes=10
    train_dataloader, test_dataloader = get_dataloaders(path, batch_size=128)


    # ########load from path only for retraining #####
    select_model='Resnet-34'
    model_path=r'PruningNAS\checkpoint\Resnet-34\FGP\Resnet-34_cifar_FGP_94.04000091552734.pth'
    model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
    model_dir = os.path.dirname(model_path)

    ################################################
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD( model.parameters(), lr=0.001,  momentum=0.9,  weight_decay=5e-4,)

    num_epochs =300
    scheduler = CosineAnnealingLR(optimizer, num_epochs)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    pruned_model_accuracy,best_pruned_model,accuracies,train_losses,test_losses=TrainingPrunned(model,train_dataloader,test_dataloader,criterion, optimizer, pruner,scheduler=None,num_finetune_epochs=num_epochs,isCallback=isCallback)

    metric,_ = evaluate(best_pruned_model, test_dataloader)
    print(f"Best model accuray:", metric)

    torch.save(best_pruned_model, f'{model_dir}/{select_model}_cifar_{metric:0.6f}.pth')
    torch.save(best_pruned_model.state_dict(), f'{model_dir}/{select_model}_cifar_{metric:0.6f}_state_dict.pth')


if __name__ == '__main__':
    freeze_support()
    main()

    
