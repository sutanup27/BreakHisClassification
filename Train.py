import copy
import math
import random


import os
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
import torchvision.models as models
from DataPreprocessing import train_transform,test_transform,get_dataloaders
from TrainingModules import evaluate
from VGG import VGG
from TrainingModules import Training
from Viewer import plot_accuracy, plot_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

seed=0
random.seed(seed)

magf='40X'

path='..\Dataset\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']

root_dir=os.path.join(path, magf)
dataloader={}
train_dataloader, test_dataloader=get_dataloaders(root_dir,train_transform, test_transform)


select_model='vgg'
if select_model=='vgg':
    model=VGG()
elif select_model=='resnet':
    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # num_classes is the number of output classes
else:
    exit

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD( model.parameters(), lr=0.001,  momentum=0.9,  weight_decay=5e-4)

# lambda_lr = lambda epoch: math.sqrt(.1) ** (epoch // 7)
# lambda_lr = lambda epoch: 0.1 ** (epoch // 5)
# scheduler=LambdaLR(optimizer,lambda_lr)
num_epochs=20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_model, losses, test_losses, accs=Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs,scheduler=scheduler)

model=copy.deepcopy(best_model)
metric,_ = evaluate(model, test_dataloader)
print(f"Best model accuray:", metric)

plot_accuracy(accs)
plot_loss(losses,test_losses)

torch.save(model, f'./checkpoint/{magf}/{select_model}/{select_model}_{metric}.pth')



    
