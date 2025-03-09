import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

from torch.utils.data import Dataset

import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader,random_split,Subset,TensorDataset
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
import torchvision.models as models
import torchvision
from torchprofile import profile_macs
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from DataPreprocessing import train_transform,test_transform,get_dataloaders
from TrainingModules import evaluate
from VGG import VGG
from Train import Training
from Viewer import plot_accuracy, plot_loss
print( torch.cuda.is_available())

seed=0
random.seed(seed)
np.random.seed(seed)

magf='200X'

path='H:\sutanu\BreaKHis\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']

root_dir=os.path.join(path, magf)
dataloader={}
dataloader['train'],dataloader['test']=get_dataloaders(root_dir,train_transform, test_transform)


model = models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # num_classes is the number of output classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = SGD(
  model.parameters(),
  lr=0.001,
  momentum=0.9,
  weight_decay=5e-4,
)

lambda_lr = lambda epoch: math.sqrt(.1) ** (epoch // 6)
#lambda_lr = lambda epoch: 0.1 ** (epoch // 5)
scheduler=LambdaLR(optimizer,lambda_lr)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_model, losses, test_losses, accs=Training( model, dataloader['train'], dataloader['test'], criterion, optimizer, num_epochs=30,scheduler=scheduler)

model=copy.deepcopy(best_model)
metric,_ = evaluate(model, dataloader["test"])
print(f"Best model accuray:", metric)

plot_accuracy(accs)
plot_loss(losses,test_losses)

torch.save(model.state_dict(), f'./checkpoint/{magf}/resnet18_{metric}.pth')
