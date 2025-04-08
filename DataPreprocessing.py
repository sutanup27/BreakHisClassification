import cv2
import os
from collections import defaultdict, OrderedDict
import numpy as np
import random
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
import time
from pygame import mixer
import copy
from typing import Union,List
 
path='../mrleyedataset'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slows training but ensures reproducibility

# Define separate transforms
image_size = 224

train_transform = Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
       # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ImageNet normalization
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
       
    ])

test_transform = Compose([
#    RandomCrop(image_size),
    transforms.Resize((460, 700)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ImageNet normalization

])
 

def get_datasets(Path,train_transform=train_transform,test_transform=test_transform,train_test_val_pecentage=[0.80, 0.20]):
    image_dataset= ImageFolder(root=Path)
    
    test_size = int(train_test_val_pecentage[1]* len(image_dataset))   # 80% for training
    train_size = len(image_dataset) - test_size                        # Remaining 20%
        
    train_indices, test_indices = random_split(image_dataset, [train_size, test_size])

    # Create separate datasets with different transforms
    train_dataset = ImageFolder(root=Path, transform=train_transform)  
    test_dataset = ImageFolder(root=Path, transform=test_transform)
    # Apply the split indices
    train_dataset= torch.utils.data.Subset(train_dataset, train_indices.indices)
    test_dataset= torch.utils.data.Subset(test_dataset, test_indices.indices)

    return train_dataset,test_dataset



def get_dataloaders(Path,train_transform=train_transform,test_transform=test_transform, train_test_val_pecentage=[0.80, 0.20], batch_size=32):
    set_seed(0)
    train_dataset,test_dataset=get_datasets(Path, train_transform, test_transform, train_test_val_pecentage)
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader,test_dataloader


