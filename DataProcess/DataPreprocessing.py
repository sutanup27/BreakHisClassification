import numpy as np
import os
import random
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
 

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    # Deterministic runs are slower; keep it optional
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

image_size = 32
train_transform=Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std),  # ImageNet normalization
    ])
test_transform= Compose([
        ToTensor(),
        Normalize(mean, std),  # ImageNet normalization
    ])
 
def get_datasets(path,train_transform=train_transform,test_transform=test_transform,train_test_val_pecentage=[0.80, 0.20]):
    transform={}
    transform["train"], transform["test"]=train_transform,test_transform
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=path,
            train=(split == "train"),
            download=True,
            transform=transform[split],)
    return dataset["train"],dataset["test"]



def get_dataloaders(
    path,
    train_transform=train_transform,
    test_transform=test_transform,
    train_test_val_pecentage=[0.80, 0.20],
    batch_size=512,
    num_workers=None,
    deterministic=False,
):
    set_seed(0, deterministic=deterministic)
    if num_workers is None:
        cpu_count = os.cpu_count() or 0
        num_workers = min(8, cpu_count) if cpu_count > 0 else 0
    dataset={}
    dataset["train"], dataset["test"]=get_datasets(path, train_transform, test_transform, train_test_val_pecentage)
    dataloader = {}
    for split in ['train', 'test']:
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": (split == 'train'),
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        dataloader[split] = DataLoader(dataset[split], **loader_kwargs)
    return dataloader['train'],dataloader['test']


