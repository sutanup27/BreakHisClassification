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
from Utill import download_url,magnitude_based_prune

class VGG(nn.Module):
  #ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
  ARCH = [32 , 32, 'M', 64, 64,'M',128, 128, 'M', 256, 256, 'M', 512, 512]

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))

    self.backbone = nn.Sequential(OrderedDict(layers))
#    self.fc1 = nn.Linear(2024, 500)
    self.fc2 = nn.Linear(512, 2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    x = x.mean([2, 3])

    # classifier: [N, 512] => [N, 10]
#    x = self.fc1(x)
    x = self.fc2(x)
    return x
  
  def fine_grained_prune(tensor: torch.Tensor, sparsity : float, prune_type="magnitude_based"):
    if prune_type=="magnitude_based" :
        return magnitude_based_prune(tensor,sparsity)
    else:
        print("Wrong prune type is passed")
        return tensor


def recover_model(PATH,model):
  cp = torch.load(download_url(PATH), map_location="cpu")
  return model.load_state_dict(cp)


