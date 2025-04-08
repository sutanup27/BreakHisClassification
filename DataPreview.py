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

path='./mrleyedataset'

# Load images using ImageFolder
image_dataset = ImageFolder(root=path, transform=transforms.ToTensor())

samples={}
# Before transformation
samples['Before']=[[],[]]
for d , l in image_dataset:
    if len(samples['Before'][l])< 2 :
        samples['Before'][l].append(d)
#    print(len(samples[0]),len(samples[1]))
    if len(samples['Before'][0])==4 and len(samples['Before'][1])==4:
        break

# Plot the images
width=2
fig, ax = plt.subplots(2, width, figsize=(40, 40))
fig.set_facecolor('lightgrey')
e_status=['Eye close','Eye open']
for l in [0,1]:
    for i in range(2):
        image=samples['Before'][l][i]
        print(image.shape)
        ax[i,l].imshow(image.permute(1, 2, 0),cmap='gray')
        ax[i,l].set_title(f"Label:{e_status[l]}", fontsize=50)
        ax[i,l].axis("off")
plt.show()

print("End")
