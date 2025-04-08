

import copy
import os
from matplotlib import path
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from DataPreprocessing import get_dataloaders,train_transform, test_transform
from TrainingModules import evaluate
from VGG import VGG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sparsity_dict = {
'backbone.conv0.weight': 0.4,
'backbone.conv1.weight': 0.90,
'backbone.conv2.weight': 0.80,
'backbone.conv3.weight': 0.6,
'backbone.conv4.weight': 0.9,
'backbone.conv5.weight': 0.9,
'backbone.conv6.weight': 0.8,
'backbone.conv7.weight': 0.95,
'backbone.conv8.weight': 0.95,
'backbone.conv9.weight': 0.97,
'fc2.weight': 0.95,
}
model=VGG()

magf='40X'
path='..\Dataset\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']
model_path='checkpoint\\40x\\vgg_95.48872375488281.pth'
root_dir=os.path.join(path, magf)
dataloader={}

# Load the saved state_dict correctly
# state_dict = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
# missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
model=torch.load(model_path,map_location=torch.device(device))
model.to(device)
train_dataloader,test_dataloader=get_dataloaders(root_dir ) # Basemodel

dense_model_accuracy=evaluate(model,test_dataloader)
print('dense_model_accuracy:',dense_model_accuracy)

# torch.save(model,model_path)
# loaded_model=torch.load(model_path,map_location=torch.device(device))

# loaded_model_accuracy=evaluate(loaded_model,test_dataloader)
# print('loaded_model_accuracy:',loaded_model_accuracy)
