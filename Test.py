import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
from DataPreprocessing import get_dataloaders,train_transform,test_transform
from Model_Evaluation import get_model_macs, get_model_size, get_model_sparsity
from Utill import get_labels_preds, measure_latency, print_model
from VGG import VGG
from TrainingModules import evaluate  # Ensure you import your correct model architecture
import torch

#fix the randomness
seed=0
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
print(device)
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='checkpoint/400x/vgg/vgg_CP_93.13186645507812.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
model.to(device)
# Print out missing/unexpected keys for debugging
magf='400X'

path='..\Dataset\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']

root_dir=os.path.join(path, magf)
dataloader={}
train_dataloader, test_dataloader=get_dataloaders(root_dir)

########################################test model#####################################
# Set model to evaluation mode
# model.eval()
input_tensor=torch.randn(1, 3, 224, 224).to(device)
# output = model(input_tensor)  # Ensure input_tensor is properly formatted
# print('output:',output)
#######################################################################################

########################################test model#####################################
print_model(model)

metric,_ = evaluate(model, test_dataloader)
print("accuracy:",metric)
#######################################################################################
######################################## model metrics ################################
macs =get_model_macs(model,input_tensor)
sparsity =get_model_sparsity(model)
model_size =get_model_size(model,count_nonzero_only=True)
print('macs:',macs)
print('sparsity:',sparsity)
print(f'model size:{model_size/ MiB:.2f}MB')
#######################################################################################
all_labels, all_preds = get_labels_preds(model,test_dataloader,nn.CrossEntropyLoss())
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
conf_matrix=confusion_matrix(all_labels, all_preds)
print(conf_matrix)
latency=measure_latency(model)
print('CPU latency:',latency)
latency=measure_latency(model,d=device)
print('GPU latency:',latency)