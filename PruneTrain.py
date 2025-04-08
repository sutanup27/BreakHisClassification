import os
from torch import nn
import copy
import random
import torch
from DataPreprocessing import get_dataloaders
from Model_Evaluation import get_model_size, get_sparsity
from TrainingModules import TrainingPrunned, evaluate, train
from Utill import ChannelPrunner, FineGrainedPruner, plot_sensitivity_scan, print_model, sensitivity_scan
from VGG import VGG
from Viewer import plot_accuracy, plot_weight_distribution  # Ensure you import your correct model architecture
seed=0
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
# Initialize the model

magf='400X'
path='..\Dataset\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']
model_path='checkpoint\\400x\\vgg\\vgg_95.05494689941406.pth'
root_dir=os.path.join(path, magf)
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary

model.to(device)
select_model='vgg'
pruning_type='CP'


sparsity_dict = {      #for FGP
'backbone.conv0.weight': 0.1,
'backbone.conv1.weight': 0.4,
'backbone.conv2.weight': 0.45,
'backbone.conv3.weight': 0.6,
'backbone.conv4.weight': 0.6,
'backbone.conv5.weight': 0.65,
'backbone.conv6.weight': 0.8,
'backbone.conv7.weight': 0.9,
'fc2.weight': 0.6,
}

# sparsity_dict = {       #for CP
# 'backbone.conv0.weight': 0.2,
# 'backbone.conv1.weight': 0.7,
# 'backbone.conv2.weight': 0.6,
# 'backbone.conv3.weight': 0.5,
# 'backbone.conv4.weight': 0.7,
# 'backbone.conv5.weight': 0.9,
# 'backbone.conv6.weight': 0.6,
# 'backbone.conv7.weight': 0.7,
# 'backbone.conv8.weight': 0.8,
# 'backbone.conv9.weight': 0.9,
# 'fc2.weight': 0.95,
# }

train_dataloader,test_dataloader=get_dataloaders(root_dir,batch_size=16 ) # Basemodel
dense_model_accuracy=evaluate(model,test_dataloader)
print('dense_model_accuracy:',dense_model_accuracy)
pruned_model=copy.deepcopy(model)
if pruning_type=='FGP':
    isCallback=True
    pruner = FineGrainedPruner(pruned_model, sparsity_dict)
elif pruning_type=='CP':
    pruned_model = ChannelPrunner(pruned_model, sparsity_dict)
    pruner=None
    isCallback=False
else:
    exit

print_model(pruned_model)
print(f'The sparsity of each layer becomes')
for name, param in pruned_model.named_parameters():
    if name in sparsity_dict:
        print(f'  {name}: {get_sparsity(param):.2f}')

dense_model_size = get_model_size(model, count_nonzero_only=True)
sparse_model_size = get_model_size(pruned_model, count_nonzero_only=True)

print(f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
sparse_model_accuracy,_ = evaluate(pruned_model, test_dataloader)
print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% before fintuning")

num_finetune_epochs = 20
optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
criterion = nn.CrossEntropyLoss()

pruned_model_accuracy,best_pruned_model,accuracies=TrainingPrunned(pruned_model,train_dataloader,test_dataloader,criterion, optimizer, pruner,scheduler=None,num_finetune_epochs=num_finetune_epochs,isCallback=isCallback)

torch.save(best_pruned_model, f'./checkpoint/{magf}/{select_model}/{select_model}_{pruning_type}_{pruned_model_accuracy}.pth')

sparse_model_accuracy,_ = evaluate(best_pruned_model, test_dataloader)

print(sparse_model_accuracy)
plot_accuracy(accuracies)
