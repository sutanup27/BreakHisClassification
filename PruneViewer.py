import os
import pickle
import random
import torch
from DataPreprocessing import get_dataloaders
from TrainingModules import evaluate
from Utill import plot_sensitivity_scan, sensitivity_scan
from VGG import VGG
from Viewer import plot_weight_distribution  # Ensure you import your correct model architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Initialize the model
model_type='vgg'
magf='400X'
path='..\Dataset\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']
model_path='checkpoint\\400x\\vgg\\vgg_95.05494689941406.pth'
root_dir=os.path.join(path, magf)
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
model.to(device)
plot_weight_distribution(model)
train_dataloader,test_dataloader=get_dataloaders(root_dir,batch_size=32 )
############# calculate sparsities (optional) #############################################
acc_pkl=f'checkpoint/{magf}/{model_type}/accuracies_{model_type}.pkl'
sparse_pkl=f'checkpoint/{magf}/{model_type}/sparsities_{model_type}.pkl'
sparsities, accuracies = sensitivity_scan(
    model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0)

with open(sparse_pkl, "wb") as f:
    pickle.dump(sparsities, f)

with open(acc_pkl, "wb") as f:
    pickle.dump(accuracies, f)

############################################################################################
with open(sparse_pkl, "rb") as f:
    sparsities = pickle.load(f)

with open(acc_pkl, "rb") as f:
    accuracies = pickle.load(f)
print(accuracies)
print(sparsities)
dense_model_accuracy,_=evaluate(model,test_dataloader)
print(dense_model_accuracy)
plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy)