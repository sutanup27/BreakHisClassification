from multiprocessing import freeze_support
import random
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
from PruningNAS.DataProcess.DataPreprocessing import get_dataloaders
from PruningNAS.Utills.EvaluatiorUtills import get_model_macs, get_model_size, get_model_sparsity
from PruningNAS.Utills.Utill import get_labels_preds, measure_latency, print_model
from PruningNAS.Utills.TrainingModulesUtills import evaluate  # Ensure you import your correct model architecture
import torch

def main():
    #fix the randomness
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    # Move model and tensors to device

    path='./dataset/cifar10'
    # Initialize the model

    model_path=r'PruningNAS\checkpoint\MobilenetV1\FGP\MobilenetV1_cifar_FGP_93.78999328613281.pth'
    # Load the saved state_dict correctly
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)  # Use 'cpu' if necessary
    model.to(device)

    # Print out missing/unexpected keys for debugging

    train_dataloader,test_dataloader=get_dataloaders(path)
    ########################################test model#####################################
    # Set model to evaluation mode
    # model.eval()
    input_tensor=torch.randn(1, 3, 32, 32).to(device)
    # output = model(input_tensor)  # Ensure input_tensor is properly formatted
    # print('output:',output)
    #######################################################################################

    ########################################test model#####################################
    print(next(model.parameters()).device)
    print_model(model)

    metric,_ = evaluate(model, test_dataloader)
    print("accuracy:",metric)
    #######################################################################################
    ######################################## model metrics ################################
    macs =get_model_macs(model)
    sparsity =get_model_sparsity(model)
    model_size =get_model_size(model,count_nonzero_only=True)
    print('macs:',macs)
    print('sparsity:',sparsity)
    print(f'model size:{model_size:.2f}MB')
    #######################################################################################
    all_labels, all_preds,all_outputs,loss = get_labels_preds(model,test_dataloader,nn.CrossEntropyLoss())
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    conf_matrix=confusion_matrix(all_labels, all_preds)
    print(conf_matrix)
    latency=measure_latency(model,input_tensor)
    print('CPU latency:',latency)
    latency=measure_latency(model,input_tensor,d=device)
    print('GPU latency:',latency)


if __name__ == '__main__':
    main()
    freeze_support()