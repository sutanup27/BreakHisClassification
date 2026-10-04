import math
from matplotlib import pyplot as plt
from pathlib import Path
from torch import nn
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
import matplotlib.image as mpimg

from PruningNAS.Utills.Utill import get_prunable_weights


def plot_accuracy(accs,titel_append='',save_path=None,):
    print(accs)
    plt.plot(range(len(accs)), accs, label='Accuracy', color='b')  # Plot first curve in blue
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning curve of accuracy'+titel_append)

    # Show the legend
    plt.legend()

    if save_path is None:
        plt.show()     # Display the plot
    else:
        plt.savefig(save_path) # or save 
    plt.close()


def plot_loss(train_losses, test_losses, titel_append='',save_path=None):
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss', color='r')  # Plot second curve in red
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss', color='g')  # Plot second curve in red

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, training and validation loss')
    plt.title('Learning curve of training and validation loss'+titel_append)

    # Show the legend
    plt.legend()

    if save_path is None:
        plt.show()     # Display the plot
    else:
        plt.savefig(save_path) # or save 
    plt.close()

def get_params(model,p_name):
    params=[]
    if model.__class__.__name__.lower() == 'densenet':
        pw=get_prunable_weights(model)
        for name, conv in pw:   
            params.append(conv.weight.detach().view(-1))
    else:
        for name, param in model.named_parameters():
            if (p_name==name[:len(p_name)]) and (('conv' in name) or ('fc' in name) or ('shortcut.0' in name)):
                params.append(param.detach().view(-1))
    params=torch.cat(params)
    return params

def plot_weight_distribution( model,names, bins=256, count_nonzero_only=False,save_path=None):
    # More precise control over layout

    for name in names:
        param_cpu = get_params(model,name).cpu()
        if count_nonzero_only:
            param_cpu = param_cpu[param_cpu != 0].view(-1)
            plt.hist(param_cpu, bins=bins, density=True,
                    color = 'blue', alpha = 0.5)
        else:
            plt.hist(param_cpu, bins=bins, density=True,
                    color = 'blue', alpha = 0.5)
        plt.xlabel(name)
        plt.ylabel('density')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path+'.'+name+'.png') 
        plt.close()

def plot_weight_distribution_depricated(model, bins=256, count_nonzero_only=False):
    layer_count=0
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
            layer_count=   layer_count+1
    col= round(3*math.sqrt(layer_count/12.0))
    row= round(layer_count/col)
    if col*row<layer_count:
        col=col+1
    fig, axes = plt.subplots(row,col, figsize=(10*col,12*row),constrained_layout=True)
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.weight.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.weight.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout(h_pad=15,w_pad=5)
    fig.subplots_adjust(top=0.925,left=0.05, bottom=0.05)
    plt.show()
    plt.close()


def accumulate_plot_figures(save_path):
    path=Path(save_path)
    if path.exists():
        print("Folder exists!")
    else:
        print("Folder not found.")
        exit()
    # More precise control over layout
    files=[]
    for file in path.iterdir():
        files.append(file)

    col= round(4*math.sqrt(len(files)/12.0))
    row= round(len(files)/col)
    if col*row<len(files):
        col=col+1
    fig, axes = plt.subplots(row, col, figsize=(col * 30, row * 20))
    i=0
    for r in range(row):
        for c in range(col):
            if i>=len(files):
                break
            img = mpimg.imread(files[i])
            axes[r, c].imshow(img)
            axes[r, c].axis('off')
            i=i+1
    plt.tight_layout(pad=-0.00)
    plt.show()
    plt.savefig(save_path)
    plt.close()
