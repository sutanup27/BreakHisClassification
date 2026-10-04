import copy
import math
from torch import nn
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from PruningNAS.Models.DenseNet import DenseBlock
from PruningNAS.Models.MobileNetV1 import DepthwiseSeparableConv
from PruningNAS.Models.MobileNetV2 import InvertedResidual
from .PrunUtillCP import ChannelPrunner, count_densenet_prunable_layers
from .PrunUtillFGP import fine_grained_prune
from ..Models.ResNetBasic import ResNetBasic
from .TrainingModulesUtills import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_url(url, model_dir='.', overwrite=False):
    import os, sys, ssl
    from urllib.request import urlretrieve
    ssl._create_default_https_context = ssl._create_unverified_context
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
        return None
        

    
def get_labels_preds(model, dataloader,criterion):
  all_labels = []
  all_preds = []
  all_outputs=[]
  for inputs, targets in dataloader:
    preds1=model(inputs.to(device))
    preds=preds1.argmax(dim=1)
    preds = preds.cpu().numpy()  # Convert to numpy array for sklearn
    all_outputs.append(preds1.cpu().detach().numpy())
    all_preds.append(preds)
    all_labels.append(targets.numpy())  # Convert to numpy array for sklearn
    loss = criterion(preds1, targets.cuda())

  all_preds=[item for sublist in all_preds for item in sublist]
  all_labels = [item for sublist in all_labels for item in sublist]
  all_outputs = [item for sublist in all_outputs for item in sublist]

  return all_labels, all_preds, all_outputs, loss

def get_prunable_weights(model):
    prunable_weights=[]
    if 'resnet' in model.__class__.__name__.lower():
        prunable_weights.append('conv1')
        for name, layer in model.named_children():
            if isinstance(layer,nn.Sequential):
                for sub_name, sub_layer in layer.named_children():
                    prunable_weights.append(name+'.'+sub_name)
    elif 'vgg' in model.__class__.__name__.lower():
        for (name, param) in model.named_modules():
            if isinstance(param, nn.Conv2d):
                prunable_weights.append(name)
    elif 'densenet' in model.__class__.__name__.lower():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parent_name = name.rsplit(".", 1)[0] if "." in name else "<root>"
                if parent_name[:-2] == 'blocks':
                    continue
                prunable_weights.append(name)
    elif 'mobilenetv1' in model.__class__.__name__.lower():
        prunable_weights.append('model.0')
        for name, module in model.model.named_children():
            if isinstance(module, DepthwiseSeparableConv):
                prunable_weights.append('model.'+name+'.depthwise')
    elif 'mobilenetv2' in model.__class__.__name__.lower():
        prunable_weights.append('features.0')
        for name, module in model.features.named_modules():
            if isinstance(module, InvertedResidual):
                if len(module.block) != 5:  # Check if expansion is used
                    prunable_weights.append('features.'+name+'.block.0')  # Expansion conv
                prunable_weights.append('features.'+name+'.block.6')  # Projection conv
        prunable_weights.append('features.18')
    else:
        print(f'model_type doesn\'t exists 3:{model.__class__.__name__}')
        exit(0)
    return prunable_weights



@torch.no_grad()
def  sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True,prune_type='FGP',select_model='Vgg-16'):
    if prune_type=='CP':
        return sensitivity_scan_CP(model, dataloader, scan_step, scan_start, scan_end, verbose,select_model)
    else:
        return sensitivity_scan_FGP(model, dataloader, scan_step, scan_start, scan_end, verbose)


def sensitivity_scan_FGP(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    names=[]
    prunable_weights = [(name, param) for (name, param) \
                          in model.named_modules() if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear)]
    for i_layer, (name, param) in enumerate(prunable_weights):
        param_clone = param.weight.detach().clone()
        accuracy = []
        print(f'Layer :{i_layer}| name :{name}')
        for sparsity in sparsities:
            fine_grained_prune(param.weight.detach(), sparsity=sparsity)
            acc,_ = evaluate(model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.weight.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
        names.append(name)
    return sparsities, accuracies, names

def sensitivity_scan_CP(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True,select_model='Vgg-16'):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    names=[]
    prunable_weights=get_prunable_weights(model)
    for i_layer, name in enumerate(prunable_weights):
        accuracy = []
        sparsity_dict=[0.0]*len(prunable_weights)
        print(f'Layer :{i_layer}| name :{name}')
        for sparsity in sparsities:
            sparsity_dict[i_layer]=float(sparsity)
            pruned_model=ChannelPrunner(model, sparsity_dict,select_model)
            acc,_ = evaluate(pruned_model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
        names.append(name)
    return sparsities, accuracies ,names


def plot_sensitivity_scan( names, sparsities, accuracies, dense_model_accuracy,save_path=None):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    # More precise control over layout
    plot_index=0
    for name in names:
        plt.plot(sparsities, accuracies[plot_index],  color='b')  # Plot first curve in blue
        plt.plot(sparsities, [lower_bound_accuracy] * len(sparsities), color='orange')  # Plot first curve in blue
        plt.xticks(np.arange(start=0.1, stop=1.0, step=0.1))
        plt.ylim(80, 100)
        plt.title(name)
        plt.xlabel('sparsity')
        plt.ylabel('top-1 accuracy')
        plt.grid(True)
        plot_index=plot_index+1
        plt.legend([
            'accuracy after pruning',
            f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
        ])
        # fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves room at top for suptitle
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path+'.'+name+'.png') 
        plt.close()

def plot_sensitivity_scan_deprecated(model, names, sparsities, accuracies, dense_model_accuracy,save_path=None):
    layer_count=len(names)
    cols= round(3*math.sqrt(layer_count/12.0))
    rows= round(layer_count/cols)
    if cols*rows<layer_count:
        cols=cols+1
    fig = plt.figure(figsize=(20*rows, 20*cols),dpi=150)  # Big enough to hold all

    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    # More precise control over layout
    gs = fig.add_gridspec(rows, cols, wspace=0.4, hspace=1)
    plot_index=0
    for i in range(rows):
        for j in range(cols):
            if plot_index>=layer_count:
                break
            ax = fig.add_subplot(gs[i, j])
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xticks(np.arange(start=0.1, stop=1.0, step=0.1))
            ax.set_ylim(80, 100)
            ax.set_title(names[plot_index])
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.grid(axis='x')
            plot_index=plot_index+1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.legend([
        'accuracy after pruning',
        f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
    ])
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves room at top for suptitle
    # fig.subplots_adjust(top=0.925,left=0.05, bottom=0.05)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path) 
    plt.close()


def recover_model(PATH,model):
        cp = torch.load(download_url(PATH), map_location="cpu")
        return model.load_state_dict(cp)


def print_model(model):
    for name, param in model.named_parameters():
            print(name, param.shape)


#   can directly leads to model size reduction and speed up.
@torch.no_grad()
def measure_latency(main_model,dummy_input, n_warmup=20, n_test=100, d='cpu'):
    model=copy.deepcopy(main_model)
    model.to(d)
    input=copy.deepcopy(dummy_input)
    input=input.to(d)
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency

def get_sparsity_dic_template(model,prun_type='CP'):
    dct={}
    if prun_type=='FGP':
        for name, param in model.named_modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
                print(f'\'{name}\':0.90,')
                dct[name]=0.90
    else:
        if isinstance(model,ResNetBasic):
            for name, layer in model.named_children():
                if isinstance(layer,nn.Conv2d):
                    key=f'{name}'
                    print(f'\'{name}\':0.90,')
                    dct[key]=0.90
                if isinstance(layer,nn.Sequential):
                    for sub_name, sub_layer in layer.named_children():
                        key=f'{name}.{sub_name}'
                        print(f'\'{name}.{sub_name}\':0.90,')
                        dct[key]=0.90
        
    return dct

def Model_to_sd(path):

    model = torch.load(path, map_location=torch.device(device))  # Use 'cpu' if necessary
    model.to(device)