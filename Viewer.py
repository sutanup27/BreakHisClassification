import os
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from DataPreprocessing import get_datasets,test_transform,train_transform


def plot_accuracy(accs):
    print(accs)
    plt.plot(range(len(accs)), accs, label='Accuracy', color='b')  # Plot first curve in blue
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning curve of accuracy')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()


def plot_loss(train_losses, test_losses):
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss', color='r')  # Plot second curve in red
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss', color='g')  # Plot second curve in red

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, training and validation loss')
    plt.title('Learning curve of training and validation loss')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()


def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(9,6, figsize=(15, 20))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()
