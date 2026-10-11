"""BreaKHis image loading. CIFAR-10 lives in experiments/cifar10."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Preserve existing preprocessing until the dedicated preprocessing fix.
image_size = 224
train_transform = Compose([
    RandomCrop(image_size, padding=4), RandomHorizontalFlip(), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = Compose([
    Resize((460, 700)), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(Path, train_transform=train_transform,
                 test_transform=test_transform,
                 train_test_val_pecentage=(0.80, 0.20), *, dataset="breakhis"):
    if dataset != "breakhis":
        raise ValueError("This loader supports dataset='breakhis' only. "
                         "Use experiments.cifar10.DataPreprocessing for CIFAR-10.")
    image_dataset = ImageFolder(root=Path)
    if image_dataset.classes != ["benign", "malignant"]:
        raise ValueError("Expected BreaKHis class folders 'benign' and 'malignant'; "
                         f"found {image_dataset.classes}. Select one magnification folder.")
    test_size = int(train_test_val_pecentage[1] * len(image_dataset))
    train_size = len(image_dataset) - test_size
    train_indices, test_indices = random_split(image_dataset, [train_size, test_size])
    return (
        Subset(ImageFolder(root=Path, transform=train_transform), train_indices.indices),
        Subset(ImageFolder(root=Path, transform=test_transform), test_indices.indices),
    )


def get_class_names(dataset):
    """Read labels from a dataset or a split wrapped in Subset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return list(dataset.classes)


def get_dataloaders(Path, train_transform=train_transform,
                    test_transform=test_transform,
                    train_test_val_pecentage=(0.80, 0.20), batch_size=32,
                    *, dataset="breakhis"):
    set_seed(0)
    train_dataset, test_dataset = get_datasets(
        Path, train_transform, test_transform, train_test_val_pecentage,
        dataset=dataset,
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
