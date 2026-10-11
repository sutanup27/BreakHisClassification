"""BreaKHis image loading. CIFAR-10 lives in experiments/cifar10."""
import random
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor,
)
from .manifest import validate_splits

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1]/'artifacts/data/manifest.json'


class ManifestDataset(Dataset):
    classes = ['benign', 'malignant']
    class_to_idx = {'benign':0, 'malignant':1}

    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root).resolve(), rows, transform
        if not rows:
            raise ValueError('Selected manifest split is empty')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root/row['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Manifest image path escapes dataset root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Image changed since audit; rebuild manifest: {path}')
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert('RGB')
        return self.transform(image), self.class_to_idx[row['label']]


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# Shared scale and output size; resizing the shorter edge preserves aspect ratio.
image_size = 224
train_transform = Compose([
    Resize(256), RandomCrop(image_size, padding=4), RandomHorizontalFlip(), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = Compose([
    Resize(256), CenterCrop(image_size), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(path, train_transform=train_transform,
                 test_transform=test_transform,
                 train_test_val_pecentage=None, *, dataset="breakhis", manifest=DEFAULT_MANIFEST):
    """Compatibility pair: training and validation (never the reserved test set)."""
    if dataset != "breakhis":
        raise ValueError("This loader supports dataset='breakhis' only. "
                         "Use experiments.cifar10.DataPreprocessing for CIFAR-10.")
    if train_test_val_pecentage is not None:
        raise ValueError('Split fractions are saved in the manifest; rebuild it to change them')
    folder = Path(path).resolve()
    payload = read_manifest(manifest)
    source = next((key for key,root in payload['roots'].items()
                   if folder.parent == Path(root).resolve()), None)
    if source not in ('breakhis_v2','cancer_binary'):
        raise ValueError('Select an audited binary dataset magnification folder')
    datasets = get_split_datasets(manifest,source,folder.name,
                                  train_transform=train_transform,test_transform=test_transform)
    return datasets['train'], datasets['validation']


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Dataset manifest missing. Run python -m breakhis prepare-data first.')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported manifest schema')
    validate_splits(payload['rows'])
    return payload


def get_split_datasets(manifest=DEFAULT_MANIFEST, source='breakhis_v2', magnification='40X',
                       data_root=None, train_transform=train_transform, test_transform=test_transform):
    if source not in ('breakhis_v2','cancer_binary'):
        raise ValueError('Primary commands support binary datasets only')
    payload = read_manifest(manifest)
    root = Path(data_root or payload['roots'][source])
    rows = [row for row in payload['rows'] if row['source']==source and
            (magnification=='all' or row['magnification']==magnification)]
    return {split:ManifestDataset(root,[row for row in rows if row['split']==split],
                                  train_transform if split=='train' else test_transform)
            for split in ('train','validation','test')}


def get_class_names(dataset):
    """Read labels from a dataset or a split wrapped in Subset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return list(dataset.classes)


def get_dataloaders(path, train_transform=train_transform,
                    test_transform=test_transform,
                    train_test_val_pecentage=None, batch_size=32,
                    *, dataset="breakhis", num_workers=0,
                    pin_memory=False, deterministic=True, manifest=DEFAULT_MANIFEST):
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    set_seed(0, deterministic=deterministic)
    train_dataset, test_dataset = get_datasets(
        path, train_transform, test_transform, train_test_val_pecentage,
        dataset=dataset, manifest=manifest,
    )
    loader_options = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(test_dataset, shuffle=False, **loader_options),
    )


def get_split_loaders(manifest=DEFAULT_MANIFEST, source='breakhis_v2', magnification='40X',
                      data_root=None, batch_size=32, num_workers=0, device='cpu', seed=0,
                      deterministic=False):
    if num_workers < 0 or batch_size < 1:
        raise ValueError('Invalid loader settings')
    set_seed(seed, deterministic)
    datasets = get_split_datasets(manifest,source,magnification,data_root)
    options = dict(batch_size=batch_size,num_workers=num_workers,pin_memory=str(device).startswith('cuda'))
    if num_workers:
        options.update(persistent_workers=True,prefetch_factor=2)
    return {split:DataLoader(dataset,shuffle=split=='train',**options)
            for split,dataset in datasets.items()}
