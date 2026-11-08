"""BreaKHis image loading from audited, group-separated manifests."""
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
from .manifest import validate_splits, validate_source_rows, requested_splits, MAGNIFICATIONS

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
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f'{row["split"]}: cannot read image: {path}') from error
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'{row["split"]}: Image changed since audit; rebuild manifest: {path}')
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.load()
                image = opened.convert('RGB')
                pixel_hash=hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest()
                if pixel_hash != row['pixel_sha256']:
                    raise ValueError(f'{row["split"]}: decoded pixels differ from audit: {path}')
        except OSError as error:
            raise ValueError(f'{row["split"]}: cannot decode image: {path}') from error
        return self.transform(image), self.class_to_idx[row['label']]


def set_seed(seed=42, deterministic=True):
    import os
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


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
        raise ValueError("This loader supports dataset='breakhis' only.")
    if train_test_val_pecentage is not None:
        raise ValueError('Split fractions are saved in the manifest; rebuild it to change them')
    folder = Path(path).resolve()
    payload = read_manifest(manifest)
    source = next((key for key,root in payload['roots'].items()
                   if folder.parent == Path(root).resolve()), None)
    if source not in ('breakhis_v2','cancer_binary'):
        raise ValueError('Select an audited binary dataset magnification folder')
    datasets = get_split_datasets(manifest,source,folder.name,
                                  train_transform=train_transform,test_transform=test_transform,
                                  splits=('train','validation'))
    return datasets['train'], datasets['validation']


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Dataset manifest missing. Run python -m breakhis prepare-data first.')
    from .experiment_io import read_json
    payload = read_json(path)
    if not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        raise ValueError('Unsupported manifest schema')
    if not isinstance(payload.get('roots'),dict) or any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in payload['roots'].items()):
        raise ValueError('Invalid manifest roots')
    if 'rows' not in payload: raise ValueError('Manifest rows missing')
    validate_splits(payload['rows'])
    return payload


def manifest_selection(payload, source, magnification):
    if source not in payload['roots']: raise ValueError('Selected source missing from manifest: '+source)
    if magnification not in (*MAGNIFICATIONS,'all'): raise ValueError('Unknown magnification')
    rows=validate_source_rows(payload['rows'],source)
    return [row for row in rows if magnification=='all' or row['magnification']==magnification]


def validate_role_rows(rows, root, source, magnification, split):
    if not rows:
        raise ValueError(f'Selected {source}/{magnification} manifest split is empty: {split}')
    if {row['label'] for row in rows}!={'benign','malignant'}:
        raise ValueError('Both classes are required in selected '+split+' split')
    root=Path(root).resolve()
    if not root.is_dir(): raise ValueError('Selected data root is missing: '+str(root))
    for row in rows:
        candidate=(root/row['path']).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f'Unsafe or missing image in {split}: {row["path"]}')


def get_split_datasets(manifest=DEFAULT_MANIFEST, source='breakhis_v2', magnification='40X',
                       data_root=None, train_transform=train_transform, test_transform=test_transform,
                       *, splits=('train','validation','test')):
    if source not in ('breakhis_v2','cancer_binary'):
        raise ValueError('Primary commands support binary datasets only')
    payload = read_manifest(manifest)
    splits=requested_splits(splits)
    rows = manifest_selection(payload,source,magnification)
    root = Path(data_root or payload['roots'][source])
    for split in splits:
        validate_role_rows([row for row in rows if row['split']==split],root,source,magnification,split)
    return {split:ManifestDataset(root,[row for row in rows if row['split']==split],
                                  train_transform if split=='train' else test_transform)
            for split in splits}


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
                      deterministic=False, *, splits=('train','validation','test')):
    if num_workers < 0 or batch_size < 1:
        raise ValueError('Invalid loader settings')
    set_seed(seed, deterministic)
    datasets = get_split_datasets(manifest,source,magnification,data_root,splits=splits)
    options = dict(batch_size=batch_size,num_workers=num_workers,pin_memory=str(device).startswith('cuda'))
    if num_workers:
        options.update(persistent_workers=True,prefetch_factor=2)
    return {split:DataLoader(dataset,shuffle=split=='train',**options)
            for split,dataset in datasets.items()}
