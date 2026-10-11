"""Minimal reconstructible checkpoints for supported commands."""
from pathlib import Path
import pickle
import torch
from torch import nn
from torch.nn.utils import prune

from .models.vgg import VGG


def create_model(architecture, classes=2, pretrained=False):
    if architecture=='vgg':
        return VGG(classes=classes)
    if architecture in ('resnet','resnet18'):
        from torchvision.models import resnet18
        model = resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features,classes)
        return model
    raise ValueError(f'Unsupported architecture: {architecture}')


def save_checkpoint(path,model,architecture,metadata,epoch=0):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = dict(format_version=1,architecture=architecture,classes=['benign','malignant'],
                   state_dict={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                   metadata=metadata,epoch=epoch)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def load_checkpoint(path,device='cpu'):
    try:
        payload = torch.load(path,map_location='cpu',weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Legacy whole-model checkpoint: explicit migration is required; supported commands load new format checkpoints only.') from error
    if not isinstance(payload,dict) or payload.get('format_version')!=1:
        raise ValueError('Expected a new breakhis checkpoint. Legacy whole-model files require explicit migration.')
    model = create_model(payload['architecture'],len(payload['classes']))
    modules = dict(model.named_modules())
    for key in payload['state_dict']:
        if key.endswith('.weight_orig'):
            prune.identity(modules[key.removesuffix('.weight_orig')],'weight')
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device), payload
