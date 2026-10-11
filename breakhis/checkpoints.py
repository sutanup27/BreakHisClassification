"""Versioned state-dictionary checkpoints and epoch-boundary training resume."""
from pathlib import Path
import pickle
import random
import platform
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from .models.vgg import VGG


def create_model(architecture, classes=2, pretrained=False, channels=None):
    if architecture == 'vgg':
        return VGG(classes=classes, channels=channels)
    if channels is not None:
        raise ValueError('Channel configurations are supported only for VGG')
    if architecture in ('resnet', 'resnet18'):
        from torchvision.models import resnet18
        model = resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    raise ValueError(f'Unsupported architecture: {architecture}')


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def capture_random_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def restore_random_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same CUDA device count as the saved RNG state')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    torch.backends.cudnn.deterministic = state['cudnn_deterministic']
    torch.backends.cudnn.benchmark = state['cudnn_benchmark']
    torch.use_deterministic_algorithms(state['deterministic_algorithms'])


def training_state(optimizer, scheduler, best_accuracy, config):
    return dict(optimizer=cpu_state(optimizer.state_dict()),
                scheduler=cpu_state(scheduler.state_dict()),
                random_state=capture_random_state(), best_accuracy=best_accuracy,
                planned_epochs=config['epochs'], device=config['resolved_device'],
                resumable=config['num_workers'] == 0,
                boundary='completed_epoch')


def save_checkpoint(path, model, architecture, metadata, epoch=0, *,
                    classes=None, config=None, state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = classes or ['benign', 'malignant']
    masks = [key.removesuffix('.weight_mask') for key in model.state_dict() if key.endswith('.weight_mask')]
    architecture_config = {'classes': len(classes)}
    if isinstance(model, VGG) and model.channels != [value for value in VGG.ARCH if value != 'M']:
        architecture_config['channels'] = model.channels
    pruning_config = {'type': 'unstructured' if masks else 'dense', 'masked_modules': masks}
    if hasattr(model, 'channel_pruning'):
        if masks:
            raise ValueError('Mixed channel and weight-mask checkpoints are unsupported')
        architecture_config['channels'] = model.channels
        pruning_config.update(type='channel', channel_config=model.channel_pruning)
    payload = dict(format_version=2, architecture=architecture,
                   architecture_config=architecture_config, classes=list(classes),
                   class_to_idx={label: index for index, label in enumerate(classes)},
                   state_dict=cpu_state(model.state_dict()), metadata=metadata, epoch=epoch,
                   config=config or {}, training_state=state,
                   pruning=pruning_config,
                   runtime={'python': platform.python_version(), 'torch': str(torch.__version__),
                            'cuda': torch.version.cuda})
    temp = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Checkpoint file does not exist: {path}')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Legacy whole-model checkpoint: explicit migration is required; only state-dictionary checkpoints are supported.') from error
    if not isinstance(payload, dict) or payload.get('format_version') not in (1, 2):
        raise ValueError('Expected a breakhis state-dictionary checkpoint; legacy whole-model files require migration.')
    required = {'architecture', 'classes', 'state_dict', 'metadata', 'epoch'}
    if required - payload.keys():
        raise ValueError(f'Checkpoint is missing fields: {sorted(required - payload.keys())}')
    if payload['architecture'] not in ('vgg', 'resnet', 'resnet18'):
        raise ValueError('Unsupported checkpoint architecture')
    if not isinstance(payload['classes'], list) or len(payload['classes']) < 2 or not all(
            isinstance(label, str) for label in payload['classes']) or len(set(payload['classes'])) != len(payload['classes']):
        raise ValueError('Invalid checkpoint class labels')
    if not isinstance(payload['epoch'], int) or isinstance(payload['epoch'], bool) or payload['epoch'] < 0:
        raise ValueError('Invalid checkpoint epoch')
    if not isinstance(payload['state_dict'], dict) or not isinstance(payload['metadata'], dict):
        raise ValueError('Invalid checkpoint weights or metadata')
    if payload['format_version'] == 2:
        expected = {label: index for index, label in enumerate(payload['classes'])}
        architecture_config = payload.get('architecture_config', {})
        if payload.get('class_to_idx') != expected or architecture_config.get('classes') != len(expected) or set(architecture_config) - {'classes', 'channels'}:
            raise ValueError('Checkpoint label mapping or architecture configuration is inconsistent')
        channels = architecture_config.get('channels')
        if channels is not None and (payload['architecture'] != 'vgg' or not isinstance(channels, list) or len(channels) != 8
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in channels)):
            raise ValueError('Invalid VGG checkpoint channel configuration')
        if not isinstance(payload.get('config'), dict):
            raise ValueError('Invalid checkpoint configuration')
        masks = [key.removesuffix('.weight_mask') for key in payload['state_dict'] if key.endswith('.weight_mask')]
        pruning_config = payload.get('pruning', {})
        expected_pruning = {'type': 'unstructured' if masks else 'dense', 'masked_modules': masks}
        if pruning_config.get('type') == 'channel':
            channel_config = pruning_config.get('channel_config', {})
            if masks or channels is None or channel_config.get('channels') != channels or channel_config.get('method') != 'l1-output-channel':
                raise ValueError('Inconsistent channel pruning metadata')
            from .pruning import validate_channel_config
            validate_channel_config(channel_config, channels)
            expected_pruning = {'type': 'channel', 'masked_modules': [], 'channel_config': channel_config}
        if pruning_config != expected_pruning:
            raise ValueError('Checkpoint pruning configuration is inconsistent')
        for key, mask in payload['state_dict'].items():
            if key.endswith('.weight_mask') and not torch.all((mask == 0) | (mask == 1)):
                raise ValueError('Checkpoint pruning masks must be binary')
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
               for value in payload['state_dict'].values()):
        raise ValueError('Checkpoint contains invalid or nonfinite weights')
    return payload


def load_checkpoint(path, device='cpu', *, payload=None):
    payload = payload if payload is not None else read_checkpoint(path)
    options = {}
    channels = payload.get('architecture_config', {}).get('channels')
    if channels is not None:
        options['channels'] = channels
    model = create_model(payload['architecture'], len(payload['classes']), **options)
    modules = dict(model.named_modules())
    for key in payload['state_dict']:
        if key.endswith('.weight_orig'):
            prune.identity(modules[key.removesuffix('.weight_orig')], 'weight')
    model.load_state_dict(payload['state_dict'], strict=True)
    if payload.get('pruning', {}).get('type') == 'channel':
        model.channel_pruning = payload['pruning']['channel_config']
    return model.to(device), payload


def restore_training_state(payload, optimizer, scheduler, config):
    state = payload.get('training_state')
    if not state or not state.get('resumable') or state.get('boundary') != 'completed_epoch':
        raise ValueError('Checkpoint cannot resume exactly; use a completed-epoch -last.pth checkpoint saved with num-workers 0')
    previous = payload['config']
    for field in ('command', 'epochs', 'learning_rate', 'batch_size', 'num_workers', 'deterministic', 'resolved_device'):
        if config[field] != previous[field]:
            raise ValueError(f'Resume configuration differs: {field} must be {previous[field]!r}')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_random_state(state['random_state'])
    return payload['epoch'] + 1, state['best_accuracy']
