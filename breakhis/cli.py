"""Import-safe commands with recorded configuration and resumable training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA_DEFAULTS = dict(manifest=ROOT/'artifacts/data/manifest.json', source='cancer_binary',
                     magnification='40X', data_root=None, batch_size=32,
                     num_workers=0, device='auto', deterministic=False)


def add_data_options(parser):
    parser.add_argument('--dataset', choices=['breakhis'], required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--source', choices=['breakhis_v2', 'cancer_binary'])
    parser.add_argument('--magnification', choices=['40X', '100X', '200X', '400X', 'all'])
    parser.add_argument('--data-root', type=Path, help='Override the selected source root after relocation')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--experiment-id', help='Run directory identifier; generated when omitted')


def resolve_input(path, label, directory=False):
    path = Path(path).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise ValueError(f'{label} {"directory" if directory else "file"} does not exist: {path}')
    return path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='BreaKHis classification, pruning and dataset auditing')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-data', help='Audit sources and save patient-separated splits')
    prepare.add_argument('--dataset-root', type=Path, default=ROOT.parent/'Dataset')
    prepare.add_argument('--output', type=Path, default=ROOT/'artifacts/data')
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--workers', type=int, default=8)
    for name in ('train', 'evaluate', 'prune', 'finetune'):
        command = commands.add_parser(name)
        add_data_options(command)
        command.add_argument('--output', type=Path, help='Checkpoint filename/path inside checkpoint/binary, or evaluation JSON path')
        if name in ('train', 'finetune'):
            command.add_argument('--epochs', type=int, help='Total planned epochs; default 20 or saved resume value')
            command.add_argument('--learning-rate', type=float, help='Initial SGD rate; default .001 or saved resume value')
            command.add_argument('--resume', type=Path, help='Restore a completed-epoch checkpoint and its optimizer/scheduler/RNG state')
        if name == 'train':
            command.add_argument('--model', choices=['vgg', 'resnet', 'resnet18'])
            command.add_argument('--pretrained', action='store_true', help='Initialize ResNet with downloaded pretrained weights')
        else:
            command.add_argument('--checkpoint', type=Path, required=name != 'finetune')
        if name == 'prune':
            command.add_argument('--sparsity', type=float, default=.5)
    args = parser.parse_args(argv)
    if args.command == 'prepare-data':
        if args.workers < 1:
            parser.error('workers must be positive')
        from .manifest import prepare_dataset
        try:
            root = resolve_input(args.dataset_root, 'Dataset root', directory=True)
            prepare_dataset(root, args.output.expanduser().resolve(), args.seed, args.workers)
        except ValueError as error:
            parser.error(str(error))
        return

    from .paths import checkpoint_output, validate_experiment_id
    resume = getattr(args, 'resume', None)
    payload, previous = None, {}
    try:
        # Validate checkpoint paths before data scans or model computation.
        if resume and getattr(args, 'checkpoint', None):
            raise ValueError('Use --resume or --checkpoint, not both')
        input_path = resume or getattr(args, 'checkpoint', None)
        if args.command == 'finetune' and input_path is None:
            raise ValueError('Fine-tuning requires --checkpoint or --resume')
        if input_path:
            input_path = resolve_input(input_path, 'Checkpoint')
        if resume:
            from .checkpoints import read_checkpoint
            payload = read_checkpoint(input_path)
            previous = payload.get('config', {})
            state = payload.get('training_state')
            if not state or not state.get('resumable'):
                raise ValueError('Exact resume requires a completed-epoch checkpoint saved with num-workers 0')
            if previous.get('command') != args.command:
                raise ValueError('Resume command must match the saved training command')
            args.resume = input_path
        elif input_path:
            args.checkpoint = input_path
        for field, default in DATA_DEFAULTS.items():
            if getattr(args, field) is None:
                setattr(args, field, previous.get(field, default))
        args.manifest = resolve_input(args.manifest, 'Manifest')
        if args.data_root:
            args.data_root = resolve_input(args.data_root, 'Data root', directory=True)
        if args.batch_size < 1 or args.num_workers < 0:
            raise ValueError('batch size must be positive; workers must be nonnegative')
        if args.command in ('train', 'finetune'):
            args.epochs = args.epochs if args.epochs is not None else previous.get('epochs', 20)
            args.learning_rate = args.learning_rate if args.learning_rate is not None else previous.get('learning_rate', .001)
            if args.epochs < 1:
                raise ValueError('epochs must be positive')
            if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
                raise ValueError('learning rate must be finite and positive')
            if resume and args.num_workers != 0:
                raise ValueError('Exact resume requires num-workers 0; background worker RNGs are not saved')
        if args.command == 'train':
            args.model = args.model or previous.get('model', 'vgg')
            if resume and args.model != payload['architecture']:
                raise ValueError('Resume model differs from the saved architecture')
            if resume and args.pretrained:
                raise ValueError('--pretrained initializes a new model and cannot be used with --resume')
        if args.command == 'prune' and not 0 <= args.sparsity <= 1:
            raise ValueError('sparsity must be between zero and one')
        args.experiment_id = validate_experiment_id(args.experiment_id or previous.get('experiment_id') or
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        if resume and args.experiment_id != previous['experiment_id']:
            raise ValueError('Resume must retain the saved experiment ID')
        if args.output:
            args.output = args.output.expanduser()
        if args.command != 'evaluate' and args.output:
            checkpoint_output(args.source, args.magnification, 'vgg', args.command, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    import torch
    from torch import nn
    from .data import get_split_loaders, read_manifest
    from .checkpoints import create_model, load_checkpoint, read_checkpoint, save_checkpoint, training_state, restore_training_state
    from .training import train, evaluate
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    # Read and check provenance before image loading or pretrained downloads.
    try:
        manifest = read_manifest(args.manifest)
        selected_root = resolve_input(args.data_root or manifest['roots'][args.source], 'Selected data root', directory=True)
        metadata = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        source=args.source, magnification=args.magnification,
                        preprocessing='resize-short-edge-256/crop224/ImageNet-normalization',
                        manifest_path=str(args.manifest), data_root=str(selected_root),
                        preprocessing_config={'resize_short_edge': 256, 'crop_size': 224,
                            'train_padding': 4, 'horizontal_flip': True,
                            'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                        split_policy={'train': 'train', 'selection': 'validation', 'final_evaluation': 'test'})
        if input_path:
            payload = payload or read_checkpoint(input_path)
            for key in ('manifest_sha256', 'source', 'magnification', 'preprocessing'):
                if payload['metadata'].get(key) != metadata[key]:
                    raise ValueError(f'Checkpoint dataset metadata differs: {key}')
            if payload['classes'] != ['benign', 'malignant']:
                raise ValueError('Checkpoint class labels differ from the binary dataset')
        architecture = args.model if args.command == 'train' else payload['architecture']
        if args.command == 'evaluate':
            output = args.output.resolve() if args.output else checkpoint_output(
                args.source, args.magnification, architecture, args.command,
                experiment_id=args.experiment_id).with_suffix('.json')
        else:
            output = checkpoint_output(args.source, args.magnification, architecture, args.command,
                args.output or (previous.get('output') if resume else None), args.experiment_id)
            occupied = (output, output.with_suffix('.config.json'), output.with_name(output.stem + '-last.pth'))
            if not resume and any(path.exists() for path in occupied):
                raise ValueError(f'Output already exists; choose another experiment ID or output: {output}')
        if input_path and output == input_path:
            raise ValueError('Output cannot replace the input checkpoint')
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(output=str(output), resolved_device=str(device), data_root=str(selected_root),
                      architecture=architecture, parent_checkpoint=str(input_path) if input_path else None)
        if resume and args.command == 'train':
            config['pretrained'] = previous.get('pretrained', False)
        if args.command in ('train', 'finetune'):
            config.update(optimizer={'name': 'SGD', 'momentum': .9, 'weight_decay': 5e-4},
                          scheduler={'name': 'CosineAnnealingLR', 'T_max': args.epochs})
        if resume:
            import platform
            if payload.get('runtime') != {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda}:
                raise ValueError('Exact resume requires the saved Python/PyTorch/CUDA runtime versions')
            for field in ('epochs', 'learning_rate', 'batch_size', 'num_workers', 'deterministic', 'resolved_device', 'output'):
                if config[field] != previous[field]:
                    raise ValueError(f'Resume configuration differs: {field} must be {previous[field]!r}')
            if payload['epoch'] >= args.epochs:
                raise ValueError('Resume checkpoint has already completed all planned epochs')
            if payload['epoch'] > 0 and not output.is_file():
                raise ValueError('Best checkpoint is missing; restore the experiment directory before resume')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))

    loaders = get_split_loaders(args.manifest, args.source, args.magnification, selected_root,
                               args.batch_size, args.num_workers, device, deterministic=args.deterministic)
    criterion = nn.CrossEntropyLoss().to(device)
    if input_path:
        model, payload = load_checkpoint(input_path, device, payload=payload)
    else:
        model = create_model(architecture, classes=len(loaders['train'].dataset.classes),
                             pretrained=args.pretrained).to(device)
    config_path = output.with_suffix('.config.json')
    write_json(config_path, config)
    print(f'Device: {device}; source: {args.source}; magnification: {args.magnification}; experiment: {args.experiment_id}', flush=True)
    print('Images:', {split: len(loader.dataset) for split, loader in loaders.items()}, flush=True)
    if args.command == 'evaluate':
        accuracy, loss = evaluate(model, loaders['test'], criterion)
        report = dict(split='test', accuracy=accuracy, accuracy_unit='percent', loss=loss,
                      experiment_id=args.experiment_id, checkpoint=str(input_path), **metadata)
        write_json(output, report)
        print(json.dumps(report, indent=2))
        print(f'Report: {output}; configuration: {config_path}')
        return
    if args.command == 'prune':
        from torch.nn.utils import prune
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, 'weight', amount=args.sparsity)
        accuracy, loss = evaluate(model, loaders['validation'], criterion)
        save_checkpoint(output, model, architecture, metadata, classes=loaders['train'].dataset.classes, config=config)
        print(f'Validation accuracy after pruning: {accuracy:.4f}%; saved {output}')
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_accuracy, first_epoch = float('-inf'), 1
    if resume:
        first_epoch, best_accuracy = restore_training_state(payload, optimizer, scheduler, config)
        print(f'Restored optimizer, scheduler and RNG; next epoch: {first_epoch}', flush=True)
    latest = output.with_name(output.stem + '-last.pth')
    def save_training(path, epoch):
        state = training_state(optimizer, scheduler, best_accuracy, config)
        save_checkpoint(path, model, architecture, metadata, epoch,
                        classes=loaders['train'].dataset.classes, config=config, state=state)
    # Epoch zero is a valid restart boundary if the first epoch is interrupted.
    save_training(latest, first_epoch - 1)
    completed_epoch = first_epoch - 1
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            started = time.perf_counter()
            loss = train(model, loaders['train'], criterion, optimizer, epoch=epoch)
            training_seconds = time.perf_counter() - started
            started = time.perf_counter()
            accuracy, val_loss = evaluate(model, loaders['validation'], criterion)
            print(f'Epoch {epoch}: train {training_seconds:.1f}s; validation {time.perf_counter()-started:.1f}s; loss {loss:.4f}; validation loss {val_loss:.4f}; accuracy {accuracy:.4f}%', flush=True)
            improved = accuracy > best_accuracy
            if improved:
                best_accuracy = accuracy
            scheduler.step()
            if improved:
                save_training(output, epoch)
            save_training(latest, epoch)
            completed_epoch = epoch
    except KeyboardInterrupt:
        interrupted = output.with_name(output.stem + '-interrupted.pth')
        save_checkpoint(interrupted, model, architecture, metadata, completed_epoch,
                        classes=loaders['train'].dataset.classes, config=config)
        print(f'Interrupted partial-epoch weights: {interrupted}; resume completed epochs from {latest}', flush=True)
        return
    print(f'Best validation accuracy: {best_accuracy:.4f}%; checkpoint: {output}; resume checkpoint: {latest}', flush=True)


if __name__ == '__main__':
    main()
