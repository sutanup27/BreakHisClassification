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
    from .experiment_io import write_json as save
    return save(path, payload)


def report_paths(output, predictions=None, *, protected=()):
    """Validate report/sidecar destinations before loading pixels or writing."""
    from .paths import output_path
    candidates = dict(report=Path(output), config=Path(output).with_suffix('.config.json'))
    if predictions is not None:
        candidates['predictions'] = Path(predictions)
    resolved = {}
    for name,path in candidates.items():
        if path.suffix.lower() != '.json':
            raise ValueError('Report and prediction destinations must end in .json')
        resolved[name] = output_path(path,root=path.absolute().parent,protected=protected)
    if len(set(resolved.values())) != len(resolved):
        raise ValueError('Report, predictions and config outputs must be distinct')
    return resolved


def add_benchmark_options(parser):
    parser.add_argument('--benchmark-batch-size', type=int, default=1, help='Resident forward benchmark batch size (separate from evaluation batches)')
    parser.add_argument('--benchmark-warmup', type=int, default=10)
    parser.add_argument('--benchmark-repeats', type=int, default=30)
    parser.add_argument('--cpu-threads', type=int, help='Fix PyTorch intra-op CPU threads for repeatable comparisons')


def main(argv=None):
    parser = argparse.ArgumentParser(description='BreaKHis classification, pruning and dataset auditing')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-data', help='Audit sources and save patient-separated splits')
    prepare.add_argument('--dataset-root', type=Path, default=ROOT.parent/'Dataset')
    prepare.add_argument('--output', type=Path, default=ROOT/'artifacts/data')
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--workers', type=int, default=8)
    prepare.add_argument('--sources', nargs='+', choices=['breakhis_v2', 'cancer_binary', 'cancer_multiclass'], help='Audit only these sources; otherwise discover available configured folders')
    for name in ('train', 'evaluate', 'prune', 'finetune'):
        command = commands.add_parser(name)
        add_data_options(command)
        command.add_argument('--output', type=Path, help='Checkpoint filename/path inside checkpoint/binary, or evaluation JSON path')
        if name == 'evaluate':
            add_benchmark_options(command)
        if name in ('train', 'finetune'):
            command.add_argument('--seed', type=int, help='Fresh-run RNG seed; inherited on resume')
            command.add_argument('--optimizer', choices=['sgd','adam'])
            command.add_argument('--weight-decay', type=float)
            command.add_argument('--scheduler', choices=['cosine','none'])
            command.add_argument('--selection-metric', choices=['accuracy','group_balanced_accuracy'])
            command.add_argument('--epochs', type=int, help='Total planned epochs; default 20 or saved resume value')
            command.add_argument('--learning-rate', type=float, help='Initial SGD rate; default .001 or saved resume value')
            command.add_argument('--resume', type=Path, help='Restore a completed-epoch checkpoint and its optimizer/scheduler/RNG state')
        if name == 'train':
            command.add_argument('--model', choices=['vgg', 'resnet', 'resnet18'])
            command.add_argument('--pretrained', action='store_true', help='Initialize ResNet with downloaded pretrained weights')
        else:
            command.add_argument('--checkpoint', type=Path, required=name != 'finetune')
        if name == 'evaluate':
            command.add_argument('--predictions-output', type=Path, help='Save ordered per-image predictions as JSON')
        if name == 'prune':
            command.add_argument('--sparsity', type=float, default=.5)
            command.add_argument('--method', choices=['unstructured', 'channel'], default='unstructured')
            command.add_argument('--layer-sparsities', type=Path, help='JSON object specifying every eligible layer name and ratio')
    comparison = commands.add_parser('compare', help='Compare dense, pruned and fine-tuned checkpoints on the reserved test split')
    add_data_options(comparison)
    add_benchmark_options(comparison)
    comparison.add_argument('--dense', type=Path, required=True)
    comparison.add_argument('--pruned', type=Path, required=True)
    comparison.add_argument('--finetuned', type=Path, required=True)
    comparison.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('evaluate', 'compare'):
        if args.benchmark_batch_size < 1 or args.benchmark_warmup < 0 or args.benchmark_repeats < 1 or (args.cpu_threads is not None and args.cpu_threads < 1):
            parser.error('Benchmark batch size, repeats and CPU threads must be positive; warmup nonnegative')
        if args.cpu_threads is not None:
            import torch
            torch.set_num_threads(args.cpu_threads)
    if args.command == 'compare':
        from .comparison import run_comparison
        return run_comparison(args, parser)
    if args.command == 'prepare-data':
        if args.workers < 1:
            parser.error('workers must be positive')
        from .manifest import prepare_dataset
        try:
            root = resolve_input(args.dataset_root, 'Dataset root', directory=True)
            prepare_dataset(root, args.output.expanduser().resolve(), args.seed, args.workers, args.sources)
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
            args.seed = args.seed if args.seed is not None else previous.get('seed', 0)
            args.optimizer = args.optimizer or previous.get('optimizer_name', 'sgd')
            args.weight_decay = args.weight_decay if args.weight_decay is not None else previous.get('weight_decay', 5e-4)
            args.scheduler = args.scheduler or previous.get('scheduler_name', 'cosine')
            args.selection_metric = args.selection_metric or previous.get('selection_metric', 'accuracy')
            if not 0 <= args.seed < 2**32 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
                raise ValueError('Seed must be uint32 and weight decay finite/nonnegative')
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
        if args.command == 'prune':
            from .pruning import validate_sparsity
            validate_sparsity(args.sparsity, channel=args.method == 'channel')
            if args.layer_sparsities:
                args.layer_sparsities = resolve_input(args.layer_sparsities, 'Layer sparsities')
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
        if args.command == 'prune' and args.method == 'channel' and architecture != 'vgg':
            raise ValueError('Channel pruning supports VGG only; residual/depthwise architectures are unsupported')
        if args.command == 'evaluate':
            output = args.output.absolute() if args.output else checkpoint_output(
                args.source, args.magnification, architecture, args.command,
                experiment_id=args.experiment_id).with_suffix('.json')
            if output.suffix.lower() != '.json':
                raise ValueError('Evaluation report output must end in .json')
            destinations = report_paths(output,args.predictions_output,protected=[args.manifest,input_path])
            output = destinations['report']
            if args.predictions_output is not None:
                args.predictions_output = destinations['predictions']
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
            config.update(optimizer_name=args.optimizer, scheduler_name=args.scheduler,
                optimizer={'name': args.optimizer, 'momentum': .9 if args.optimizer == 'sgd' else None, 'weight_decay': args.weight_decay},
                scheduler={'name': args.scheduler, 'T_max': args.epochs})
        if resume:
            import platform
            if payload.get('runtime') != {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda}:
                raise ValueError('Exact resume requires the saved Python/PyTorch/CUDA runtime versions')
            for field in ('epochs', 'learning_rate', 'batch_size', 'num_workers', 'deterministic', 'resolved_device', 'output'):
                if config[field] != previous[field]:
                    raise ValueError(f'Resume configuration differs: {field} must be {previous[field]!r}')
            for field in ('seed','optimizer_name','weight_decay','scheduler_name','selection_metric'):
                if field in previous and config[field] != previous[field]:
                    raise ValueError(f'Resume configuration differs: {field}')
            if payload['epoch'] >= args.epochs:
                raise ValueError('Resume checkpoint has already completed all planned epochs')
            if payload['epoch'] > 0 and not output.is_file():
                raise ValueError('Best checkpoint is missing; restore the experiment directory before resume')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))

    loaders = get_split_loaders(args.manifest, args.source, args.magnification, selected_root,
                               args.batch_size, args.num_workers, device, seed=getattr(args, 'seed', 0), deterministic=args.deterministic)
    criterion = nn.CrossEntropyLoss().to(device)
    if input_path:
        model, payload = load_checkpoint(input_path, device, payload=payload)
    else:
        model = create_model(architecture, classes=len(loaders['train'].dataset.classes),
                             pretrained=args.pretrained).to(device)
    config_path = output.with_suffix('.config.json')
    if args.command != 'prune':
        write_json(config_path, config)
    print(f'Device: {device}; source: {args.source}; magnification: {args.magnification}; experiment: {args.experiment_id}', flush=True)
    print('Images:', {split: len(loader.dataset) for split, loader in loaders.items()}, flush=True)
    if args.command == 'evaluate':
        from .reporting import checkpoint_report
        predictions = []
        report = checkpoint_report(model, loaders['test'], input_path,
            batch_size=args.benchmark_batch_size, warmup=args.benchmark_warmup, repeats=args.benchmark_repeats,
            prediction_rows=predictions)
        report.update(schema_version=1, split='test', experiment_id=args.experiment_id, **metadata)
        write_json(output, report)
        if args.predictions_output:
            write_json(args.predictions_output, dict(schema_version=1, split='test', classes=payload['classes'],
                checkpoint_sha256=report['checkpoint_sha256'], manifest_sha256=metadata['manifest_sha256'], rows=predictions))
        print(json.dumps(report, indent=2))
        print(f'Report: {output}; configuration: {config_path}')
        return
    if args.command == 'prune':
        from .pruning import prune_weights, prune_vgg_channels
        try:
            settings = json.loads(args.layer_sparsities.read_text(encoding='utf-8-sig')) if args.layer_sparsities else None
            model = (prune_vgg_channels if args.method == 'channel' else prune_weights)(model, args.sparsity, settings)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        config['layer_sparsities_resolved'] = settings
        write_json(config_path, config)
        accuracy, loss = evaluate(model, loaders['validation'], criterion)
        write_json(output.with_suffix('.validation.json'), dict(split='validation', accuracy=accuracy,
                   accuracy_unit='percent', loss=loss, method=args.method, sparsity=args.sparsity,
                   layer_sparsities=settings, experiment_id=args.experiment_id, checkpoint=str(input_path),
                   source=args.source, magnification=args.magnification))
        save_checkpoint(output, model, architecture, metadata, classes=loaders['train'].dataset.classes, config=config)
        print(f'Validation accuracy after pruning: {accuracy:.4f}%; saved {output}')
        return
    optimizer = (torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9, weight_decay=args.weight_decay)
                 if args.optimizer == 'sgd' else torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.scheduler == 'cosine'
                 else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.0))
    best_accuracy, first_epoch = float('-inf'), 1
    if resume:
        first_epoch, best_accuracy = restore_training_state(payload, optimizer, scheduler, config)
        print(f'Restored optimizer, scheduler and RNG; next epoch: {first_epoch}', flush=True)
    latest = output.with_name(output.stem + '-last.pth')
    history_path = output.with_suffix('.history.json')
    history = json.loads(history_path.read_text(encoding='utf-8'))['epochs'] if resume and history_path.exists() else []
    history = [row for row in history if row['epoch'] < first_epoch]
    best_val_loss = min((row['validation_loss'] for row in history
                        if row['selection_score_percent'] == best_accuracy), default=float('inf'))
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
            if args.selection_metric == 'group_balanced_accuracy':
                from .reporting import collect_predictions, magnification_balanced_metrics
                predictions = []
                image_metrics, _, _ = collect_predictions(model, loaders['validation'], prediction_rows=predictions)
                group_metrics = magnification_balanced_metrics(loaders['validation'].dataset.rows,
                    [v['target'] for v in predictions], [v['probabilities'] for v in predictions], loaders['validation'].dataset.classes)
                if not group_metrics['available']:
                    raise ValueError('Group selection requires magnification metadata')
                accuracy, val_loss = image_metrics['accuracy'], image_metrics['loss']
                selection_score = 100 * group_metrics['balanced_accuracy']
            else:
                accuracy, val_loss = evaluate(model, loaders['validation'], criterion)
                selection_score = accuracy
            print(f'Epoch {epoch}: train {training_seconds:.1f}s; validation {time.perf_counter()-started:.1f}s; loss {loss:.4f}; validation loss {val_loss:.4f}; accuracy {accuracy:.4f}%', flush=True)
            improved = selection_score > best_accuracy or (args.selection_metric == 'group_balanced_accuracy'
                and selection_score == best_accuracy and val_loss < best_val_loss)
            if improved:
                best_accuracy = selection_score
                best_val_loss = val_loss
            scheduler.step()
            if improved:
                save_training(output, epoch)
            history.append(dict(epoch=epoch, training_loss=loss, validation_loss=val_loss,
                validation_accuracy_percent=accuracy, selection_metric=args.selection_metric,
                selection_score_percent=selection_score, training_seconds=training_seconds,
                learning_rate=optimizer.param_groups[0]['lr']))
            write_json(history_path, dict(schema_version=1, epochs=history, selection_split='validation'))
            save_training(latest, epoch)
            completed_epoch = epoch
    except KeyboardInterrupt:
        interrupted = output.with_name(output.stem + '-interrupted.pth')
        save_checkpoint(interrupted, model, architecture, metadata, completed_epoch,
                        classes=loaders['train'].dataset.classes, config=config)
        print(f'Interrupted partial-epoch weights: {interrupted}; resume completed epochs from {latest}', flush=True)
        return
    print(f'Best validation {args.selection_metric}: {best_accuracy:.4f}%; checkpoint: {output}; resume checkpoint: {latest}', flush=True)


if __name__ == '__main__':
    main()
