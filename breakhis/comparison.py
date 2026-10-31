"""A fixed-protocol dense/pruned/fine-tuned checkpoint comparison."""
import gc
import hashlib
from pathlib import Path
import uuid
from datetime import datetime, timezone

import torch


def compare_deltas(baseline, candidate):
    original_ms = baseline['benchmark']['latency_ms']['median']
    candidate_ms = candidate['benchmark']['latency_ms']['median']
    def ratio(key):
        original = baseline['size'][key]
        return candidate['size'][key] / original if original else None
    original_macs = baseline['benchmark']['conv_linear_macs_per_image']
    return dict(accuracy_delta_percentage_points=candidate['accuracy'] - baseline['accuracy'],
                logical_parameter_ratio=ratio('logical_parameters'),
                actual_checkpoint_byte_ratio=ratio('checkpoint_file_bytes'),
                model_state_byte_ratio=ratio('model_state_serialized_bytes'),
                conv_linear_mac_ratio=candidate['benchmark']['conv_linear_macs_per_image'] / original_macs if original_macs else None,
                observed_median_speedup=original_ms / candidate_ms,
                median_latency_change_percent=100 * (candidate_ms / original_ms - 1),
                interpretation='A speedup below one is slower; timings include noise and do not establish statistical significance. Checkpoint bytes may include different optimizer/RNG payloads.')


def run_comparison(args, parser):
    from .cli import DATA_DEFAULTS, resolve_input, write_json
    from .paths import checkpoint_output, validate_experiment_id
    from .data import read_manifest, get_split_loaders
    from .checkpoints import read_checkpoint, load_checkpoint
    from .reporting import checkpoint_report
    try:
        for field, default in DATA_DEFAULTS.items():
            if getattr(args, field) is None: setattr(args, field, default)
        args.manifest = resolve_input(args.manifest, 'Manifest')
        if args.batch_size < 1 or args.num_workers < 0:
            raise ValueError('Batch size must be positive and workers nonnegative')
        paths = {role: resolve_input(getattr(args, role), f'{role} checkpoint')
                 for role in ('dense', 'pruned', 'finetuned')}
        if len(set(paths.values())) != 3:
            raise ValueError('Comparison requires three distinct checkpoint files')
        payloads = {role: read_checkpoint(path) for role, path in paths.items()}
        manifest = read_manifest(args.manifest)
        root = resolve_input(args.data_root or manifest['roots'][args.source], 'Selected data root', directory=True)
        metadata = dict(source=args.source, magnification=args.magnification,
                        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        preprocessing='resize-short-edge-256/crop224/ImageNet-normalization')
        baseline = payloads['dense']
        architecture = baseline['architecture']
        for role, payload in payloads.items():
            if any(payload['metadata'].get(key) != value for key, value in metadata.items()):
                raise ValueError(f'{role} checkpoint source, manifest, magnification or preprocessing differs')
            if payload['classes'] != ['benign', 'malignant']:
                raise ValueError(f'{role} checkpoint has incompatible class labels')
            if payload['architecture'].replace('resnet18', 'resnet') != architecture.replace('resnet18', 'resnet'):
                raise ValueError('Comparison checkpoints must use the same baseline architecture')
        def pruning_type(payload):
            return payload.get('pruning', {}).get('type', 'unstructured' if any(key.endswith('.weight_mask') for key in payload['state_dict']) else 'dense')
        if pruning_type(baseline) != 'dense' or pruning_type(payloads['pruned']) == 'dense':
            raise ValueError('Provide a dense baseline and a pruned candidate in their named roles')
        before, after = payloads['pruned'], payloads['finetuned']
        if pruning_type(before) != pruning_type(after):
            raise ValueError('Pruned and fine-tuned checkpoints must use the same pruning method')
        if before.get('architecture_config', {'classes': len(before['classes'])}) != after.get('architecture_config', {'classes': len(after['classes'])}):
            raise ValueError('Pruned and fine-tuned architecture configurations differ')
        masks_before = {key for key in before['state_dict'] if key.endswith('.weight_mask')}
        masks_after = {key for key in after['state_dict'] if key.endswith('.weight_mask')}
        if masks_before != masks_after or any(not torch.equal(before['state_dict'][key], after['state_dict'][key]) for key in masks_before):
            raise ValueError('Pruned and fine-tuned masks differ')
        if pruning_type(before) == 'channel' and before['pruning'] != after['pruning']:
            raise ValueError('Pruned and fine-tuned channel selections differ')
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                              'cpu' if args.device == 'auto' else args.device)
        args.experiment_id = validate_experiment_id(args.experiment_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        output = args.output.expanduser().resolve() if args.output else checkpoint_output(
            args.source, args.magnification, architecture, 'compare', experiment_id=args.experiment_id).with_suffix('.json')
        if output.suffix.lower() != '.json' or output in paths.values():
            raise ValueError('Comparison output must be a separate .json file')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    loaders = get_split_loaders(args.manifest, args.source, args.magnification, root,
                               args.batch_size, args.num_workers, device, deterministic=args.deterministic)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(manifest=str(args.manifest), data_root=str(root), output=str(output),
                  resolved_device=str(device), **{role: str(path) for role, path in paths.items()})
    reports = {}
    for role, path in paths.items():
        print(f'Comparing {role}: {path}', flush=True)
        model, _ = load_checkpoint(path, device, payload=payloads[role])
        reports[role] = checkpoint_report(model, loaders['test'], path,
            batch_size=args.benchmark_batch_size, warmup=args.benchmark_warmup, repeats=args.benchmark_repeats)
        reports[role]['architecture'] = payloads[role]['architecture_config'] if 'architecture_config' in payloads[role] else {'classes': 2}
        reports[role]['pruning'] = payloads[role].get('pruning', {'type': pruning_type(payloads[role])})
        model.cpu()
        del model
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    report = dict(schema_version=1, split='test', experiment_id=args.experiment_id, **metadata,
                  config=config, models=reports,
                  relative_to_dense={role: compare_deltas(reports['dense'], reports[role]) for role in ('pruned', 'finetuned')},
                  limitations=['Roles are supplied by the operator; masks/topology/provenance are verified, but training history is not independently established.',
                               'Patient metrics use filename slide IDs, not independently verified patient identity.',
                               'MAC counts cover Conv2d/Linear only; unstructured zeros do not skip dense compute.',
                               'Benchmark excludes loading/preprocessing and is hardware/runtime/protocol specific.',
                               'Negative efficiency changes are retained; no speedup or full-dataset quality is assumed.'])
    write_json(output.with_suffix('.config.json'), config)
    write_json(output, report)
    print(f'Comparison report: {output}', flush=True)
    for role, result in report['relative_to_dense'].items():
        print(f'{role}: accuracy delta {result["accuracy_delta_percentage_points"]:+.3f} pp; observed median speedup {result["observed_median_speedup"]:.3f}x', flush=True)
