"""Small config-driven adapter around the existing train/prune/evaluate commands.

No research approval imports. Every run keeps its inputs, commands and results.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'artifacts/experiments'
PREPROCESSING = 'resize-short-edge-256/crop224/ImageNet-normalization'


def read_json(path):
    from .experiment_io import read_json as read
    return read(path)


def write_json(path, value):
    from .experiment_io import write_json as save
    save(Path(path), value)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('Invalid '+name+' fields; expected '+', '.join(sorted(fields)))


def integer(value, minimum, name):
    if type(value) is not int or value < minimum: raise ValueError('Invalid '+name)


def load_config(path, *, seed=None, device=None):
    path = Path(path).resolve()
    config = read_json(path)
    exact(config, {'schema_version','name','data','training','pruning','evaluation'}, 'experiment')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('Unsupported experiment schema')
    from .paths import validate_experiment_id
    validate_experiment_id(config['name'])
    exact(config['data'], {'manifest','source','magnification','data_root'}, 'data')
    data = config['data']
    if data['source'] != 'cancer_binary' or data['magnification'] not in ('40X','100X','200X','400X','all'):
        raise ValueError('Experiment workflow supports the v1 binary source and known magnifications')
    for key in ('manifest','data_root'):
        value = data[key]
        if value is None and key == 'data_root': continue
        if not isinstance(value, str) or not value.strip(): raise ValueError('Invalid '+key)
        location = Path(value).expanduser()
        data[key] = str((location if location.is_absolute() else path.parent/location).resolve())
    training = config['training']
    exact(training, {'model','epochs','batch_size','num_workers','device','seed','deterministic',
        'learning_rate','optimizer','weight_decay','scheduler','selection_metric','cpu_threads'}, 'training')
    if seed is not None: training['seed'] = seed
    if device is not None: training['device'] = device
    for field in ('epochs','batch_size','cpu_threads'): integer(training[field], 1, field)
    if training['num_workers'] != 0 or type(training['num_workers']) is not int:
        raise ValueError('Use workers zero for reproducible epoch-boundary resume')
    integer(training['seed'], 0, 'seed')
    if training['seed'] >= 2**32: raise ValueError('Seed must be uint32')
    if type(training['deterministic']) is not bool: raise ValueError('Invalid deterministic flag')
    if training['model'] not in ('vgg','resnet','resnet18') or training['device'] not in ('auto','cpu','cuda'):
        raise ValueError('Unsupported model/device')
    if training['optimizer'] not in ('sgd','adam') or training['scheduler'] not in ('cosine','none'):
        raise ValueError('Unsupported optimizer/scheduler')
    if training['selection_metric'] not in ('accuracy','group_balanced_accuracy'): raise ValueError('Unknown selection metric')
    for field in ('learning_rate','weight_decay'):
        value = training[field]
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or (field == 'learning_rate' and value == 0):
            raise ValueError('Invalid '+field)
    pruning = config['pruning']
    if pruning is not None:
        exact(pruning, {'method','sparsity','finetune_epochs'}, 'pruning')
        if pruning['method'] not in ('channel','unstructured'): raise ValueError('Unsupported pruning method')
        if pruning['method'] == 'channel' and training['model'] != 'vgg': raise ValueError('Channel pruning supports VGG only')
        value = pruning['sparsity']
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Pruning sparsity must be strictly between zero and one')
        integer(pruning['finetune_epochs'], 1, 'finetune epochs')
    exact(config['evaluation'], {'warmup','repeats','batch_size','device','cpu_threads'}, 'evaluation')
    evaluation = config['evaluation']
    for field in ('repeats','batch_size','cpu_threads'): integer(evaluation[field], 1, field)
    integer(evaluation['warmup'], 0, 'warmup')
    if evaluation['device'] not in ('cpu','cuda'): raise ValueError('Choose a fixed evaluation device for comparisons')
    # Validate serializability/finite nested values before creating a run.
    json.dumps(config, allow_nan=False)
    return config


def validate_data(config, *, splits=('train','validation','test')):
    from .data import read_manifest, manifest_selection, validate_role_rows
    from .manifest import requested_splits
    splits=requested_splits(splits)
    payload = read_manifest(config['data']['manifest'])
    data = config['data']
    rows = manifest_selection(payload,data['source'],data['magnification'])
    root = Path(data['data_root'] or payload['roots'][data['source']]).resolve()
    if not root.is_dir(): raise ValueError('Data root does not exist: '+str(root))
    data['data_root'] = str(root)
    counts = {}
    for split in ('train','validation','test'):
        subset = [row for row in rows if row['split'] == split]
        if split in splits: validate_role_rows(subset,root,data['source'],data['magnification'],split)
        counts[split] = dict(images=len(subset), filename_groups=len({row['patient_id'] for row in subset}))
    return payload, counts


def revision():
    def git(*args):
        try:
            result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    source = {str(path.relative_to(ROOT)).replace('\\','/'):digest(path)
              for path in sorted((ROOT/'breakhis').rglob('*.py'))}
    commit, status = git('rev-parse','HEAD'), git('status','--porcelain')
    return dict(commit=commit, dirty=None if status is None else bool(status), git_available=commit is not None,
        source_sha256=source, note='Source hashes and dirty state recorded; no approval/archive required')


def run_experiment(config_file, *, run_id=None, resume=False, seed=None, device=None):
    from .development import run_development
    return run_development(config_file,run_id=run_id,resume=resume,seed=seed,device=device)


def compare_runs(runs, output):
    results=[]
    for value in runs:
        directory=Path(value) if Path(value).is_dir() else RUNS/value
        state=read_json(directory/'run.json')
        if state['status']!='completed': raise ValueError('Only completed runs can be compared')
        for stage,record in state['stages'].items():
            if stage.startswith('evaluate-') and digest(record['output'])!=record['output_sha256']:
                raise ValueError('Evaluation report changed: '+stage)
        results.append(read_json(directory/'results.json'))
    if len(results)<2: raise ValueError('Compare at least two runs')
    first=results[0]
    def protocol(result):
        config=result['config']
        return dict(manifest=result['identity']['manifest_sha256'],preprocessing=result['identity']['preprocessing'],
            source=config['data']['source'],magnification=config['data']['magnification'],
            training={key:value for key,value in config['training'].items() if key not in ('seed','epochs','model')},
            evaluation=config['evaluation'],runtime=result['identity']['runtime'],
            training_device=result['identity']['resolved_device'],
            benchmark_runtime=result['final_metrics']['benchmark']['runtime'])
    if any(protocol(result)!=protocol(first) for result in results[1:]):
        raise ValueError('Comparison requires identical split, preprocessing, optimizer/selection and evaluation/runtime settings')
    table=[]
    for result in results:
        for role,path in result['reports'].items():
            report=read_json(path)
            group=report['group_metrics']; image=report['image_metrics']
            if not group.get('available'): raise ValueError('Group metrics missing')
            table.append(dict(run_id=result['run_id'],role=role,
                model=result['config']['training']['model'],seed=result['config']['training']['seed'],
                method=(result['config']['pruning'] or {}).get('method','dense') if role!='dense' else 'dense',
                group_balanced_accuracy=group['balanced_accuracy'],group_sensitivity=group['sensitivity'],group_specificity=group['specificity'],
                image_accuracy_percent=image['accuracy'],parameters=report['size']['logical_parameters'],
                median_latency_ms=report['benchmark']['latency_ms']['median'],
                training_epochs=result['config']['training']['epochs'],
                extra_finetune_epochs=(result['config']['pruning'] or {}).get('finetune_epochs',0) if role=='finetuned' else 0))
    output=Path(output).resolve()
    if output.suffix!='.json' or output.exists() or output.with_suffix('.csv').exists():
        raise ValueError('Choose a new .json comparison output')
    write_json(output,dict(schema_version=1,comparison_protocol=protocol(first),rows=table,
        note='Descriptive held-out comparison, not model selection. Seeds/models and training budgets are listed explicitly. Seed repeats are not independent patient samples. Fine-tuned models receive extra epochs; include an equal-budget dense control for method claims.'))
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true',help='Continue an interrupted run at completed-epoch/stage boundaries')
    parser.add_argument('--seed',type=int,help='Override and record the training seed; split seed remains in the manifest')
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    parser.add_argument('--phase',choices=['development','final'],help='Development is the default; final execution is added in M1.4')
    parser.add_argument('--prepare-only',action='store_true',help='Validate and save schema-2 inputs without training or inference')
    args=parser.parse_args(argv)
    try:
        if args.prepare_only or args.phase == 'final':
            from .experiment_setup import prepare_run, validate_request
            phase=args.phase or 'development'
            validate_request(config_file=args.config,run_id=args.run_id,phase=phase,
                resume=args.resume,seed=args.seed,device=args.device)
            if not args.prepare_only:
                raise ValueError('Final execution is not implemented yet (M1.4); development execution is available')
            prepared=prepare_run(args.config,run_id=args.run_id,phase=phase,resume=args.resume,seed=args.seed,device=args.device)
            print('Prepared/validated inputs only; no training or inference: '+str(prepared['run']))
        else:
            if args.config is None:
                raise ValueError('Development requires --config')
            run_experiment(args.config,run_id=args.run_id,resume=args.resume,seed=args.seed,device=args.device)
    except (ValueError,OSError,RuntimeError,InterruptedError) as error: parser.exit(1,str(error)+'\n')
    except KeyboardInterrupt: parser.exit(130,'Interrupted; use the same config/run ID with --resume.\n')


def comparison_main(argv=None):
    parser=argparse.ArgumentParser(description='Export comparable completed runs as JSON and CSV')
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: print(compare_runs(args.runs,args.output))
    except (ValueError,OSError,KeyError) as error: parser.exit(1,str(error)+'\n')
