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
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON field: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def write_json(path, value):
    from .cli import write_json as save
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


def validate_data(config):
    from .data import read_manifest
    payload = read_manifest(config['data']['manifest'])
    data = config['data']
    root = Path(data['data_root'] or payload['roots'][data['source']]).resolve()
    if not root.is_dir(): raise ValueError('Data root does not exist: '+str(root))
    data['data_root'] = str(root)
    rows = [row for row in payload['rows'] if row['source'] == data['source'] and
        (data['magnification'] == 'all' or row['magnification'] == data['magnification'])]
    counts = {}
    paths = set()
    for row in rows:
        candidate = (root/row['path']).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file(): raise ValueError('Unsafe or missing image: '+row['path'])
        if row['path'].casefold() in paths: raise ValueError('Duplicate image path')
        paths.add(row['path'].casefold())
    for split in ('train','validation','test'):
        subset = [row for row in rows if row['split'] == split]
        if {row['label'] for row in subset} != {'benign','malignant'}:
            raise ValueError('Both classes are required in selected '+split+' split')
        counts[split] = dict(images=len(subset), filename_groups=len({row['patient_id'] for row in subset}))
    # Detect transitive grouping as well as direct filename/pixel leakage.
    memberships = {}
    labels = {}
    raw_splits = {}
    for row in rows:
        key = row.get('group_id', row['patient_id'])
        memberships.setdefault(key, set()).add(row['split'])
        labels.setdefault(key, set()).add(row['label'])
        raw_splits.setdefault(row['sha256'],set()).add(row['split'])
    if any(len(values) != 1 for values in memberships.values()): raise ValueError('Linked group crosses splits')
    if any(len(values) != 1 for values in labels.values()): raise ValueError('Conflicting labels in linked group')
    if any(len(values) != 1 for values in raw_splits.values()): raise ValueError('Duplicate raw image crosses splits')
    return payload, counts


def revision():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source = {str(path.relative_to(ROOT)).replace('\\','/'):digest(path)
              for path in sorted((ROOT/'breakhis').rglob('*.py'))}
    return dict(commit=git('rev-parse','HEAD'), dirty=bool(git('status','--porcelain')),
        source_sha256=source, note='Source hashes and dirty state recorded; no approval/archive required')


def run_experiment(config_file, *, run_id=None, resume=False, seed=None, device=None):
    from .paths import validate_experiment_id, checkpoint_output
    config = load_config(config_file, seed=seed, device=device)
    payload, counts = validate_data(config)
    import torch
    training = config['training']
    resolved_device = 'cuda' if training['device'] == 'auto' and torch.cuda.is_available() else 'cpu' if training['device'] == 'auto' else training['device']
    if 'cuda' in (resolved_device, config['evaluation']['device']) and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if resume and run_id is None: raise ValueError('Resume requires the original --run-id')
    run_id = validate_experiment_id(run_id or config['name'][:55]+'-'+uuid.uuid4().hex[:8])
    run = RUNS/run_id
    checkpoint = checkpoint_output(config['data']['source'], config['data']['magnification'],training['model'],'train',experiment_id=run_id)
    identity = dict(manifest_sha256=digest(config['data']['manifest']), preprocessing=PREPROCESSING,
        resolved_device=resolved_device, runtime=dict(python=sys.version.split()[0],torch=str(torch.__version__),cuda=torch.version.cuda))
    if resume:
        previous = read_json(run/'config.json')
        if previous != config or read_json(run/'inputs.json')['identity'] != identity:
            raise ValueError('Resume requires identical config, manifest and runtime/device; use a new run ID for changes')
        if digest(run/'manifest.json') != identity['manifest_sha256']:
            raise ValueError('Frozen run manifest changed')
        if read_json(run/'inputs.json')['code']['source_sha256'] != revision()['source_sha256']:
            raise ValueError('Implementation changed since run creation; use a new run ID')
        state = read_json(run/'run.json')
        if state['status'] == 'completed': raise ValueError('Run already completed; use a new ID for a fresh rerun')
    else:
        if run.exists() or checkpoint.parent.exists(): raise ValueError('Run/checkpoint directory exists; choose a new ID or explicit --resume')
        run.mkdir(parents=True)
        write_json(run/'config.json', config)
        manifest_bytes = Path(config['data']['manifest']).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != identity['manifest_sha256']:
            raise ValueError('Manifest changed during setup')
        (run/'manifest.json').write_bytes(manifest_bytes)
        write_json(run/'inputs.json',dict(identity=identity, counts=counts, code=revision(), config_source=str(Path(config_file).resolve())))
        state = dict(schema_version=1, run_id=run_id,status='running',started_at=datetime.now(timezone.utc).isoformat(),stages={})
    state['status']='running'
    write_json(run/'run.json', state)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    # Use the frozen manifest, not a mutable shared file, for every stage.
    data = config['data']
    common = ['--dataset','breakhis','--manifest',str(run/'manifest.json'),'--source',data['source'],
        '--magnification',data['magnification'],'--data-root',data['data_root'],'--batch-size',str(training['batch_size']),
        '--num-workers','0','--device',resolved_device,'--experiment-id',run_id,
        '--deterministic' if training['deterministic'] else '--no-deterministic']
    env = dict(os.environ, OMP_NUM_THREADS=str(training['cpu_threads']), MKL_NUM_THREADS=str(training['cpu_threads']), PYTHONUTF8='1')

    def execute(stage, args, *, output, epochs=None):
        from .checkpoints import read_checkpoint
        record = state['stages'].get(stage, {})
        if record.get('status') == 'completed':
            if not output.is_file() or digest(output) != record['output_sha256']:
                raise ValueError('Completed stage output changed or missing: '+stage)
            return
        last = output.with_name(output.stem+'-last.pth')
        if epochs is not None and last.is_file():
            boundary = read_checkpoint(last)
            if boundary['epoch'] >= epochs and output.is_file():
                state['stages'][stage] = dict(status='completed',output=str(output),output_sha256=digest(output),recovered_completed_boundary=True)
                write_json(run/'run.json',state)
                return
            args = [args[0], *common, '--resume',str(last)]
        log = run/'logs'/f'{stage}.log'
        log.parent.mkdir(exist_ok=True)
        argv = [sys.executable,'-m','breakhis',*map(str,args)]
        record = dict(status='running',argv=argv,log=str(log),started_at=datetime.now(timezone.utc).isoformat())
        state['stages'][stage]=record
        write_json(run/'run.json',state)
        print('Running '+stage+'; log: '+str(log),flush=True)
        started=time.perf_counter()
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(return_code=result.returncode,seconds=time.perf_counter()-started)
        if result.returncode != 0: raise RuntimeError(stage+' failed; see '+str(log))
        if epochs is not None and (not last.is_file() or read_checkpoint(last)['epoch'] < epochs):
            raise InterruptedError(stage+' stopped before planned epochs; resume this run explicitly')
        if not output.is_file(): raise RuntimeError(stage+' produced no expected output')
        record.update(status='completed',output=str(output),output_sha256=digest(output))
        write_json(run/'run.json',state)

    fit = ['--learning-rate',str(training['learning_rate']),'--optimizer',training['optimizer'],
        '--weight-decay',str(training['weight_decay']),'--scheduler',training['scheduler'],
        '--seed',str(training['seed']),'--selection-metric',training['selection_metric']]
    try:
        execute('train',['train',*common,*fit,'--model',training['model'],'--epochs',str(training['epochs']),'--output',str(checkpoint)],
            output=checkpoint,epochs=training['epochs'])
        models={'dense':checkpoint}
        if config['pruning']:
            pruning=config['pruning']; pruned=checkpoint.parent/'prune.pth'; finetuned=checkpoint.parent/'finetune.pth'
            execute('prune',['prune',*common,'--checkpoint',str(checkpoint),'--method',pruning['method'],'--sparsity',str(pruning['sparsity']),'--output',str(pruned)],output=pruned)
            execute('finetune',['finetune',*common,*fit,'--checkpoint',str(pruned),'--epochs',str(pruning['finetune_epochs']),'--output',str(finetuned)],output=finetuned,epochs=pruning['finetune_epochs'])
            models.update(pruned=pruned,finetuned=finetuned)
        # All development choices are fixed before this final test phase.
        evaluation=config['evaluation']
        env.update(OMP_NUM_THREADS=str(evaluation['cpu_threads']),MKL_NUM_THREADS=str(evaluation['cpu_threads']))
        reports={}
        for role,path in models.items():
            output=run/f'{role}.metrics.json'
            args=['evaluate',*common,'--device',evaluation['device'],'--checkpoint',str(path),'--output',str(output),
                '--predictions-output',str(run/f'{role}.predictions.json'),'--benchmark-warmup',str(evaluation['warmup']),
                '--benchmark-repeats',str(evaluation['repeats']),'--benchmark-batch-size',str(evaluation['batch_size']),
                '--cpu-threads',str(evaluation['cpu_threads'])]
            execute('evaluate-'+role,args,output=output)
            reports[role]=str(output)
        for command in ('train','finetune'):
            history=checkpoint.parent/f'{command}.history.json'
            if history.is_file(): write_json(run/f'{command}.history.json',read_json(history))
        final='finetuned' if config['pruning'] else 'dense'
        write_json(run/'results.json',dict(schema_version=1,run_id=run_id,config=config,identity=identity,reports=reports,
            final_role=final,final_metrics=read_json(reports[final]),
            limitations=['Single held-out split, not nested cross-validation.',
                'Fixed argmax operating point; no separate threshold calibration or constrained search.',
                'Filename groups are not independently verified patients; prior dataset use acknowledged.',
                'Timing is one session on the recorded evaluation device; replicate before drawing speedup conclusions.']))
        state.update(status='completed',ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',error=str(error))
        for record in state['stages'].values():
            if record['status']=='running': record['status']=state['status']
        write_json(run/'run.json',state)
        raise
    write_json(run/'run.json',state)
    print('Completed experiment: '+str(run),flush=True)
    return run


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
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true',help='Continue an interrupted run at completed-epoch/stage boundaries')
    parser.add_argument('--seed',type=int,help='Override and record the training seed; split seed remains in the manifest')
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    args=parser.parse_args(argv)
    try: run_experiment(args.config,run_id=args.run_id,resume=args.resume,seed=args.seed,device=args.device)
    except (ValueError,OSError,RuntimeError,InterruptedError) as error: parser.exit(1,str(error)+'\n')
    except KeyboardInterrupt: parser.exit(130,'Interrupted; use the same config/run ID with --resume.\n')


def comparison_main(argv=None):
    parser=argparse.ArgumentParser(description='Export comparable completed runs as JSON and CSV')
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: print(compare_runs(args.runs,args.output))
    except (ValueError,OSError,KeyError) as error: parser.exit(1,str(error)+'\n')
