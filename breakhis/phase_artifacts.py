"""Read-only verification of phase outputs and their frozen selection bindings."""
from pathlib import Path
import math
import statistics
from . import experiments as e, experiment_setup as setup
from .reporting import CLASSES, AGGREGATION, THRESHOLD, BENCHMARK_PROTOCOL


def verify_stage(run,inputs,state,stage):
    roots=dict(run=run,checkpoint=Path(inputs['checkpoint_directory']))
    record=state['stages'].get(stage,{})
    if record.get('status')!='completed': raise ValueError('Required stage is not completed: '+stage)
    actual=[setup.verify_reference(r,roots=roots) for r in record.get('artifacts',[])]
    if set(actual)!={Path(p) for p in inputs['expected_artifacts'][stage]}:
        raise ValueError('Incomplete stage artifacts: '+stage)


def development(run):
    run=Path(run).resolve(); config,state,inputs=setup._read_prepared(run)
    if state['phases']['development']['status']!='completed':
        raise ValueError('Final/comparison requires completed development')
    roles=['dense','pruned','finetuned'] if config['pruning'] else ['dense']
    stages=['train',*(['prune','finetune'] if config['pruning'] else []),
        *['development-'+r for r in roles],'development-complete']
    for stage in stages: verify_stage(run,inputs,state,stage)
    selection=e.read_json(run/'development/selection.json')
    identity=inputs['identity']
    required=dict(schema_version=2,run_id=state['run_id'],config_sha256=identity['config_sha256'],
        manifest_sha256=identity['manifest_sha256'],preprocessing=identity['preprocessing'],
        aggregation=AGGREGATION,threshold=THRESHOLD,classes=CLASSES,
        class_to_idx={v:i for i,v in enumerate(CLASSES)},designated_role='finetuned' if config['pruning'] else 'dense')
    if any(selection.get(k)!=v for k,v in required.items()) or set(selection.get('roles',{}))!=set(roles):
        raise ValueError('Selection config/manifest/policy/role binding differs')
    roots=dict(run=run,checkpoint=Path(inputs['checkpoint_directory']))
    for role,binding in selection['roles'].items():
        command={'dense':'train','pruned':'prune','finetuned':'finetune'}[role]
        expected=dict(checkpoint=Path(inputs['checkpoint_directory'])/f'{command}.pth',
            report=run/'development'/f'{role}.metrics.json',predictions=run/'development'/f'{role}.predictions.json')
        for key,path in expected.items():
            if setup.verify_reference(binding[key],roots=roots)!=path:
                raise ValueError('Selection points to a different '+key)
        criterion='fixed_recipe' if role=='pruned' else config['training']['selection_metric']
        if binding.get('criterion')!=criterion or type(binding.get('epoch')) is not int:
            raise ValueError('Selection criterion/epoch differs')
        if role=='pruned':
            if binding['epoch']!=0: raise ValueError('Fixed pruning recipe must use epoch zero')
        else:
            history=e.read_json(run/f'{command}.history.json')['epochs']
            planned=config['training']['epochs'] if command=='train' else config['pruning']['finetune_epochs']
            if not history or len(history)!=planned or [r['epoch'] for r in history]!=list(range(1,planned+1)):
                raise ValueError('History coverage differs')
            chosen=max(history,key=lambda r:(r['selection_score_percent'],
                -r['validation_loss'] if criterion=='group_balanced_accuracy' else 0,-r['epoch']))
            if chosen['epoch']!=binding['epoch']: raise ValueError('Selection epoch is not the development choice')
        validate_report(run,config,inputs,selection,role,'development')
    return config,state,inputs,selection


def validate_report(run,config,inputs,selection,role,phase):
    """Check complete ordered predictions against metadata without opening pixels."""
    from .data import read_manifest, manifest_selection
    from .reporting import validate_probabilities, magnification_balanced_metrics, grouped_patient_metrics
    split='validation' if phase=='development' else 'test'
    report=e.read_json(run/phase/f'{role}.metrics.json')
    predictions=e.read_json(run/phase/f'{role}.predictions.json')
    fields=dict(schema_version=2,phase=phase,split=split,run_id=selection['run_id'],model_role=role,
        config_sha256=inputs['identity']['config_sha256'],manifest_sha256=inputs['identity']['manifest_sha256'],
        checkpoint_sha256=selection['roles'][role]['checkpoint']['sha256'],
        classes=CLASSES,class_to_idx={v:i for i,v in enumerate(CLASSES)},
        aggregation=AGGREGATION,threshold=THRESHOLD,threshold_probability=.5,tie_class='benign',
        source=config['data']['source'],magnification=config['data']['magnification'],
        preprocessing=inputs['identity']['preprocessing'])
    for document in (report,predictions):
        if any(document.get(k)!=v for k,v in fields.items()): raise ValueError('Report/prediction phase or identity differs')
        if document.get('evaluation_runtime')!=report['benchmark']['runtime']:
            raise ValueError('Evaluation runtime attribution differs')
    benchmark=report['benchmark']; evaluation=config['evaluation']
    if (benchmark.get('protocol')!=BENCHMARK_PROTOCOL or benchmark.get('input_seed')!=0 or
        benchmark.get('input_shape',[None])[0]!=evaluation['batch_size'] or
        benchmark.get('repeats')!=evaluation['repeats'] or benchmark.get('warmup')!=evaluation['warmup'] or
        len(benchmark.get('samples_ms',[]))!=evaluation['repeats'] or
        benchmark['runtime']['device'].split(':')[0]!=evaluation['device'] or
        benchmark['runtime']['torch_cpu_threads']!=evaluation['cpu_threads']):
        raise ValueError('Benchmark protocol/config differs')
    samples=benchmark['samples_ms']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
        raise ValueError('Invalid benchmark samples')
    median=statistics.median(samples)
    if not math.isclose(benchmark['latency_ms']['median'],median,rel_tol=1e-12):
        raise ValueError('Benchmark median differs from saved samples')
    import torch, hashlib
    shape=benchmark['input_shape']
    if any(type(n) is not int or n<1 for n in shape): raise ValueError('Invalid benchmark shape')
    dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}.get(benchmark['dtype'])
    if dtype is None: raise ValueError('Unsupported benchmark dtype')
    synthetic=torch.randn(tuple(shape),generator=torch.Generator(device='cpu').manual_seed(0),dtype=dtype)
    if hashlib.sha256(synthetic.numpy().tobytes()).hexdigest()!=benchmark.get('input_sha256'):
        raise ValueError('Benchmark synthetic input differs from protocol')
    payload=read_manifest(run/'manifest.json')
    rows=[r for r in manifest_selection(payload,config['data']['source'],config['data']['magnification']) if r['split']==split]
    values=predictions.get('rows',[])
    if len(values)!=len(rows): raise ValueError('Prediction coverage differs')
    for row,value in zip(rows,values):
        if any(value.get(k)!=row[k] for k in ('path','sha256','pixel_sha256','patient_id','group_id','magnification','subtype','split')):
            raise ValueError('Prediction ordering/identity differs')
        if value.get('target')!=CLASSES.index(row['label']): raise ValueError('Prediction target differs')
        validate_probabilities([value['target']],[value['probabilities']],CLASSES)
        if value.get('prediction')!=max(range(2),key=lambda i:value['probabilities'][i]):
            raise ValueError('Prediction decision differs from fixed threshold')
    targets=[v['target'] for v in values]; scores=[v['probabilities'] for v in values]
    primary=magnification_balanced_metrics(rows,targets,scores,CLASSES)
    secondary=grouped_patient_metrics(rows,targets,scores,CLASSES)
    if (report['group_metrics']!=primary or report['patient_metrics']!=secondary or
        predictions.get('groups')!=primary['groups'] or predictions.get('image_mean_groups')!=secondary['groups']):
        raise ValueError('Group metrics/prediction attribution differs')
    from .reporting import classification_metrics
    image=classification_metrics(targets,[v['prediction'] for v in values],CLASSES)
    if any(report['image_metrics'].get(k)!=v for k,v in image.items()):
        raise ValueError('Image metrics differ from predictions')
    return report,predictions
