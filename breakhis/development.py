"""Validation-only development orchestration using the existing CLI primitives."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from . import experiments as e
from . import experiment_setup as setup


def now():
    return datetime.now(timezone.utc).isoformat()


def run_development(config_file, *, run_id=None, resume=False, seed=None, device=None):
    from .baseline_init import kind, continuation, inherited_dense, commands, designated, parent_path, reference_roots
    prepared=setup.prepare_run(config_file,run_id=run_id,resume=resume,seed=seed,device=device)
    run,config,state,inputs=(prepared[k] for k in ('run','config','state','inputs'))
    training=config['training']; evaluation=config['evaluation']; identity=inputs['identity']
    import torch
    resolved=('cuda' if torch.cuda.is_available() else 'cpu') if training['device']=='auto' else training['device']
    if resolved!=identity['resolved_device'] or (resolved=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Resume resolved training device differs from preparation')
    if evaluation['device']=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA evaluation requested but unavailable')
    frozen=dict(config,data=dict(config['data'],manifest=str(run/'manifest.json')))
    e.validate_data(frozen,splits=('train','validation'))
    checkpoint=Path(inputs['checkpoint_directory']); run_id=state['run_id']
    roots=reference_roots(run,inputs)
    expected=inputs['expected_artifacts']
    common=['--dataset','breakhis','--manifest',str(run/'manifest.json'),
        '--source',config['data']['source'],'--magnification',config['data']['magnification'],
        '--data-root',config['data']['data_root'],'--batch-size',str(training['batch_size']),
        '--num-workers','0','--device',resolved,'--experiment-id',run_id,
        '--deterministic' if training['deterministic'] else '--no-deterministic']
    env=dict(os.environ,OMP_NUM_THREADS=str(training['cpu_threads']),
        MKL_NUM_THREADS=str(training['cpu_threads']),PYTHONUTF8='1')
    phase=state['phases']['development']

    def save(): e.write_json(run/'run.json',state)

    def references(stage):
        return [setup.artifact_reference(p,root=checkpoint if Path(p).is_relative_to(checkpoint) else run,
            kind='checkpoint' if Path(p).is_relative_to(checkpoint) else 'run') for p in expected[stage]]

    def verify_completed(stage, record):
        artifacts=record.get('artifacts',[])
        actual=[setup.verify_reference(ref,roots=roots) for ref in artifacts]
        if set(actual)!={Path(p) for p in expected[stage]}:
            raise ValueError('Incomplete artifact contract: '+stage)

    def execute(stage,args,*,epochs=None):
        from .checkpoints import read_checkpoint
        from .fit_boundary import reconcile
        record=state['stages'].get(stage,{})
        if record.get('status')=='completed':
            verify_completed(stage,record)
            return
        output=Path(expected[stage][0])
        if epochs is not None:
            latest=output.with_name(output.stem+'-last.pth')
            if latest.is_file():
                boundary=read_checkpoint(latest)
                saved=boundary['config']
                required=dict(command=stage,epochs=epochs,seed=training['seed'],
                    batch_size=training['batch_size'],num_workers=0,deterministic=training['deterministic'],
                    resolved_device=resolved,learning_rate=training['learning_rate'],
                    optimizer_name=training['optimizer'],scheduler_name=training['scheduler'],
                    weight_decay=training['weight_decay'],selection_metric=training['selection_metric'],
                    cpu_threads=training['cpu_threads'],source_sha256=inputs['code']['source_sha256'],
                    output=str(output),experiment_id=run_id,data_root=config['data']['data_root'])
                if (any(saved.get(k)!=v for k,v in required.items()) or
                    boundary.get('runtime')!=identity['runtime'] or
                    boundary['architecture'].replace('resnet18','resnet')!=training['model'].replace('resnet18','resnet') or
                    any(boundary['metadata'].get(k)!=v for k,v in dict(
                        manifest_sha256=identity['manifest_sha256'],source=config['data']['source'],
                        magnification=config['data']['magnification'],preprocessing=identity['preprocessing']).items()) or
                    not (boundary.get('training_state') or {}).get('resumable')):
                    raise ValueError('Restart boundary identity/config/runtime differs: '+stage)
                if boundary['epoch']>epochs: raise ValueError('Boundary exceeds planned epochs')
                reconcile(output,boundary)
                if boundary['epoch']==epochs:
                    state['stages'][stage]=dict(status='completed',recovered_completed_boundary=True,
                        artifacts=references(stage))
                    save(); return
                args=[args[0],*common,'--resume',str(latest)]
            elif any(Path(p).exists() for p in expected[stage]):
                raise ValueError('Training artifacts exist without a committed restart boundary: '+stage)
        if stage=='prune' and record.get('status') in ('failed','interrupted','running'):
            # These uncommitted, deterministic products are regenerated from the
            # same frozen parent. Never clear a committed stage or any parent file.
            from .paths import output_path
            owned={checkpoint/'prune.pth',checkpoint/'prune.config.json',checkpoint/'prune.validation.json'}
            if {Path(p) for p in expected[stage]}!=owned:
                raise ValueError('Unrecognized incomplete pruning destinations')
            for path in owned:
                output_path(path,root=checkpoint,protected=[dense])
                path.unlink(missing_ok=True)
        log=run/'logs'/f'{stage}.log'; log.parent.mkdir(exist_ok=True)
        argv=[sys.executable,'-m','breakhis',*map(str,args)]
        record=dict(status='running',argv=argv,log=str(log),started_at=now())
        state['stages'][stage]=record; save()
        print('Running '+stage+'; log: '+str(log),flush=True)
        started=time.perf_counter()
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run(argv,cwd=e.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(return_code=result.returncode,seconds=time.perf_counter()-started)
        if result.returncode!=0:
            raise RuntimeError(stage+' failed with exit '+str(result.returncode)+'; see '+str(log))
        if epochs is not None:
            if not latest.is_file() or read_checkpoint(latest)['epoch']!=epochs:
                raise InterruptedError(stage+' stopped before planned epochs; resume explicitly')
            reconcile(output,read_checkpoint(latest))
        record.update(status='completed',artifacts=references(stage),ended_at=now())
        save()

    fit=['--learning-rate',str(training['learning_rate']),'--optimizer',training['optimizer'],
        '--weight-decay',str(training['weight_decay']),'--scheduler',training['scheduler'],
        '--seed',str(training['seed']),'--selection-metric',training['selection_metric']]
    if config.get('baseline'): fit += ['--initialization',str(run/'initialization.json')]
    try:
        state.update(status='running',active_phase='development')
        state.pop('error',None); phase.update(status='running',started_at=now()); phase.pop('error',None); save()
        if config.get('baseline'):
            if state['stages'].get('initialization',{}).get('status')=='completed':
                verify_completed('initialization',state['stages']['initialization'])
            else:
                state['stages']['initialization']=dict(status='completed',artifacts=references('initialization'))
            if kind(config)!='standard':
                if state['stages'].get('parent',{}).get('status')=='completed':
                    verify_completed('parent',state['stages']['parent'])
                else:
                    e.write_json(run/'parent.json',dict(kind=kind(config),parent=inputs['initialization']['parent'],
                        topology=inputs['initialization']['topology'],child_parent_fit_steps=0,
                        note='Immutable input acquisition; no parent fitting or synthetic optimizer/history boundary'))
                    state['stages']['parent']=dict(status='completed',artifacts=references('parent'),optimizer_steps=0)
                save()
        if continuation(config):
            dense=parent_path(inputs['initialization']); finetuned=checkpoint/'finetune.pth'
            execute('finetune',['finetune',*common,*fit,'--checkpoint',str(dense),'--epochs',str(training['epochs']),
                '--output',str(finetuned)],epochs=training['epochs'])
            models=dict(dense=dense,finetuned=finetuned)
        elif kind(config)=='prune_from_parent':
            dense=parent_path(inputs['initialization']);models=dict(dense=dense)
        else:
            dense=checkpoint/'train.pth'
            execute('train',['train',*common,*fit,'--model',training['model'],'--epochs',str(training['epochs']),
                '--output',str(dense)],epochs=training['epochs'])
            models=dict(dense=dense)
        if config['pruning']:
            pruning=config['pruning']; pruned=checkpoint/'prune.pth'; finetuned=checkpoint/'finetune.pth'
            prune_args=['prune',*common,'--checkpoint',str(dense),'--method',pruning['method'],
                '--sparsity',str(pruning['sparsity']),'--output',str(pruned)]
            if kind(config)=='prune_from_parent':
                prune_args += ['--initialization',str(run/'initialization.json'),'--ranking',config['baseline']['ranking']]
                if config['baseline']['ranking_seed'] is not None:
                    prune_args += ['--ranking-seed',str(config['baseline']['ranking_seed'])]
            execute('prune',prune_args)
            execute('finetune',['finetune',*common,*fit,'--checkpoint',str(pruned),
                '--epochs',str(pruning['finetune_epochs']),'--output',str(finetuned)],epochs=pruning['finetune_epochs'])
            models.update(pruned=pruned,finetuned=finetuned)
        env.update(OMP_NUM_THREADS=str(evaluation['cpu_threads']),MKL_NUM_THREADS=str(evaluation['cpu_threads']))
        reports={}; predictions={}
        for role,path in models.items():
            report=run/'development'/f'{role}.metrics.json'; prediction=report.with_name(f'{role}.predictions.json')
            execute('development-'+role,['evaluate',*common,'--split','validation','--model-role',role,
                '--experiment-config',str(run/'config.json'),'--device',evaluation['device'],
                '--checkpoint',str(path),'--output',str(report),'--predictions-output',str(prediction),
                '--benchmark-warmup',str(evaluation['warmup']),'--benchmark-repeats',str(evaluation['repeats']),
                '--benchmark-batch-size',str(evaluation['batch_size']),'--cpu-threads',str(evaluation['cpu_threads'])])
            if e.read_json(report).get('split')!='validation' or e.read_json(prediction).get('split')!='validation':
                raise ValueError('Development received a non-validation report')
            reports[role]=str(report); predictions[role]=str(prediction)
        for command in commands(config):
            e.write_json(run/f'{command}.history.json',e.read_json(checkpoint/f'{command}.history.json'))
        from .checkpoints import read_checkpoint
        selected={}
        def checkpoint_reference(path):
            for name in ('checkpoint','parent_checkpoint'):
                if name in roots and path.is_relative_to(roots[name]):
                    return setup.artifact_reference(path,root=roots[name],kind=name)
            raise ValueError('Selected checkpoint is outside declared roots')
        for role,path in models.items():
            payload=read_checkpoint(path)
            selected[role]=dict(checkpoint=checkpoint_reference(path),
                epoch=payload['epoch'],criterion='fixed_recipe' if role=='pruned' else training['selection_metric'],
                report=setup.artifact_reference(reports[role],root=run,kind='run'),
                predictions=setup.artifact_reference(predictions[role],root=run,kind='run'))
        designated_role=designated(config)
        selection=dict(schema_version=2,run_id=run_id,config_sha256=identity['config_sha256'],
            manifest_sha256=identity['manifest_sha256'],preprocessing=identity['preprocessing'],
            aggregation='filename-group/equal-magnification/mean-probability',threshold='argmax',
            classes=['benign','malignant'],class_to_idx={'benign':0,'malignant':1},
            designated_role=designated_role,roles=selected)
        e.write_json(run/'development/selection.json',selection)
        result=dict(schema_version=2,run_id=run_id,phase='development',split='validation',config=config,
            identity=identity,reports=reports,predictions=predictions,designated_role=designated_role,
            designated_metrics=e.read_json(reports[designated_role]),
            note='Validation development results; final evaluation has not run.')
        if config.get('baseline'):
            from .baseline_init import budget_summary
            result['baseline_budget']=budget_summary(config,inputs['initialization'],
                {command:e.read_json(run/f'{command}.history.json')['epochs'] for command in commands(config)})
        e.write_json(run/'development/results.json',result)
        state['stages']['development-complete']=dict(status='completed',artifacts=references('development-complete'))
        overall=e.read_json(run/'results.json'); overall['phases']['development']=str(run/'development/results.json')
        e.write_json(run/'results.json',overall)
        phase.update(status='completed',ended_at=now()); state.update(status='development_completed'); save()
    except BaseException as error:
        status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed'
        detail=dict(type=type(error).__name__,message=str(error))
        phase.update(status=status,error=detail); state.update(status=status,error=detail)
        for record in state['stages'].values():
            if record.get('status')=='running': record.update(status=status,error=detail)
        try: save()
        except OSError: pass  # Unavailable storage cannot provide a successful completion record.
        raise
    print('Completed development (validation only): '+str(run),flush=True)
    return run
