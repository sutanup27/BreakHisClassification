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
    roots=dict(run=run,checkpoint=checkpoint)
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
                    boundary['architecture']!=training['model'] or
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
    try:
        state.update(status='running',active_phase='development')
        state.pop('error',None); phase.update(status='running',started_at=now()); phase.pop('error',None); save()
        dense=checkpoint/'train.pth'
        execute('train',['train',*common,*fit,'--model',training['model'],'--epochs',str(training['epochs']),
            '--output',str(dense)],epochs=training['epochs'])
        models=dict(dense=dense)
        if config['pruning']:
            pruning=config['pruning']; pruned=checkpoint/'prune.pth'; finetuned=checkpoint/'finetune.pth'
            execute('prune',['prune',*common,'--checkpoint',str(dense),'--method',pruning['method'],
                '--sparsity',str(pruning['sparsity']),'--output',str(pruned)])
            execute('finetune',['finetune',*common,*fit,'--checkpoint',str(pruned),
                '--epochs',str(pruning['finetune_epochs']),'--output',str(finetuned)],epochs=pruning['finetune_epochs'])
            models.update(pruned=pruned,finetuned=finetuned)
        env.update(OMP_NUM_THREADS=str(evaluation['cpu_threads']),MKL_NUM_THREADS=str(evaluation['cpu_threads']))
        reports={}; predictions={}
        for role,path in models.items():
            report=run/'development'/f'{role}.metrics.json'; prediction=report.with_name(f'{role}.predictions.json')
            execute('development-'+role,['evaluate',*common,'--split','validation','--device',evaluation['device'],
                '--checkpoint',str(path),'--output',str(report),'--predictions-output',str(prediction),
                '--benchmark-warmup',str(evaluation['warmup']),'--benchmark-repeats',str(evaluation['repeats']),
                '--benchmark-batch-size',str(evaluation['batch_size']),'--cpu-threads',str(evaluation['cpu_threads'])])
            if e.read_json(report).get('split')!='validation' or e.read_json(prediction).get('split')!='validation':
                raise ValueError('Development received a non-validation report')
            reports[role]=str(report); predictions[role]=str(prediction)
        for command in ('train','finetune') if config['pruning'] else ('train',):
            e.write_json(run/f'{command}.history.json',e.read_json(checkpoint/f'{command}.history.json'))
        from .checkpoints import read_checkpoint
        selected={}
        for role,path in models.items():
            payload=read_checkpoint(path)
            selected[role]=dict(checkpoint=setup.artifact_reference(path,root=checkpoint,kind='checkpoint'),
                epoch=payload['epoch'],criterion='fixed_recipe' if role=='pruned' else training['selection_metric'],
                report=setup.artifact_reference(reports[role],root=run,kind='run'),
                predictions=setup.artifact_reference(predictions[role],root=run,kind='run'))
        designated='finetuned' if config['pruning'] else 'dense'
        selection=dict(schema_version=2,run_id=run_id,config_sha256=identity['config_sha256'],
            manifest_sha256=identity['manifest_sha256'],preprocessing=identity['preprocessing'],
            aggregation='filename-group/equal-magnification/mean-probability',threshold='argmax',
            classes=['benign','malignant'],class_to_idx={'benign':0,'malignant':1},
            designated_role=designated,roles=selected)
        e.write_json(run/'development/selection.json',selection)
        result=dict(schema_version=2,run_id=run_id,phase='development',split='validation',config=config,
            identity=identity,reports=reports,predictions=predictions,designated_role=designated,
            designated_metrics=e.read_json(reports[designated]),
            note='Validation development results; final evaluation has not run.')
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
