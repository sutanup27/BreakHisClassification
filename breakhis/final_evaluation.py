"""Explicit test inference from verified, development-selected portable weights."""
from pathlib import Path
import os
import subprocess
import sys
import time
from . import experiments as e, experiment_setup as setup
from .development import now
from .phase_artifacts import development, verify_stage, validate_report


def run_final(config_file=None,*,run_id=None,resume=False,seed=None,device=None):
    prepared=setup.prepare_run(config_file,run_id=run_id,phase='final',resume=resume,seed=seed,device=device)
    run=prepared['run']; config,state,inputs,selection=development(run)
    phase=state['phases']['final']
    if phase['status']!='not_requested' and not resume:
        raise ValueError('Final phase was already requested; use --resume for an interrupted/failed final phase')
    import torch
    evaluation=config['evaluation']
    if evaluation['device']=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA evaluation requested but unavailable')
    # No training availability/runtime or training-image requirement here.
    frozen=dict(config,data=dict(config['data'],manifest=str(run/'manifest.json')))
    e.validate_data(frozen,splits=('test',))
    roots=dict(run=run,checkpoint=Path(inputs['checkpoint_directory']))
    from .checkpoints import read_checkpoint
    selected={}
    for role,binding in selection['roles'].items():
        path=setup.verify_reference(binding['checkpoint'],roots=roots); payload=read_checkpoint(path)
        if (payload['classes']!=selection['classes'] or payload['epoch']!=binding['epoch'] or
            any(payload['metadata'].get(k)!=v for k,v in dict(
                manifest_sha256=inputs['identity']['manifest_sha256'],source=config['data']['source'],
                magnification=config['data']['magnification'],preprocessing=inputs['identity']['preprocessing']).items())):
            raise ValueError('Selected checkpoint attribution differs')
        selected[role]=path
    expected=inputs['expected_artifacts']
    env=dict(os.environ,OMP_NUM_THREADS=str(evaluation['cpu_threads']),
        MKL_NUM_THREADS=str(evaluation['cpu_threads']),PYTHONUTF8='1')
    def save(): e.write_json(run/'run.json',state)
    def refs(stage):
        return [setup.artifact_reference(p,root=run,kind='run') for p in expected[stage]]
    started=time.perf_counter()
    try:
        state.update(status='running',active_phase='final'); state.pop('error',None)
        phase.update(status='running',started_at=now()); phase.pop('error',None); save()
        reports={}; predictions={}
        for role,path in selected.items():
            stage='final-'+role; record=state['stages'].get(stage,{})
            report=run/'final'/f'{role}.metrics.json'; prediction=run/'final'/f'{role}.predictions.json'
            if record.get('status')=='completed':
                verify_stage(run,inputs,state,stage)
                validate_report(run,config,inputs,selection,role,'final')
            else:
                argv=[sys.executable,'-m','breakhis','evaluate','--dataset','breakhis',
                    '--manifest',str(run/'manifest.json'),'--source',config['data']['source'],
                    '--magnification',config['data']['magnification'],'--data-root',config['data']['data_root'],
                    '--batch-size',str(config['training']['batch_size']),'--num-workers','0',
                    '--device',evaluation['device'],'--experiment-id',state['run_id'],
                    '--deterministic' if config['training']['deterministic'] else '--no-deterministic',
                    '--split','test','--model-role',role,'--experiment-config',str(run/'config.json'),
                    '--checkpoint',str(path),'--output',str(report),'--predictions-output',str(prediction),
                    '--benchmark-warmup',str(evaluation['warmup']),'--benchmark-repeats',str(evaluation['repeats']),
                    '--benchmark-batch-size',str(evaluation['batch_size']),'--cpu-threads',str(evaluation['cpu_threads'])]
                from .paths import output_path
                protected=[run/'config.json',run/'manifest.json',run/'inputs.json',run/'development',*selected.values()]
                for destination in expected[stage]: output_path(destination,root=run,protected=protected)
                log=run/'logs'/f'{stage}.log'; log.parent.mkdir(exist_ok=True)
                record=dict(status='running',argv=argv,log=str(log),started_at=now())
                state['stages'][stage]=record; save(); tick=time.perf_counter()
                print('Running '+stage+'; log: '+str(log),flush=True)
                with log.open('a',encoding='utf-8') as stream:
                    result=subprocess.run(argv,cwd=e.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record.update(return_code=result.returncode,seconds=time.perf_counter()-tick)
                if result.returncode!=0: raise RuntimeError(stage+' failed; see '+str(log))
                validate_report(run,config,inputs,selection,role,'final')
                record.update(status='completed',artifacts=refs(stage),ended_at=now()); save()
            reports[role]=str(report); predictions[role]=str(prediction)
        # Verify that source bindings did not drift while children ran.
        development(run)
        role=selection['designated_role']
        result=dict(schema_version=2,run_id=state['run_id'],phase='final',split='test',config=config,
            identity=inputs['identity'],reports=reports,predictions=predictions,designated_role=role,
            designated_metrics=e.read_json(reports[role]),
            selection=setup.artifact_reference(run/'development/selection.json',root=run,kind='run'),
            note='All fixed roles evaluated; designated result follows development recipe, never test quality.')
        e.write_json(run/'final/results.json',result)
        state['stages']['final-complete']=dict(status='completed',artifacts=refs('final-complete'))
        overall=e.read_json(run/'results.json'); overall['phases']['final']=str(run/'final/results.json')
        e.write_json(run/'results.json',overall)
        phase.update(status='completed',ended_at=now(),seconds=time.perf_counter()-started)
        state.update(status='final_completed'); save()
    except BaseException as error:
        status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        detail=dict(type=type(error).__name__,message=str(error)); state.update(status=status,error=detail)
        phase.update(status=status,error=detail,seconds=time.perf_counter()-started)
        for record in state['stages'].values():
            if record.get('status')=='running': record.update(status=status,error=detail)
        try: save()
        except OSError: pass
        raise
    print('Completed explicit final evaluation: '+str(run),flush=True)
    return run
