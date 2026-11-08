"""Development-only study executor. Reuses experiment fitting and committed recovery."""
import contextlib
import copy
import csv
from datetime import datetime,timezone
import math
from pathlib import Path
import subprocess
import sys
import time
from . import baseline_plan as plan_api,experiments as e,experiment_io as io
from . import phase_artifacts as artifacts


def now():return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def lease(root):
    path=root/'.executor.lock'
    with path.open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if sys.platform=='win32':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:raise ValueError('Another executor owns this study') from error
        try:yield
        finally:
            stream.seek(0)
            if sys.platform=='win32':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def frozen(study_file):
    study=plan_api.load_study(study_file);root=plan_api.STUDIES/study['study_id']
    status=io.read_json(root/'status.json');plan=io.read_json(root/'jobs.json')
    if status['status']!='prepared' or status['plan_sha256']!=plan_api.sha(plan) or study!=plan['study']:
        raise ValueError('Study differs from frozen preparation')
    for name,digest in status['artifacts'].items():
        if e.digest(root/name)!=digest:raise ValueError('Prepared artifact changed: '+name)
    if e.revision()['source_sha256']!=plan['inputs']['implementation_sha256']:
        raise ValueError('Study source changed; prepare a new revision before fitting')
    return root,plan


def verified_job(job,record):
    run=e.RUNS/job['job_id'];config,state,inputs,selection=artifacts.development(run)
    if (e.digest(run/'development/results.json')!=record['results_sha256'] or
        e.digest(run/'development/selection.json')!=record['selection_sha256']):
        raise ValueError('Completed study job changed: '+job['job_id'])
    for ref in record.get('resources',[]):
        if e.digest(ref['path'])!=ref['sha256']:raise ValueError('Resource accounting changed')
    return config,state,inputs,selection


def choose(plan,ledger,selection_id):
    if selection_id in ledger['selections']:
        selected=ledger['selections'][selection_id]
        return selected['job_id']
    selector=next((s for s in plan['selections'] if s['selection_id']==selection_id),None)
    if selector is None:return selection_id
    choices=[]
    for option in selector['options']:
        rec=ledger['jobs'].get(option,{})
        if rec.get('status')!='completed':continue
        job=next(j for j in plan['fit_jobs'] if j['job_id']==option)
        verified_job(job,rec)
        report=e.read_json(e.RUNS/option/'development/results.json')['designated_metrics']
        metric=report['group_metrics']
        if not metric.get('available') or metric.get('balanced_accuracy') is None:raise ValueError('Parent group support missing')
        choices.append((-metric['balanced_accuracy'],report['image_metrics']['loss'],job['recipe_index'],option))
    if not choices:raise ValueError('Selected parent dependency is incomplete: '+selection_id)
    chosen=min(choices)[-1];ledger['selections'][selection_id]=dict(job_id=chosen,policy=selector['policy'],frozen_at=now())
    return chosen


def config_for(root,plan,ledger,job):
    config=copy.deepcopy(e.read_json(root/job['config_file'])['config'])
    for field,dependency in job['bindings'].items():
        origin=choose(plan,ledger,dependency)
        parent_job=next(j for j in plan['fit_jobs'] if j['job_id']==origin)
        rec=ledger['jobs'].get(origin,{})
        if rec.get('status')!='completed':raise ValueError('Dependency incomplete: '+origin)
        pc,ps,pi,selection=verified_job(parent_job,rec)
        if field=='training.recipe':
            for key in plan_api.RECIPE:config['training'][key]=pc['training'][key]
        elif field=='baseline.parent_selection':config['baseline']['parent_selection']=str(e.RUNS/origin/'development/selection.json')
        elif field=='baseline.topology_checkpoint':
            topology=Path(pi['checkpoint_directory'])/'prune.pth'
            from .checkpoints import read_checkpoint
            payload=read_checkpoint(topology);widths=payload['architecture_config'].get('resnet_channels',payload['architecture_config'].get('channels'))
            if widths!=job['planned_widths']:raise ValueError('Resolved topology differs from frozen planned widths')
            config['baseline'].update(topology_checkpoint=str(topology),channels=widths)
        else:raise ValueError('Unknown binding '+field)
    return config


def resources(run):
    if not (run/'inputs.json').exists():return [],[]
    inputs=e.read_json(run/'inputs.json');rows=[];refs=[]
    for name in ('train','finetune'):
        path=Path(inputs['checkpoint_directory'])/(name+'.resources.json')
        if path.exists():
            value=e.read_json(path);rows.extend(dict(row,command=name) for row in value['attempts'])
            refs.append(dict(path=str(path),sha256=e.digest(path)))
    return rows,refs


def job_cost(run,attempts):
    histories=[]
    for command in ('train','finetune'):
        path=run/(command+'.history.json')
        if path.exists():histories.extend(e.read_json(path)['epochs'])
    rows,refs=resources(run)
    if not (run/'inputs.json').exists():return dict(committed_epochs=0,actual_optimizer_steps=None,image_presentations=None,resource_attempts=[],resources=[],unknown_work=True)
    inputs=e.read_json(run/'inputs.json');counts=inputs.get('counts',{})
    ntrain=counts.get('train',{}).get('images')
    # counts format is obtained from frozen manifest if the older input schema differs.
    if ntrain is None:
        manifest=e.read_json(run/'manifest.json');cfg=e.read_json(run/'config.json')
        from .data import manifest_selection
        ntrain=sum(r['split']=='train' for r in manifest_selection(manifest,cfg['data']['source'],cfg['data']['magnification']))
    stage_seconds={name:rec.get('seconds',0) for name,rec in e.read_json(run/'run.json')['stages'].items()}
    return dict(stage_seconds=stage_seconds,committed_epochs=len(histories),committed_optimizer_steps=sum(r['optimizer_steps'] for r in histories),
        committed_image_presentations=len(histories)*ntrain,
        actual_optimizer_steps=sum(r['optimizer_steps'] for r in rows) if rows else None,
        image_presentations=sum(r['image_presentations'] for r in rows) if rows else None,
        replayed_optimizer_steps=max(0,sum(r['optimizer_steps'] for r in rows)-sum(r['optimizer_steps'] for r in histories)) if rows else None,
        training_seconds=sum(r['training_seconds'] for r in histories),validation_seconds=sum(r.get('validation_seconds',0) for r in histories),
        child_wall_seconds=sum(a.get('seconds',0) for a in attempts),resource_attempts=rows,resources=refs,
        unknown_work=not rows or any(a.get('unknown_termination',False) or 'seconds' not in a for a in attempts),
        note='Actual optimizer callbacks include replayed/partial epochs. Hard-killed or pre-profiler work is unknown, never assumed zero.')


def execute_job(root,plan,ledger,job,resume):
    key=job['job_id'];old=ledger['jobs'].get(key,{})
    if old.get('status')=='completed':verified_job(job,old);return
    run=e.RUNS/key
    if (old or run.exists()) and not resume:raise ValueError('Incomplete job requires --resume: '+key)
    config=config_for(root,plan,ledger,job);file=root/'execution/configs'/(key+'.json')
    if file.exists():
        if e.read_json(file)!=config:raise ValueError('Resolved config changed')
    else:io.write_json(file,config,root=root,overwrite=False)
    attempts=old.get('attempts',[]);attempt=dict(started_at=now(),status='running',config_sha256=e.digest(file))
    attempts.append(attempt);record=dict(status='running',attempts=attempts,run=str(run));ledger['jobs'][key]=record
    io.write_json(root/'execution/ledger.json',ledger,root=root)
    start=time.perf_counter()
    print('STUDY '+job['kind']+' '+job['model']+' '+key,flush=True)
    try:
        e.run_experiment(file,run_id=key,resume=run.exists())
        artifacts.development(run)
        attempt.update(status='completed');record.update(status='completed',results_sha256=e.digest(run/'development/results.json'),selection_sha256=e.digest(run/'development/selection.json'))
    except BaseException as error:
        attempt.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',error=dict(type=type(error).__name__,message=str(error)))
        record['status']=attempt['status'];record['error']=attempt['error']
        raise
    finally:
        attempt.update(seconds=time.perf_counter()-start,ended_at=now())
        record.update(job_cost(run,attempts));io.write_json(root/'execution/ledger.json',ledger,root=root)


def pilot_estimates(root,plan,ledger):
    costs={};basis={}
    for model in {j['model'] for j in plan['fit_jobs']}:
        samples=[]
        for job in plan['fit_jobs']:
            rec=ledger['jobs'].get(job['job_id'],{})
            if job['kind']=='pilot' and job['model']==model and rec.get('status')=='completed':
                verified_job(job,rec)
                if rec.get('unknown_work'):raise ValueError('Pilot has unmeasured interrupted work; repeat in a new revision')
                samples.append((job,rec))
        if len(samples)!=2:raise ValueError('Both 40X/pooled pilots required for '+model)
        epoch=max((r['training_seconds']+r['validation_seconds'])/max(1,r['committed_epochs']) for j,r in samples)
        fixed=max(r['child_wall_seconds']-r['training_seconds']-r['validation_seconds'] for j,r in samples)
        basis[model]=(epoch,fixed,[j['job_id'] for j,r in samples])
    for job in plan['fit_jobs']:
        epoch,fixed,origins=basis[job['model']]
        # Masks/scratch/pruned recovery are conservatively bounded by their dense family.
        costs[job['cost_key']]=dict(epoch_seconds=epoch*1.5,fixed_seconds=max(1,fixed)*3+30,
            basis='Conservative estimate: 1.5x worst measured family epoch across 40X/pooled, 3x worst non-epoch wall plus 30s; includes recipe uncertainty/three roles, not a measured candidate cost. Pilots: '+','.join(origins))
    value=dict(schema_version=1,plan_sha256=plan_api.sha(plan),estimates=costs)
    io.write_json(root/'execution/measured_cost_inputs.json',value,root=root)
    return value


def profile_pilot(root,plan,ledger,job):
    rec=ledger['jobs'][job['job_id']];run=e.RUNS/job['job_id']
    profile=root/'execution/profiles'/(job['job_id']+'.cuda.json')
    if rec.get('cuda_profile'):
        if e.digest(profile)!=rec['cuda_profile']['sha256']:raise ValueError('Pilot CUDA profile changed')
        return
    # Separate validation-only CUDA inference profile. CPU profile remains in the run.
    if not any(p['device']=='cuda' for p in plan['study']['evaluation_profiles']):return
    config=e.read_json(run/'config.json');inputs=e.read_json(run/'inputs.json')
    from . import cli
    profile.parent.mkdir(parents=True,exist_ok=True)
    settings=next(p for p in plan['study']['evaluation_profiles'] if p['device']=='cuda')
    argv=[sys.executable,'-m','breakhis','evaluate','--dataset','breakhis','--manifest',str(run/'manifest.json'),'--data-root',config['data']['data_root'],
        '--source',config['data']['source'],'--magnification',config['data']['magnification'],'--checkpoint',str(Path(inputs['checkpoint_directory'])/'train.pth'),
        '--device','cuda','--split','validation','--batch-size',str(config['training']['batch_size']),
        '--benchmark-batch-size',str(settings['batch_size']),'--benchmark-warmup',str(settings['warmup']),'--benchmark-repeats',str(settings['repeats']),'--output',str(profile)]
    log=profile.with_suffix('.log');started=time.perf_counter()
    from .paths import output_path
    for owned in (profile,profile.with_suffix('.config.json')):
        output_path(owned,root=root,protected=[run,inputs['checkpoint_directory']]);owned.unlink(missing_ok=True)
    import os
    attempt=dict(status='running',started_at=now());rec.setdefault('profile_attempts',[]).append(attempt)
    try:
        with log.open('a',encoding='utf-8') as stream:
            subprocess.run(argv,cwd=e.ROOT,env=dict(os.environ,OMP_NUM_THREADS=str(settings['cpu_threads']),MKL_NUM_THREADS=str(settings['cpu_threads']),PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
        attempt['status']='completed'
        rec['cuda_profile']=dict(path=str(profile),sha256=e.digest(profile))
    except BaseException as error:
        attempt.update(status='failed',error=str(error));raise
    finally:
        attempt['seconds']=time.perf_counter()-started
        io.write_json(root/'execution/ledger.json',ledger,root=root)


def report(root,plan,ledger):
    rows=[];diagnostics=[]
    by_id={j['job_id']:j for j in plan['fit_jobs']}
    for key,job in by_id.items():
        rec=ledger['jobs'].get(key,{})
        if rec.get('status')!='completed':
            diagnostics.append(dict(job_id=key,kind=job['kind'],model=job['model'],status=rec.get('status','not_run'),optional=job['optional_recipe'],error=rec.get('error')));continue
        config,state,inputs,selection=verified_job(job,rec)
        metrics=e.read_json(e.RUNS/key/'development/results.json')['designated_metrics'];g=metrics['group_metrics'];benchmark=metrics['benchmark']
        parent=job['bindings'].get('baseline.parent_selection');parent=choose(plan,ledger,parent) if parent else None
        parent_cost=ledger['jobs'].get(parent,{}) if parent else {}
        row=dict(job_id=key,model=job['model'],kind=job['kind'],scope=job['scope'],seed=job['seed'],ratio=job['ratio'],
            phase='development',split='validation',image_accuracy_percent=metrics['image_metrics']['accuracy'],group_sensitivity=g.get('sensitivity'),group_specificity=g.get('specificity'),group_balanced_accuracy=g.get('balanced_accuracy'),
            latency_median_ms=benchmark['latency_ms']['median'],evaluation_device=benchmark['runtime']['device'],
            selected_epoch=selection['roles'][selection['designated_role']]['epoch'],executed_epochs=rec['committed_epochs'],
            incremental_steps=rec['actual_optimizer_steps'],incremental_presentations=rec['image_presentations'],incremental_seconds=rec['child_wall_seconds'],
            parent_job_id=parent,parent_shared_steps=parent_cost.get('actual_optimizer_steps',0),
            full_lifecycle_steps=(rec['actual_optimizer_steps']+parent_cost.get('actual_optimizer_steps',0)) if rec['actual_optimizer_steps'] is not None and parent_cost.get('actual_optimizer_steps',0) is not None and job['kind']!='narrow_scratch' else rec['actual_optimizer_steps'],
            stage_seconds=rec.get('stage_seconds',{}),
            full_lifecycle_presentations=rec['image_presentations']+parent_cost.get('image_presentations',0) if rec['image_presentations'] is not None and parent_cost.get('image_presentations',0) is not None and job['kind']!='narrow_scratch' else rec['image_presentations'],
            full_lifecycle_seconds=rec['child_wall_seconds']+(parent_cost.get('child_wall_seconds',0) if job['kind']!='narrow_scratch' else 0),
            topology_discovery_seconds=parent_cost.get('child_wall_seconds',0) if job['kind']=='narrow_scratch' else 0,
            topology_discovery_steps=parent_cost.get('actual_optimizer_steps',0) if job['kind']=='narrow_scratch' else 0,
            resource_attempts=rec['resource_attempts'])
        rows.append(row)
    frontier=[]
    # Descriptive only: no cross-family matched causal claim; keep separate device/scope/seed strata.
    for row in rows:
        if row['kind']=='pilot' or any(row[k] is None for k in ('group_sensitivity','group_specificity','latency_median_ms')):continue
        def dominates(other):
            return (other['scope'],other['seed'],other['evaluation_device'])==(row['scope'],row['seed'],row['evaluation_device']) and all(other[k] is not None for k in ('group_sensitivity','group_specificity','latency_median_ms')) and other['group_sensitivity']>=row['group_sensitivity'] and other['group_specificity']>=row['group_specificity'] and other['latency_median_ms']<=row['latency_median_ms'] and (other['group_sensitivity']>row['group_sensitivity'] or other['group_specificity']>row['group_specificity'] or other['latency_median_ms']<row['latency_median_ms'])
        if not any(other['kind']!='pilot' and dominates(other) for other in rows):frontier.append(row['job_id'])
    required=[j for j in plan['fit_jobs'] if not j['optional_recipe']]
    result=dict(schema_version=1,phase='development',rows=rows,diagnostics=diagnostics,descriptive_frontier=frontier,
        complete_required=all(ledger['jobs'].get(j['job_id'],{}).get('status')=='completed' for j in required),
        scientific_approval=False,note='Software execution completeness is not baseline strength. Descriptive cross-family frontier; matched-control/quality review remains M2.5. No final test inference.',
        study_actual_child_seconds=sum(a.get('seconds',0) for r in ledger['jobs'].values() for a in r.get('attempts',[])),
        study_cuda_profile_seconds=sum(a.get('seconds',0) for r in ledger['jobs'].values() for a in r.get('profile_attempts',[])),
        unknown_work_jobs=[k for k,r in ledger['jobs'].items() if r.get('unknown_work')])
    result['study_preparation_seconds']=e.read_json(root/'status.json').get('preparation_seconds')
    result['study_actual_total_known_seconds']=result['study_actual_child_seconds']+result['study_cuda_profile_seconds']+(result['study_preparation_seconds'] or 0)
    io.write_json(root/'development/results.json',result,root=root)
    from io import StringIO
    text=StringIO();fields=[k for k in rows[0] if k not in ('resource_attempts','stage_seconds')] if rows else ['job_id','phase']
    writer=csv.DictWriter(text,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    io.publish(root/'development/results.csv',text.getvalue().encode('utf-8'),root=root)
    return result


def execute(study_file,stage,*,resume=False,session_hours=8,max_jobs=None,second_recipes=False):
    if type(session_hours) not in (int,float) or not math.isfinite(session_hours) or session_hours<=0:raise ValueError('Positive session hours required')
    root,plan=frozen(study_file)
    with lease(root):
        path=root/'execution/ledger.json'
        ledger=e.read_json(path) if path.exists() else dict(schema_version=1,plan_sha256=plan_api.sha(plan),jobs={},selections={},sessions=[],policy={})
        if ledger['plan_sha256']!=plan_api.sha(plan):raise ValueError('Execution ledger belongs to another plan')
        if stage=='report':return report(root,plan,ledger)
        if stage not in ('pilot','parents','candidates'):raise ValueError('Unknown execution stage')
        session=dict(stage=stage,started_at=now(),limit_hours=session_hours,status='running');ledger['sessions'].append(session)
        started=time.perf_counter();count=0
        try:
            if stage!='pilot':
                if stage=='parents':
                    if 'second_recipes' in ledger['policy'] and ledger['policy']['second_recipes']!=second_recipes:raise ValueError('Recipe policy is frozen; use a new study revision')
                    ledger['policy']['second_recipes']=second_recipes
                estimates=pilot_estimates(root,plan,ledger)
                retries=[]
                for key,record in ledger['jobs'].items():
                    if record.get('unknown_work'):raise ValueError('Unknown interrupted cost prevents capacity certification: '+key)
                    seconds=sum(a.get('seconds',0) for a in record.get('attempts',[])[:-1])+sum(a.get('seconds',0) for a in record.get('profile_attempts',[]))
                    if seconds:retries.append(dict(job_id=key,extra_fit_jobs=0,extra_seconds=seconds,basis='Recorded prior attempts and separate CUDA profile wall time, conservatively additive to full-plan estimates'))
                first=plan['fit_jobs'][0]['job_id']
                retries.append(dict(job_id=first,extra_fit_jobs=0,extra_seconds=e.read_json(root/'status.json').get('preparation_seconds',0),basis='Shared measured metadata preparation wall time, charged once'))
                capacity=plan_api.require_capacity(plan,estimates,retries)
                io.write_json(root/'execution/capacity.json',capacity,root=root)
            jobs=[j for j in plan['fit_jobs'] if (j['kind']=='pilot' if stage=='pilot' else j['kind']=='parent' if stage=='parents' else j['kind'] not in ('pilot','parent'))]
            for job in jobs:
                if job['optional_recipe'] and not ledger['policy'].get('second_recipes',False):continue
                if ledger['jobs'].get(job['job_id'],{}).get('status')=='completed':
                    verified_job(job,ledger['jobs'][job['job_id']])
                    if stage=='pilot':profile_pilot(root,plan,ledger,job)
                    continue
                elapsed=time.perf_counter()-started
                estimate=0 if stage=='pilot' else capacity['job_seconds'][job['job_id']]
                if elapsed+estimate>=session_hours*3600 or max_jobs is not None and count>=max_jobs:
                    session['status']='paused_at_job_boundary';break
                execute_job(root,plan,ledger,job,resume);count+=1
                if stage=='pilot':profile_pilot(root,plan,ledger,job)
            else:session['status']='completed'
            if stage=='parents' and session['status']=='completed':
                for selection in plan['selections']:choose(plan,ledger,selection['selection_id'])
            if stage=='pilot' and session['status']=='completed':pilot_estimates(root,plan,ledger)
            result=report(root,plan,ledger)
        except BaseException as error:
            session.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            session.update(seconds=time.perf_counter()-started,overrun_seconds=max(0,time.perf_counter()-started-session_hours*3600),ended_at=now());io.write_json(path,ledger,root=root)
        return result
