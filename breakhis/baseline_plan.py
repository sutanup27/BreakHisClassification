"""M2.1: strict, deterministic baseline study planning. No fitting/inference here."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from . import experiments as e, experiment_io as io
from .paths import output_path, validate_experiment_id, checkpoint_directory

STUDIES=e.ROOT/'artifacts/baselines'
FIELDS={'schema_version','study_id','experiment_template','pilot_magnification','main_magnification',
    'primary_seed','repeat_seed','ranking_seed','ratios','repeat_ratio','dense_epochs','recovery_epochs',
    'compact_model','recipe_candidates','evaluation_profiles','max_fit_jobs','available_wall_hours'}
RECIPE={'optimizer','learning_rate','weight_decay','scheduler'}
PROFILE={'profile_id','device','cpu_threads','batch_size','warmup','repeats'}


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def number(value,name,*,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or positive and value==0:
        raise ValueError('Invalid '+name)


def seed(value,name):
    e.integer(value,0,name)
    if value>=2**32: raise ValueError(name+' must be uint32')


def load_study(path):
    path=Path(path).expanduser().resolve(); study=io.read_json(path)
    e.exact(study,FIELDS,'baseline study')
    if type(study['schema_version']) is not int or study['schema_version']!=1: raise ValueError('Unsupported study schema')
    validate_experiment_id(study['study_id'])
    # A case-insensitive Windows workspace must not receive reserved device names.
    reserved={'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
    if study['study_id'].upper() in reserved: raise ValueError('Reserved study ID')
    template=study['experiment_template']
    if not isinstance(template,str) or not template.strip(): raise ValueError('Invalid experiment_template')
    template=Path(template).expanduser()
    study['experiment_template']=str((template if template.is_absolute() else path.parent/template).resolve())
    if study['pilot_magnification']!='40X' or study['main_magnification']!='all':
        raise ValueError('M2 study requires a 40X pilot and pooled all main magnifications')
    for key in ('primary_seed','repeat_seed','ranking_seed'): seed(study[key],key)
    if study['primary_seed']==study['repeat_seed']: raise ValueError('Primary/repeat seeds must differ')
    for key in ('dense_epochs','recovery_epochs'): e.integer(study[key],1,key)
    if study['compact_model']!='mobilenet_v3_small': raise ValueError('Unsupported compact model')
    ratios=study['ratios']
    if not isinstance(ratios,list) or not ratios: raise ValueError('Ratios must be a nonempty list')
    for value in [*ratios,study['repeat_ratio']]:
        number(value,'removal ratio',positive=True)
        if value>=1: raise ValueError('Removal ratios must be below one')
    if len(set(ratios))!=len(ratios) or study['repeat_ratio'] not in ratios:
        raise ValueError('Ratios must be distinct and contain repeat_ratio')
    recipes=study['recipe_candidates']
    if not isinstance(recipes,list) or not 1<=len(recipes)<=2: raise ValueError('Declare one or two recipes')
    for recipe in recipes:
        e.exact(recipe,RECIPE,'recipe')
        if recipe['optimizer'] not in ('adam','sgd') or recipe['scheduler'] not in ('none','cosine'):
            raise ValueError('Unsupported recipe optimizer/scheduler')
        number(recipe['learning_rate'],'recipe learning_rate',positive=True)
        number(recipe['weight_decay'],'recipe weight_decay')
    if len({sha(r) for r in recipes})!=len(recipes): raise ValueError('Duplicate recipe')
    profiles=study['evaluation_profiles']
    if not isinstance(profiles,list) or not 1<=len(profiles)<=2: raise ValueError('Declare CPU and optional CUDA profiles')
    for profile in profiles:
        e.exact(profile,PROFILE,'evaluation profile'); validate_experiment_id(profile['profile_id'])
        for key in ('cpu_threads','batch_size','repeats'): e.integer(profile[key],1,key)
        e.integer(profile['warmup'],0,'warmup')
    if ([p['device'] for p in profiles] not in (['cpu'],['cpu','cuda']) or
        profiles[0]['batch_size']!=1 or len({p['profile_id'].casefold() for p in profiles})!=len(profiles)):
        raise ValueError('Require primary CPU batch one, optional CUDA, and distinct profile IDs')
    if study['max_fit_jobs'] is not None: e.integer(study['max_fit_jobs'],1,'max_fit_jobs')
    if study['available_wall_hours'] is not None: number(study['available_wall_hours'],'available_wall_hours',positive=True)
    return study


def compile_plan(study_file):
    """Validate metadata/required-role paths only, then compile a prospective graph."""
    source=Path(study_file).resolve(); source_bytes=source.read_bytes()
    study=load_study(source); template_path=Path(study['experiment_template'])
    template_bytes=template_path.read_bytes(); template=e.load_config(template_path)
    if template['pruning'] is not None or template['training']['model']!='vgg':
        raise ValueError('Use an unpruned VGG baseline experiment template')
    if template['training']['selection_metric']!='group_balanced_accuracy':
        raise ValueError('M2 requires the group-balanced validation selection policy')
    from .data import read_manifest, manifest_selection, validate_role_rows
    from .reporting import AGGREGATION, THRESHOLD, BENCHMARK_PROTOCOL
    manifest=Path(template['data']['manifest']); manifest_bytes=manifest.read_bytes()
    payload=read_manifest(manifest); rows=manifest_selection(payload,template['data']['source'],'all')
    data_root=Path(template['data']['data_root'] or payload['roots'][template['data']['source']]).resolve()
    template['data']['data_root']=str(data_root); counts={}
    for mag in ('40X','all'):
        chosen=[r for r in rows if mag=='all' or r['magnification']==mag]
        counts[mag]={}
        for split in ('train','validation','test'):
            subset=[r for r in chosen if r['split']==split]
            if split!='test': validate_role_rows(subset,data_root,template['data']['source'],mag,split)
            counts[mag][split]=dict(images=len(subset),filename_groups=len({r['patient_id'] for r in subset}))
    identity=dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),source=template['data']['source'],
        main_magnification='all',pilot_magnification='40X',preprocessing=e.PREPROCESSING,
        aggregation=AGGREGATION,threshold=THRESHOLD,threshold_probability=.5,tie_class='benign',
        phase='development',selection_split='validation',classes=['benign','malignant'],
        benchmark_protocol=BENCHMARK_PROTOCOL,precision='float32',split_seed=payload.get('seed'))
    from .models.vgg import VGG
    widths=[v for v in VGG.ARCH if v!='M']
    jobs=[]; selectors=[]; seen={}; ids=set(); configs={}
    root=STUDIES/study['study_id']

    def add(kind,model,fit_seed,epochs,*,scope='main',recipe=0,ratio=None,dependencies=(),bindings=None,optional=False):
        dependencies=list(dict.fromkeys([*dependencies,*(bindings or {}).values()]))
        signature=dict(study_id=study['study_id'],kind=kind,model=model,seed=fit_seed,epochs=epochs,scope=scope,recipe=study['recipe_candidates'][recipe],ratio=ratio,
            dependencies=list(dependencies),bindings=bindings or {},identity=identity,template_sha256=sha(template))
        key=sha(signature)
        if key in seen: return seen[key]
        job_id=validate_experiment_id(study['study_id'][:28]+'-'+kind[:16]+'-'+key[:12])
        if job_id.casefold() in ids: raise ValueError('Duplicate generated job ID')
        ids.add(job_id.casefold()); seen[key]=job_id
        mag='40X' if scope=='pilot-40X' else 'all'
        config=copy.deepcopy(template); config.update(schema_version=2,name=job_id)
        config['data'].update(magnification=mag,manifest=str(root/'manifest.json'))
        config['training'].update(model=model,seed=fit_seed,epochs=epochs,**study['recipe_candidates'][recipe])
        config['evaluation']={k:v for k,v in study['evaluation_profiles'][0].items() if k!='profile_id'}
        baseline=dict(kind='standard',parent_selection=None,channels=None,topology_checkpoint=None,ranking=None,ranking_seed=None)
        if kind in ('l1_channel','random_channel','l1_mask'):
            baseline.update(kind='prune_from_parent',ranking='random' if kind=='random_channel' else 'l1',
                ranking_seed=study['ranking_seed'] if kind=='random_channel' else None)
            config['pruning']=dict(method='unstructured' if kind=='l1_mask' else 'channel',sparsity=ratio,
                finetune_epochs=study['recovery_epochs'])
        elif kind=='narrow_scratch': baseline['kind']='narrow_scratch'
        elif kind=='dense_continue': baseline['kind']='dense_continue'
        config['baseline']=baseline
        planned_widths=[max(1,c-round(c*ratio)) for c in widths] if ratio is not None and kind!='l1_mask' else None
        # Drafts never guess a selected parent or pretend null lineage is runnable.
        unresolved=list((bindings or {}).keys())
        if kind=='narrow_scratch':
            baseline['channels']=planned_widths; unresolved.append('baseline.topology_checkpoint')
        fit_epochs=study['recovery_epochs'] if baseline['kind']=='prune_from_parent' else epochs
        jobs.append(dict(job_id=job_id,kind=kind,model=model,seed=fit_seed,scope=scope,
            magnification=mag,phase='development',recipe_index=recipe,optional_recipe=optional,
            condition='bounded_second_recipe_requested' if optional else 'required',dependencies=list(dependencies),
            bindings=bindings or {},fit_epochs=fit_epochs,ratio=ratio,planned_widths=planned_widths,
            parent_identity_key=key if kind=='parent' else None,
            cost_key='/'.join([mag,model,kind]+([str(ratio)] if ratio is not None else [])),
            config_file=f'configs/{job_id}.json',requires_resolution=list(dict.fromkeys(unresolved)),runnable=False,
            checkpoint_directory=str(checkpoint_directory('cancer_binary')/mag/model/job_id)))
        configs[job_id]=dict(schema_version=1,document_type='m2-experiment-draft',runnable=False,
            reason='M2.2-M2.4 must resolve parent/recipe/topology bindings and implement config-2 execution',
            job_id=job_id,config=config,bindings=bindings or {})
        return job_id

    B,F=study['dense_epochs'],study['recovery_epochs']; primary=study['primary_seed']
    parents={}
    for model in ('vgg','resnet18',study['compact_model']):
        options=[]
        for index in range(len(study['recipe_candidates'])):
            options.append(add('parent',model,primary,B+F if model==study['compact_model'] else B,
                recipe=index,optional=index>0))
        selector=validate_experiment_id(study['study_id'][:28]+'-select-'+model+'-'+sha(study['study_id'])[:8])
        selectors.append(dict(selection_id=selector,model=model,seed=primary,options=options,
            policy='validation-group-BA/lower-loss/earlier-recipe',phase='development'))
        parents[(primary,model)]=selector
    repeat=study['repeat_seed']
    repeat_parent=add('parent','vgg',repeat,B,dependencies=[parents[(primary,'vgg')]],
        bindings={'training.recipe':parents[(primary,'vgg')]})
    parents[(repeat,'vgg')]=repeat_parent
    for fit_seed,ratios in ((primary,study['ratios']),(repeat,[study['repeat_ratio']])):
        parent=parents[(fit_seed,'vgg')]
        for ratio in ratios:
            l1=add('l1_channel','vgg',fit_seed,B,ratio=ratio,dependencies=[parent],
                bindings={'baseline.parent_selection':parent,'training.recipe':parents[(primary,'vgg')]})
            for kind in ('random_channel','l1_mask','narrow_scratch'):
                deps=[parent,*([l1] if kind=='narrow_scratch' else [])]
                bindings={'baseline.parent_selection':parent,'training.recipe':parents[(primary,'vgg')]}
                if kind=='narrow_scratch': bindings['baseline.topology_checkpoint']=l1
                add(kind,'vgg',fit_seed,B+F if kind=='narrow_scratch' else B,ratio=ratio,
                    dependencies=deps,bindings=bindings)
        add('dense_continue','vgg',fit_seed,F,dependencies=[parent],
            bindings={'baseline.parent_selection':parent,'training.recipe':parents[(primary,'vgg')]})
    for mag in ('40X','all'):
        for model in ('vgg','resnet18',study['compact_model']):
            add('pilot',model,primary,1,scope='pilot-'+mag)
    plan=dict(schema_version=1,document_type='m2-study-plan',study=study,identity=identity,counts=counts,
        template=template,inputs=dict(study_source=str(source),study_sha256=hashlib.sha256(source_bytes).hexdigest(),
            template_source=str(template_path),template_sha256=hashlib.sha256(template_bytes).hexdigest(),
            implementation_sha256={p.relative_to(e.ROOT).as_posix():e.digest(p) for p in sorted((e.ROOT/'breakhis').rglob('*.py'))}),
        fit_jobs=jobs,selections=selectors,
        totals=dict(required_main=sum(j['scope']=='main' and not j['optional_recipe'] for j in jobs),
            optional_recipes=sum(j['optional_recipe'] for j in jobs),pilot=sum(j['scope']!='main' for j in jobs),
            maximum_fit_jobs=len(jobs)),
        execution_supported=False,note='Planning only. No test inference, fit, checkpoint reservation or profile measurement.')
    validate_graph(plan)
    if (source.read_bytes()!=source_bytes or template_path.read_bytes()!=template_bytes or manifest.read_bytes()!=manifest_bytes):
        raise ValueError('Inputs changed during planning')
    return plan,configs,manifest_bytes,template_bytes


def validate_graph(plan):
    from .reporting import AGGREGATION, THRESHOLD, BENCHMARK_PROTOCOL
    required=dict(phase='development',selection_split='validation',source='cancer_binary',
        main_magnification='all',pilot_magnification='40X',preprocessing=e.PREPROCESSING,
        aggregation=AGGREGATION,threshold=THRESHOLD,benchmark_protocol=BENCHMARK_PROTOCOL,
        classes=['benign','malignant'],precision='float32',threshold_probability=.5,tie_class='benign')
    if any(plan['identity'].get(k)!=v for k,v in required.items()):
        raise ValueError('Study data/phase/metric identity differs')
    nodes=plan['fit_jobs']+plan['selections']; keys=[n.get('job_id',n.get('selection_id')) for n in nodes]
    if any(not isinstance(k,str) for k in keys) or len({k.casefold() for k in keys})!=len(keys):
        raise ValueError('Duplicate/invalid plan node IDs')
    graph={k:n.get('dependencies',n.get('options',[])) for k,n in zip(keys,nodes)}
    visiting=set(); done=set()
    def visit(node):
        if node not in graph: raise ValueError('Unknown parent dependency: '+node)
        if node in visiting: raise ValueError('Cyclic parent dependency')
        if node in done: return
        visiting.add(node)
        for dependency in graph[node]: visit(dependency)
        visiting.remove(node); done.add(node)
    for node in graph: visit(node)
    by_id={k:n for k,n in zip(keys,nodes)}
    for selection in plan['selections']:
        if selection['phase']!='development': raise ValueError('Selection must use development')
        for option in selection['options']:
            parent=by_id[option]
            if parent.get('kind')!='parent' or parent['seed']!=selection['seed'] or parent['model']!=selection['model']:
                raise ValueError('Recipe selection parent identity differs')
    for job in plan['fit_jobs']:
        if job['phase']!='development' or job['magnification'] not in ('40X','all'):
            raise ValueError('Plan phase/magnification differs')
        if job['scope']=='main' and job['magnification']!='all': raise ValueError('Main study cannot use pilot data')
        for dependency in job['bindings'].values():
            if dependency not in job['dependencies']: raise ValueError('Unbound parent dependency')
        parent_id=job['bindings'].get('baseline.parent_selection')
        if parent_id:
            parent=by_id[parent_id]
            if parent['seed']!=job['seed'] or parent['model']!='vgg' or parent.get('phase')!='development':
                raise ValueError('Bound parent seed/model/phase differs')
        topology=job['bindings'].get('baseline.topology_checkpoint')
        if topology:
            origin=by_id[topology]
            if (origin.get('kind')!='l1_channel' or origin['seed']!=job['seed'] or
                    origin['ratio']!=job['ratio'] or origin['planned_widths']!=job['planned_widths']):
                raise ValueError('Topology source differs from matched L1 control')
    return plan


def estimate_cost(plan, estimates=None, retries=()):
    """Conservative maximum-plan capacity; values are supplied, not measured here."""
    jobs={j['job_id']:j for j in plan['fit_jobs']}; costs={}; missing=[]
    if estimates is not None:
        e.exact(estimates,{'schema_version','plan_sha256','estimates'},'cost estimates')
        if type(estimates['schema_version']) is not int or estimates['schema_version']!=1 or estimates['plan_sha256']!=sha(plan):
            raise ValueError('Cost estimates belong to a different plan')
        if not isinstance(estimates['estimates'],dict): raise ValueError('Invalid cost estimates')
        allowed={j['cost_key'] for j in jobs.values()}
        if set(estimates['estimates'])-allowed: raise ValueError('Unknown cost key')
        for key,entry in estimates['estimates'].items():
            e.exact(entry,{'epoch_seconds','fixed_seconds','basis'},'cost entry')
            number(entry['epoch_seconds'],'epoch_seconds',positive=True); number(entry['fixed_seconds'],'fixed_seconds')
            if not isinstance(entry['basis'],str) or not entry['basis'].strip(): raise ValueError('Cost basis is required')
    for job in jobs.values():
        entry=None if estimates is None else estimates['estimates'].get(job['cost_key'])
        if entry is None: missing.append(job['cost_key'])
        else:
            seconds=job['fit_epochs']*entry['epoch_seconds']+entry['fixed_seconds']
            if not math.isfinite(seconds): raise ValueError('Cost estimate overflows finite seconds')
            costs[job['job_id']]=seconds
    extra_jobs=0; extra_seconds=0
    for retry in retries:
        e.exact(retry,{'job_id','extra_fit_jobs','extra_seconds','basis'},'retry overhead')
        if retry['job_id'] not in jobs: raise ValueError('Unknown retry job')
        e.integer(retry['extra_fit_jobs'],0,'extra_fit_jobs'); number(retry['extra_seconds'],'retry seconds')
        if not isinstance(retry['basis'],str) or not retry['basis'].strip(): raise ValueError('Retry basis required')
        extra_jobs+=retry['extra_fit_jobs']; extra_seconds+=retry['extra_seconds']
    required=len(jobs)+extra_jobs
    seconds=None if missing else sum(costs.values())+extra_seconds
    if not math.isfinite(extra_seconds) or seconds is not None and not math.isfinite(seconds):
        raise ValueError('Combined cost estimate overflows finite seconds')
    study=plan['study']; reasons=[]
    if missing: reasons.append('Missing cost estimates: '+', '.join(sorted(set(missing))))
    if study['max_fit_jobs'] is None: reasons.append('max_fit_jobs not supplied')
    elif required>study['max_fit_jobs']: reasons.append('Fit-job capacity shortfall')
    if study['available_wall_hours'] is None: reasons.append('available_wall_hours not supplied')
    elif seconds is not None and seconds/3600>study['available_wall_hours']: reasons.append('Wall-time capacity shortfall')
    return dict(schema_version=1,plan_sha256=sha(plan),ready_for_main_capacity=not reasons,
        execution_supported=False,required_fit_jobs=required,estimated_seconds=seconds,
        missing_cost_keys=sorted(set(missing)),job_seconds=costs,retry_seconds=extra_seconds,reasons=reasons,
        note='Worst-case includes every pilot and optional recipe once, shared parents once, plus supplied replay costs. Estimates are not verified measurements or a calendar guarantee.')


def require_capacity(plan,estimates=None,retries=()):
    result=estimate_cost(plan,estimates,retries)
    if not result['ready_for_main_capacity']: raise ValueError('; '.join(result['reasons']))
    return result


def prepare(study_file,*,cost_file=None):
    plan,configs,manifest,template=compile_plan(study_file)
    estimates=io.read_json(cost_file) if cost_file else None
    cost=estimate_cost(plan,estimates)
    protected=[study_file,plan['inputs']['template_source'],plan['template']['data']['manifest'],plan['template']['data']['data_root']]
    if cost_file: protected.append(cost_file)
    root=output_path(STUDIES/plan['study']['study_id'],root=STUDIES,protected=protected)
    for job in plan['fit_jobs']:
        run=e.RUNS/job['job_id']; checkpoint=Path(job['checkpoint_directory'])
        output_path(run/'run.json',root=e.RUNS,protected=protected)
        output_path(checkpoint/'train.pth',root=checkpoint_directory('cancer_binary'),protected=protected)
        if run.exists() or checkpoint.exists():
            raise ValueError('Planned run/checkpoint output already exists: '+job['job_id'])
    for name in ('study.json','jobs.json','manifest.json','template.json','status.json','cost_estimate.json',*[f'configs/{key}.json' for key in configs]):
        output_path(root/name,root=STUDIES,protected=protected)
    if root.exists(): raise ValueError('Study output exists; choose a new study_id')
    root.parent.mkdir(parents=True,exist_ok=True); root.mkdir()
    status=dict(schema_version=1,status='preparing',fitting_started=False,execution_supported=False)
    try:
        io.write_json(root/'status.json',status,root=root,overwrite=False)
        io.write_json(root/'study.json',plan['study'],root=root,overwrite=False)
        io.write_json(root/'jobs.json',plan,root=root,overwrite=False)
        io.publish(root/'manifest.json',manifest,root=root,overwrite=False)
        io.publish(root/'template.json',template,root=root,overwrite=False)
        io.write_json(root/'cost_estimate.json',cost,root=root,overwrite=False)
        if estimates is not None: io.write_json(root/'cost_inputs.json',estimates,root=root,overwrite=False)
        for job_id,config in configs.items(): io.write_json(root/f'configs/{job_id}.json',config,root=root,overwrite=False)
        artifacts={p.relative_to(root).as_posix():e.digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='status.json'}
        status.update(status='prepared',artifacts=artifacts,plan_sha256=sha(plan))
        io.write_json(root/'status.json',status,root=root)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=dict(type=type(error).__name__,message=str(error)))
        try: io.write_json(root/'status.json',status,root=root)
        except OSError: pass
        raise
    return root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','pilot','parents','candidates','report'],required=True)
    parser.add_argument('--cost-estimates',type=Path,help='Optional plan-bound cost JSON; no measurements are made by preparation')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.stage!='prepare': parser.exit(1,'Only prepare is implemented (M2.1); execution/report stages require M2.2-M2.5.\n')
    if args.resume: parser.exit(1,'Preparation never overwrites a study; use a new study_id. Execution resume is added in M2.4.\n')
    try:
        root=prepare(args.study,cost_file=args.cost_estimates)
        print('Prepared baseline plan only; no training/inference: '+str(root))
        print('Config files are non-runnable drafts until M2.2-M2.4 resolve/execute their bindings.')
    except (ValueError,OSError,KeyError) as error: parser.exit(1,str(error)+'\n')
