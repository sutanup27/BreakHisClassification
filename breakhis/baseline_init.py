"""Explicit fresh-fit initialization and immutable external parent bindings."""
from pathlib import Path
from . import experiments as e, experiment_io as io
from .paths import output_path

BASELINE_FIELDS={'kind','parent_selection','channels','topology_checkpoint','ranking','ranking_seed'}


def kind(config): return config.get('baseline',{}).get('kind','standard')
def continuation(config): return kind(config)=='dense_continue'
def inherited_dense(config): return kind(config) in ('dense_continue','prune_from_parent')
def commands(config): return ['finetune'] if kind(config) in ('dense_continue','prune_from_parent') else ['train',*(['finetune'] if config['pruning'] else [])]
def roles(config): return ['dense','finetuned'] if continuation(config) else ['dense',*(['pruned','finetuned'] if config['pruning'] else [])]
def designated(config): return 'finetuned' if continuation(config) or config['pruning'] else 'dense'


def validate_baseline(config,base):
    b=config['baseline']; e.exact(b,BASELINE_FIELDS,'baseline')
    mode=b['kind']
    if mode not in ('standard','prune_from_parent','dense_continue','narrow_scratch'):
        raise ValueError('Unknown baseline kind')
    for field in ('parent_selection','topology_checkpoint'):
        value=b[field]
        if value is not None:
            if not isinstance(value,str) or not value.strip(): raise ValueError('Invalid baseline '+field)
            path=Path(value).expanduser(); b[field]=str((path if path.is_absolute() else base/path).resolve())
    if mode=='standard':
        if config['pruning'] is not None or any(b[k] is not None for k in BASELINE_FIELDS-{'kind'}):
            raise ValueError('Standard baseline requires pruning and initialization overrides null')
        return
    if b['parent_selection'] is None: raise ValueError('Parent selection is required')
    if mode=='prune_from_parent':
        if config['pruning'] is None or b['channels'] is not None or b['topology_checkpoint'] is not None:
            raise ValueError('Parent pruning requires pruning and no topology override')
        if b['ranking'] not in ('l1','random'): raise ValueError('Unknown channel ranking')
        if config['pruning']['method']=='unstructured' and b['ranking']!='l1': raise ValueError('Masks require L1 ranking')
        if b['ranking']=='random':
            e.integer(b['ranking_seed'],0,'ranking_seed')
            if b['ranking_seed']>=2**32: raise ValueError('Ranking seed must be uint32')
        elif b['ranking_seed'] is not None: raise ValueError('L1 ranking seed must be null')
    else:
        if config['pruning'] is not None or b['ranking'] is not None or b['ranking_seed'] is not None:
            raise ValueError('Continuation/scratch cannot prune or rank')
        if mode=='dense_continue':
            if b['channels'] is not None or b['topology_checkpoint'] is not None: raise ValueError('Continuation cannot override topology')
        else:
            if b['topology_checkpoint'] is None: raise ValueError('Scratch requires topology checkpoint')
            if config['training']['model'] in ('resnet','resnet18'):
                from .models.resnet import validate_widths
                validate_widths(b['channels'])
            elif (config['training']['model']!='vgg' or not isinstance(b['channels'],list) or len(b['channels'])!=8 or any(type(v) is not int or v<1 for v in b['channels'])):
                raise ValueError('Scratch requires supported architecture channel widths')


def reference_roots(run,inputs):
    result=dict(run=Path(run),checkpoint=Path(inputs['checkpoint_directory']))
    result.update({k:Path(v) for k,v in inputs.get('initialization',{}).get('roots',{}).items()})
    return result


def fit_contract(config,manifest_sha256):
    t=config['training']
    return dict(epochs=config['pruning']['finetune_epochs'] if kind(config)=='prune_from_parent' else t['epochs'],
        architecture=t['model'].replace('resnet18','resnet'),seed=t['seed'],learning_rate=t['learning_rate'],optimizer_name=t['optimizer'],
        weight_decay=t['weight_decay'],scheduler_name=t['scheduler'],selection_metric=t['selection_metric'],
        batch_size=t['batch_size'],deterministic=t['deterministic'],manifest_sha256=manifest_sha256,
        source=config['data']['source'],magnification=config['data']['magnification'],preprocessing=e.PREPROCESSING)


def inspect_initialization(config,manifest_sha256):
    """Verify a completed supported parent as new input, not exact parent resume."""
    from . import experiment_setup as s
    from .reporting import CLASSES
    from .checkpoints import read_checkpoint
    record=dict(schema_version=2,kind=kind(config),contract=fit_contract(config,manifest_sha256),
        channels=config['baseline']['channels'],roots={},references=[],parent=None,topology=None,
        pruning=config['pruning'],ranking=config['baseline']['ranking'],ranking_seed=config['baseline']['ranking_seed'])
    if kind(config)=='standard': return record
    selection_path=Path(config['baseline']['parent_selection'])
    if selection_path.name!='selection.json' or selection_path.parent.name!='development':
        raise ValueError('Parent must be a development/selection.json binding')
    parent_run=selection_path.parent.parent
    from .phase_artifacts import development
    pc,ps,pi,selection=development(parent_run,check_source=False)
    if selection['designated_role']!='dense' or pc['pruning'] is not None:
        raise ValueError('Parent must designate an unpruned dense fit')
    if any(selection.get(k)!=v for k,v in dict(manifest_sha256=manifest_sha256,classes=CLASSES,preprocessing=e.PREPROCESSING).items()):
        raise ValueError('Parent manifest/classes/preprocessing differ')
    if any(pc['data'][k]!=config['data'][k] for k in ('source','magnification')):
        raise ValueError('Parent source/magnification differs')
    for field in ('seed','learning_rate','optimizer','weight_decay','scheduler','selection_metric','batch_size','deterministic'):
        if pc['training'][field]!=config['training'][field]: raise ValueError('Parent recipe/seed differs: '+field)
    if pc['training']['model'].replace('resnet18','resnet')!=config['training']['model'].replace('resnet18','resnet'):
        raise ValueError('Parent model differs')
    if kind(config)=='prune_from_parent' and config['training']['epochs']!=pc['training']['epochs']:
        raise ValueError('Parent epoch allowance differs from frozen baseline')
    roots=dict(parent_run=str(parent_run.resolve()),parent_checkpoint=str(Path(pi['checkpoint_directory']).resolve()))
    record['roots']=roots
    # Bind immutable completed development artifacts, not mutable final-phase state.
    paths={parent_run/name for name in ('config.json','inputs.json','manifest.json')}
    for stage,stage_record in ps['stages'].items():
        if stage in ('train','parent','development-complete') or stage.startswith('development-'):
            for ref in stage_record.get('artifacts',[]):
                paths.add(s.verify_reference(ref,roots=reference_roots(parent_run,pi)))
    for path in sorted(paths):
        anchor='parent_run' if path.is_relative_to(parent_run) else 'parent_checkpoint'
        # Dense parents may have a topology-only origin of their own. Their own
        # inputs bind that origin; include transitive references under fixed names.
        if not path.is_relative_to(Path(roots[anchor])):
            raise ValueError('Parent artifact is outside declared parent roots')
        record['references'].append(s.artifact_reference(path,root=roots[anchor],kind=anchor))
    parent_path=s.verify_reference(selection['roles']['dense']['checkpoint'],roots=reference_roots(parent_run,pi))
    payload=read_checkpoint(parent_path)
    expected=dict(manifest_sha256=manifest_sha256,source=config['data']['source'],
        magnification=config['data']['magnification'],preprocessing=e.PREPROCESSING)
    if (payload['format_version']!=2 or payload['classes']!=CLASSES or payload['pruning']['type']!='dense' or
        payload['epoch']!=selection['roles']['dense']['epoch'] or
        any(payload['metadata'].get(k)!=v for k,v in expected.items()) or
        payload['architecture'].replace('resnet18','resnet')!=pc['training']['model'].replace('resnet18','resnet') or
        payload['config'].get('seed')!=pc['training']['seed']):
        raise ValueError('Parent checkpoint identity/topology differs')
    expected_fit=dict(seed=pc['training']['seed'],epochs=pc['training']['epochs'],
        learning_rate=pc['training']['learning_rate'],optimizer_name=pc['training']['optimizer'],
        weight_decay=pc['training']['weight_decay'],scheduler_name=pc['training']['scheduler'],
        selection_metric=pc['training']['selection_metric'],batch_size=pc['training']['batch_size'],
        deterministic=pc['training']['deterministic'],source_sha256=pi['code']['source_sha256'])
    if any(payload['config'].get(k)!=v for k,v in expected_fit.items()):
        raise ValueError('Parent checkpoint fitting/code identity differs from frozen parent')
    history=e.read_json(parent_run/'train.history.json')['epochs']
    record['parent']=dict(run_id=ps['run_id'],selection=s.artifact_reference(selection_path,root=parent_run,kind='parent_run'),
        checkpoint=s.artifact_reference(parent_path,root=roots['parent_checkpoint'],kind='parent_checkpoint'),
        selected_epoch=payload['epoch'],executed_epochs=len(history),optimizer_steps=sum(v['optimizer_steps'] for v in history),
        code=pi['code'],runtime=pi['identity']['runtime'],architecture=payload['architecture'])
    if kind(config)=='narrow_scratch':
        topology_path=Path(config['baseline']['topology_checkpoint']); topology=read_checkpoint(topology_path)
        if (topology['architecture'].replace('resnet18','resnet')!=pc['training']['model'].replace('resnet18','resnet') or topology['classes']!=CLASSES or topology['pruning']['type']!='channel' or
            topology['pruning']['channel_config']['method']!='l1-output-channel' or
            topology['architecture_config'].get('resnet_channels',topology['architecture_config'].get('channels'))!=record['channels'] or
            any(topology['metadata'].get(k)!=v for k,v in expected.items()) or
            topology['config'].get('parent_checkpoint_sha256')!=e.digest(parent_path) or
            Path(topology['config'].get('parent_checkpoint','')).resolve()!=parent_path.resolve()):
            raise ValueError('Topology is not the declared parent\'s bound L1 channel topology; regenerate it with current provenance')
        record['roots']['topology_checkpoint']=str(topology_path.parent.resolve())
        record['topology']=s.artifact_reference(topology_path,root=topology_path.parent,kind='topology_checkpoint')
        record['references'].append(record['topology'])
    verify_record(record)
    return record


def verify_record(record):
    from .experiment_setup import verify_reference
    version=record.get('schema_version')
    fields={'schema_version','kind','contract','channels','roots','references','parent','topology'}
    e.exact(record,fields | ({'pruning','ranking','ranking_seed'} if version==2 else set()),'initialization record')
    if type(version) is not int or version not in (1,2) or record.get('kind') not in ('standard','prune_from_parent','dense_continue','narrow_scratch'):
        raise ValueError('Unsupported initialization record')
    if set(record['roots'])-{'parent_run','parent_checkpoint','topology_checkpoint'}:
        raise ValueError('Unknown initialization roots')
    for name,value in record['roots'].items():
        path=Path(value)
        if not path.is_absolute(): raise ValueError('Initialization roots must be absolute')
        output_path(path/'.reference-check',root=path)
    for ref in record['references']: verify_reference(ref,roots=record['roots'])
    if record['parent']:
        for name in ('checkpoint','selection'): verify_reference(record['parent'][name],roots=record['roots'])
    if record['topology']: verify_reference(record['topology'],roots=record['roots'])
    return record


def parent_path(record):
    from .experiment_setup import verify_reference
    return verify_reference(record['parent']['checkpoint'],roots=record['roots'])


def verify_pruned(record,payload):
    expected=record['pruning']; saved=payload['config']
    if (saved.get('parent_checkpoint_sha256')!=record['parent']['checkpoint']['sha256'] or
        Path(saved.get('parent_checkpoint','')).resolve()!=parent_path(record).resolve() or
        saved.get('method')!=expected['method'] or saved.get('sparsity')!=expected['sparsity'] or
        saved.get('ranking')!=record['ranking'] or saved.get('ranking_seed')!=record['ranking_seed']):
        raise ValueError('Pruned checkpoint parent/ranking/ratio binding differs')
    pruning=payload['pruning']
    if expected['method']=='channel':
        cfg=pruning.get('channel_config',{})
        if (pruning['type']!='channel' or cfg.get('method')!=record['ranking']+'-output-channel' or
            cfg.get('ranking_seed')!=record['ranking_seed'] or
            any(v['requested_sparsity']!=expected['sparsity'] for v in cfg.get('groups',cfg.get('layers',{})).values())):
            raise ValueError('Pruned channel topology differs from frozen control')
    elif pruning['type']!='unstructured': raise ValueError('Expected L1 masked checkpoint')


def verify_cli(record,config,metadata,*,resume=False,input_path=None):
    verify_record(record)
    shared=record['kind']=='prune_from_parent'
    if shared and record['schema_version']!=2: raise ValueError('Regenerate shared-parent initialization with current ranking contract')
    pruning_command=shared and config['command']=='prune'
    for k,v in record['contract'].items():
        if pruning_command and k not in ('architecture','manifest_sha256','source','magnification','preprocessing'): continue
        actual=metadata.get(k) if k in ('manifest_sha256','source','magnification','preprocessing') else config.get(k)
        if k=='architecture' and isinstance(actual,str): actual=actual.replace('resnet18','resnet')
        if actual!=v: raise ValueError('Initialization fit contract differs: '+k)
    if pruning_command:
        p=record['pruning']
        if (Path(input_path)!=parent_path(record) or (config.get('layer_sparsities') is not None or config.get('group_sparsities') is not None) or
            config['method']!=p['method'] or config['sparsity']!=p['sparsity'] or
            config['ranking']!=record['ranking'] or config['ranking_seed']!=record['ranking_seed']):
            raise ValueError('Pruning command differs from frozen parent/ranking/ratio')
        return
    command='finetune' if record['kind'] in ('dense_continue','prune_from_parent') else 'train'
    if config['command']!=command or config.get('pretrained',False): raise ValueError('Initialization command/pretraining differs')
    if not resume and shared:
        from .checkpoints import read_checkpoint
        verify_pruned(record,read_checkpoint(input_path))
    elif not resume and command=='finetune' and Path(input_path)!=parent_path(record):
        raise ValueError('Continuation must initialize the bound parent')


def fresh_model(record,classes,device):
    from .checkpoints import create_model, load_checkpoint
    verify_record(record)
    if record['kind']=='prune_from_parent': raise ValueError('Parent pruning uses bound prune/finetune commands, not a fresh model')
    if record['kind']=='dense_continue': return load_checkpoint(parent_path(record),device)[0]
    options={'channels':record['channels']} if record['kind']=='narrow_scratch' else {}
    return create_model(record['contract']['architecture'],classes=classes,pretrained=False,**options).to(device)


def budget_summary(config,record,histories):
    parent=record['parent'] or {}
    inherited=kind(config) in ('dense_continue','prune_from_parent')
    discovery=kind(config)=='narrow_scratch'
    own_epochs=sum(len(rows) for rows in histories.values())
    parent_epochs=parent.get('executed_epochs',0)
    return dict(kind=kind(config),parent_selected_epoch=parent.get('selected_epoch'),
        parent_executed_epochs=parent_epochs,parent_optimizer_steps=parent.get('optimizer_steps',0),
        child_completed_epochs=own_epochs,
        child_optimizer_steps=sum(r['optimizer_steps'] for rows in histories.values() for r in rows),
        shared_parent_fit_epochs=parent_epochs if inherited else 0,
        topology_discovery_parent_epochs=parent_epochs if discovery else 0,
        full_lifecycle_fit_epochs=own_epochs+(parent_epochs if inherited else 0),
        including_topology_discovery_epochs=own_epochs+(parent_epochs if inherited or discovery else 0),
        note='Parent work is shared, not refitted in the child. Equal data passes are not equal FLOPs or wall time.')
