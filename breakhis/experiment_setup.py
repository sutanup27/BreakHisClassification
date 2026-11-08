"""M1.1 run preparation and identity contracts. No fitting or inference here."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

from . import experiment_io as io
from .paths import output_path, validate_experiment_id, checkpoint_output, checkpoint_directory

SCHEMA_VERSION = 2
PHASES = ('development', 'final')


def validate_request(*, config_file, run_id, phase, resume=False, seed=None, device=None):
    if phase not in PHASES:
        raise ValueError('Unknown experiment phase')
    if run_id is not None:
        validate_experiment_id(run_id)
    if phase == 'development' and config_file is None:
        raise ValueError('Development requires --config')
    if (phase == 'final' or resume) and run_id is None:
        raise ValueError('Final/resume requires the original --run-id')
    if phase == 'final' and (seed is not None or device is not None):
        raise ValueError('Seed/training-device overrides are development-only')


def artifact_reference(path, *, root, kind):
    from .experiments import digest
    if kind not in ('run', 'checkpoint', 'parent_run', 'parent_checkpoint', 'topology_checkpoint'):
        raise ValueError('Unknown artifact root kind')
    path = output_path(path, root=root)
    if not path.is_file():
        raise ValueError('Required artifact is missing: '+str(path))
    return dict(root=kind, path=path.relative_to(Path(root).resolve()).as_posix(),
                sha256=digest(path), bytes=path.stat().st_size)


def verify_reference(reference, *, roots):
    from .experiments import digest
    if not isinstance(reference, dict) or set(reference) != {'root','path','sha256','bytes'}:
        raise ValueError('Invalid artifact reference')
    if not isinstance(reference['root'],str) or reference['root'] not in roots or not isinstance(reference['path'],str):
        raise ValueError('Unknown artifact root/path')
    relative = Path(reference['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe artifact path')
    root = Path(roots[reference['root']])
    path = output_path(root/relative, root=root)
    if (not path.is_file() or type(reference['bytes']) is not int or
            path.stat().st_size != reference['bytes'] or digest(path) != reference['sha256']):
        raise ValueError('Artifact missing or changed: '+str(path))
    return path


def expected_outputs(config, run, checkpoint):
    """List every required output, not just the checkpoint/metrics primary file."""
    outputs = {}
    from .baseline_init import roles as baseline_roles, commands as baseline_commands, kind
    roles = baseline_roles(config)
    commands = baseline_commands(config)
    if config['pruning']:
        outputs['prune'] = [checkpoint/'prune.pth', checkpoint/'prune.config.json', checkpoint/'prune.validation.json']
    for command in commands:
        outputs[command] = [checkpoint/f'{command}{suffix}' for suffix in
            ('.pth','-last.pth','.config.json','.history.json')]
    for phase in PHASES:
        for role in roles:
            outputs[phase+'-'+role] = [run/phase/f'{role}.metrics.json', run/phase/f'{role}.predictions.json',
                run/phase/f'{role}.metrics.config.json']
    if config.get('baseline'):
        outputs['initialization']=[run/'initialization.json']
        if kind(config)!='standard': outputs['parent']=[run/'parent.json']
    outputs['development-complete'] = [run/'development/selection.json',run/'development/results.json',
        *[run/f'{command}.history.json' for command in commands],
        *([run/'initialization.json'] if config.get('baseline') else [])]
    outputs['final-complete'] = [run/'final/results.json']
    return {stage:list(map(str, paths)) for stage,paths in outputs.items()}


def _read_prepared(run, *, check_source=True):
    from .experiments import read_json, digest, revision
    for name in ('run.json','config.json','inputs.json','manifest.json','results.json'):
        output_path(run/name, root=run)
    state = read_json(run/'run.json')
    if not isinstance(state,dict) or type(state.get('schema_version')) is not int or state['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Schema-1 runs are not phase-aware; keep them unchanged and use a new run ID')
    config = read_json(run/'config.json')
    inputs = read_json(run/'inputs.json')
    if not isinstance(inputs,dict) or not isinstance(inputs.get('identity'),dict) or not isinstance(inputs.get('code'),dict):
        raise ValueError('Malformed prepared-run identity records')
    identity = inputs['identity']
    if not {'config_sha256','manifest_sha256','runtime'} <= identity.keys() or 'source_sha256' not in inputs['code']:
        raise ValueError('Incomplete prepared-run identity')
    if not isinstance(state.get('phases'),dict) or any(
            not isinstance(state['phases'].get(name),dict) or state['phases'][name].get('status') not in
            ('not_requested','running','completed','failed','interrupted') for name in PHASES):
        raise ValueError('Malformed phase states')
    if digest(run/'config.json') != identity['config_sha256'] or digest(run/'manifest.json') != identity['manifest_sha256']:
        raise ValueError('Frozen config/manifest changed')
    if check_source and revision()['source_sha256'] != inputs['code']['source_sha256']:
        raise ValueError('Implementation changed since preparation')
    if config.get('baseline'):
        from .baseline_init import verify_record
        if (digest(run/'initialization.json')!=inputs.get('initialization_sha256') or
                read_json(run/'initialization.json')!=inputs.get('initialization')):
            raise ValueError('Frozen initialization changed')
        verify_record(inputs['initialization'])
    return config, state, inputs


def prepare_run(config_file=None, *, run_id=None, phase='development', resume=False, seed=None, device=None):
    """Prepare schema-2 records or validate an existing phase request, never execute it."""
    from . import experiments as e
    validate_request(config_file=config_file, run_id=run_id, phase=phase, resume=resume, seed=seed, device=device)
    if phase == 'final' or resume:
        run = output_path(e.RUNS/run_id, root=e.RUNS)
        config, state, inputs = _read_prepared(run)
        if config_file is not None:
            supplied = e.load_config(config_file, seed=seed, device=device)
            if supplied['data']['data_root'] is None:
                supplied['data']['data_root'] = config['data']['data_root']
            if supplied != config:
                raise ValueError('Config differs from frozen effective settings')
        if phase == 'final':
            if state['phases']['development']['status'] != 'completed':
                raise ValueError('Final requires completed development; preparation is not development completion')
            if state['phases']['final']['status'] == 'completed':
                raise ValueError('Final phase is already completed')
        else:
            import torch
            runtime = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda)
            if runtime != inputs['identity']['runtime']:
                raise ValueError('Resume runtime differs from preparation')
            if e.digest(config['data']['manifest']) != inputs['identity']['manifest_sha256']:
                raise ValueError('Original manifest changed')
            if state['phases']['development']['status'] == 'completed':
                raise ValueError('Development phase is already completed')
        return dict(run=run, config=config, state=state, inputs=inputs)

    source_config = Path(config_file).resolve()
    config_hash = e.digest(source_config)
    config = e.load_config(source_config, seed=seed, device=device)
    manifest_hash = e.digest(config['data']['manifest'])
    _, counts = e.validate_data(config,splits=('train','validation'))
    manifest_bytes = Path(config['data']['manifest']).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash or e.digest(source_config) != config_hash:
        raise ValueError('Inputs changed during preparation')
    initialization=None
    if config.get('baseline'):
        from .baseline_init import inspect_initialization
        initialization=inspect_initialization(config,manifest_hash)
    import torch
    training_device = config['training']['device']
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if training_device == 'auto' else training_device
    if resolved == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA training requested but unavailable')
    run_id = validate_experiment_id(run_id or config['name'][:55]+'-'+uuid.uuid4().hex[:8])
    protected = [source_config, config['data']['manifest'], config['data']['data_root']]
    if initialization: protected.extend(initialization['roots'].values())
    run = output_path(e.RUNS/run_id, root=e.RUNS, protected=protected)
    checkpoint = checkpoint_output('cancer_binary',config['data']['magnification'],config['training']['model'],'train',experiment_id=run_id).parent
    output_path(checkpoint/'train.pth',root=checkpoint_directory('cancer_binary'),protected=[*protected,run])
    output_path(run/'run.json',root=e.RUNS,protected=[*protected,checkpoint])
    if run.exists() or checkpoint.exists():
        raise ValueError('Run/checkpoint directory exists; choose a new ID or explicit --resume')
    config_bytes = io.json_bytes(config)
    identity = dict(config_sha256=hashlib.sha256(config_bytes).hexdigest(),manifest_sha256=manifest_hash,
        preprocessing=e.PREPROCESSING, resolved_device=resolved,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda))
    inputs = dict(schema_version=SCHEMA_VERSION,identity=identity,counts=counts,code=e.revision(),
        config_source=str(source_config),config_source_sha256=config_hash,
        seeds=dict(split=json.loads(manifest_bytes).get('seed'),fitting=config['training']['seed']),
        checkpoint_directory=str(checkpoint), expected_artifacts=expected_outputs(config,run,checkpoint))
    if initialization:
        inputs.update(initialization=initialization,
            initialization_sha256=hashlib.sha256(io.json_bytes(initialization)).hexdigest())
    state = dict(schema_version=SCHEMA_VERSION,run_id=run_id,status='prepared',active_phase='development',
        prepared_at=datetime.now(timezone.utc).isoformat(),
        phases={name:dict(status='not_requested') for name in PHASES},stages={})
    run.parent.mkdir(parents=True,exist_ok=True)
    run.mkdir()  # Atomic reservation; a competing fresh run cannot replace it.
    try:
        io.write_json(run/'run.json',state,root=run,overwrite=False)
        checkpoint.mkdir(parents=True)
        io.publish(run/'config.json',config_bytes,root=run,protected=protected,overwrite=False)
        io.publish(run/'manifest.json',manifest_bytes,root=run,protected=protected,overwrite=False)
        if initialization: io.write_json(run/'initialization.json',initialization,root=run,overwrite=False)
        io.write_json(run/'inputs.json',inputs,root=run,overwrite=False)
        io.write_json(run/'results.json',dict(schema_version=SCHEMA_VERSION,run_id=run_id,
            phases={name:None for name in PHASES}),root=run,overwrite=False)
    except Exception as error:
        state.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        state['phases']['development']['status']='failed'
        try:
            io.write_json(run/'run.json',state,root=run)
        except OSError:
            pass  # Permanently unavailable storage cannot be turned into successful evidence.
        raise
    return dict(run=run,config=config,state=state,inputs=inputs)
