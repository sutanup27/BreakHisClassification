"""M1.1 contracts and role permissions; stdlib only, no folds/fits/G1 on import.

This prevents accidental pipeline misuse, not hostile arbitrary Python code.
M0's default schemas stay unchanged; M1 uses an explicit separate registry.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from . import records as r

SPEC_ID = 'M1-SPEC-001'
SPEC_VERSION = '1.0'
RECORD_SHAPES = {
    'm1.run':'data', 'm1.config':'data', 'm1.entry':'data', 'm1.permissions':'data',
    'm1.failure':'data', 'm1.operations':'items', 'm1.evidence_index':'items',
}
ROLES = frozenset({'inner_train','development_selection','development_calibration',
                  'shortlist_confirmation','outer_test','excluded','development_only'})
OPERATIONS = frozenset({'read_metadata','diagnose','plan_folds','construct_dataset','iterate_images',
    'fit','export_predictions','calibrate_threshold','select_epoch','select_candidate',
    'benchmark','validate_oof_fixture'})
SEED_NAMES = frozenset({'split','training','initialization','loader','augmentation','pruning','search'})
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_CONFIG = {'schema_version','spec_id','spec_version','mode','repo_root','data_root','output_root',
    'm0_archive','m0_index_sha256','m0_receipt_sha256','protocol_sha256','decisions_sha256',
    'source','component_ids','fold_policy','seeds','loader','device','deterministic',
    'prediction_policy','selection_policy','timeouts','predecessor','unknown_reasons'}
_ROW = {'schema_version','record_type','source_instance_id','image_id','relative_path',
        'raw_sha256','decoded_pixel_sha256','filename_group_id','component_id',
        'class_label','subtype','magnification','status','unavailable_reasons'}
_PINS = ('m0_index_sha256','m0_receipt_sha256','protocol_sha256','decisions_sha256')


class PermissionDenied(r.RecordError):
    """A valid record does not imply authority to perform the requested operation."""


def _text(value, field):
    if not isinstance(value,str) or not value.strip(): raise r.RecordError(field+' must be nonempty text')


def _digest(value):
    if not isinstance(value,str) or not _SHA.fullmatch(value): raise r.RecordError('Expected lowercase SHA-256')
    return value


def _keys(value, fields, label):
    if not isinstance(value,dict) or set(value)!=set(fields): raise r.RecordError('Invalid '+label+' fields')


def _ids(values):
    if not isinstance(values,list) or not all(isinstance(v,str) for v in values) or len(set(values))!=len(values):
        raise r.RecordError('Component IDs must be a distinct list')
    for value in values: _digest(value)


def new_record(kind,run_id,payload,*,created_at_utc=None):
    return r.new_record(kind,run_id,payload,created_at_utc=created_at_utc,registry=RECORD_SHAPES)


def validate_record(record,*,expected_type=None):
    return r.validate_record(record,expected_type=expected_type,registry=RECORD_SHAPES)


def read_record(run,relative,*,expected_type=None):
    return r.read_record(run,relative,expected_type=expected_type,registry=RECORD_SHAPES)


def write_record(run,relative,record,*,overwrite=False):
    return r.write_record(run,relative,record,overwrite=overwrite,registry=RECORD_SHAPES)


def validate_input_row(row):
    """Flat identity-only JSONL contract. Assignment/prediction rows are later packages."""
    _keys(row,_ROW,'M1 input row')
    r.canonical_json_bytes(row)
    if type(row['schema_version']) is not int or row['schema_version']!=1 or row['record_type']!='m1.input_row':
        raise r.RecordError('Unsupported M1 JSONL row type/version')
    _text(row['source_instance_id'],'source_instance_id')
    r.portable_relative_path(row['relative_path'])
    for field in ('image_id','component_id'): _digest(row[field])
    if row['status'] not in ('included','excluded'): raise r.RecordError('Invalid input status')
    if row['class_label'] not in ('benign','malignant',None) or row['magnification'] not in ('40X','100X','200X','400X',None):
        raise r.RecordError('Invalid input class/magnification')
    if not isinstance(row['unavailable_reasons'],dict): raise r.RecordError('Missing unavailable reasons')
    for field in ('raw_sha256','decoded_pixel_sha256','filename_group_id','class_label','subtype','magnification'):
        value=row[field]
        if value is None:
            _text(row['unavailable_reasons'].get(field),'Unavailable '+field)
        elif field.endswith('sha256'): _digest(value)
        else: _text(value,field)
    if row['status']=='included' and any(row[field] is None for field in ('raw_sha256','decoded_pixel_sha256','filename_group_id','class_label','magnification')):
        raise r.RecordError('Included identities require available hashes/group/class/magnification')
    return row


def identity_sha256(rows,policy):
    """Stable identity over complete sorted rows plus policy; roots/timestamps are absent."""
    if not isinstance(policy,dict): raise r.RecordError('Identity policy must be an object')
    if not isinstance(rows,list): raise r.RecordError('Rows must be a list')
    seen_ids,seen_paths=set(),set()
    for row in rows:
        validate_input_row(row)
        key=(row['source_instance_id'],row['relative_path'].casefold())
        if row['image_id'] in seen_ids or key in seen_paths: raise r.RecordError('Duplicate/case-colliding input identity')
        seen_ids.add(row['image_id']); seen_paths.add(key)
    return r.semantic_sha256({'rows':sorted(rows,key=lambda item:(item['source_instance_id'],item['relative_path'])),
                              'policy':policy},domain='m1.input_identity')


def write_input_rows(run,relative,rows,*,policy):
    digest=identity_sha256(rows,policy)
    if Path(relative).suffix!='.jsonl': raise r.RecordError('Input row destination must be JSONL')
    r.publish_bytes(run,relative,b''.join(r.canonical_json_bytes(row)+b'\n' for row in rows))
    return dict(semantic_sha256=digest,file_sha256=r.file_sha256(r.safe_child(run,relative)))


def read_input_rows(run,relative):
    rows=r.read_jsonl(r.safe_child(run,relative,must_exist=True),expected_type='m1.input_row')
    for row in rows: validate_input_row(row)
    identity_sha256(rows,{})  # Includes duplicate/case collision checks.
    return rows


def validate_config(config,*,config_file,require_pins=False):
    """No writes or archive verification; incomplete template pins are named, never active."""
    _keys(config,_CONFIG,'M1 config')
    r.canonical_json_bytes(config)
    if type(config['schema_version']) is not int or config['schema_version']!=1 or config['spec_id']!=SPEC_ID or config['spec_version']!=SPEC_VERSION:
        raise r.RecordError('Unsupported M1 schema/spec identity')
    if config['mode'] not in ('engineering','diagnostic','real_design'): raise r.RecordError('Unknown M1 mode')
    _keys(config['source'],{'kind','instance_id'},'source')
    source=config['source']
    if source['kind'] not in ('synthetic','real'): raise r.RecordError('Invalid source kind')
    _text(source['instance_id'],'source instance')
    if (config['mode']=='engineering')!=(source['kind']=='synthetic'):
        raise r.RecordError('Engineering requires synthetic identity; real metadata uses diagnostic/real_design')
    _ids(config['component_ids'])
    if not isinstance(config['unknown_reasons'],dict): raise r.RecordError('Unknown reasons required')
    for key in _PINS:
        if config[key] is None:
            _text(config['unknown_reasons'].get(key),'Unresolved '+key)
            if require_pins: raise PermissionDenied('Fill unresolved verified M0/protocol pins before active entry: '+key)
        else: _digest(config[key])
    _keys(config['fold_policy'],{'outer_folds','inner_roles','calibration_reuse','status'},'fold policy')
    folds=config['fold_policy']
    if type(folds['outer_folds']) is not int or folds['outer_folds']<2 or type(folds['calibration_reuse']) is not bool:
        raise r.RecordError('Invalid fold count/reuse policy')
    roles=folds['inner_roles']
    if not isinstance(roles,list) or not all(isinstance(role,str) for role in roles) or len(set(roles))!=len(roles) or not {'inner_train','development_selection','development_calibration'}<=set(roles) or set(roles)-{'inner_train','development_selection','development_calibration','shortlist_confirmation'}:
        raise r.RecordError('Invalid inner role policy')
    if folds['status'] not in ('synthetic_fixture','proposed','agreed'): raise r.RecordError('Invalid policy status')
    _keys(config['seeds'],SEED_NAMES,'seed namespaces')
    if any(type(v) is not int or not 0<=v<2**32 for v in config['seeds'].values()): raise r.RecordError('Seeds must be named uint32 values')
    _keys(config['loader'],{'batch_size','num_workers'},'loader')
    if type(config['loader']['batch_size']) is not int or config['loader']['batch_size']<1 or type(config['loader']['num_workers']) is not int or config['loader']['num_workers']!=0:
        raise r.RecordError('Positive batch size and workers zero required in M1.1')
    if config['device'] not in ('cpu','cuda') or type(config['deterministic']) is not bool: raise r.RecordError('Invalid device/determinism')
    _keys(config['prediction_policy'],{'class_names','scope','status'},'prediction policy')
    if config['prediction_policy']['class_names']!=['benign','malignant'] or config['prediction_policy']['scope']!='development':
        raise r.RecordError('Only binary development predictions supported')
    _keys(config['selection_policy'],{'floor','grid','tie_tolerance','status'},'selection policy')
    for item in (config['prediction_policy'],config['selection_policy']):
        if item['status'] not in ('synthetic_fixture','proposed','agreed'): raise r.RecordError('Invalid policy status')
    selection=config['selection_policy']
    for key in ('floor','tie_tolerance'):
        value=selection[key]
        if value is None: _text(config['unknown_reasons'].get('selection_policy.'+key),'Unresolved selection '+key)
        elif type(value) not in (int,float) or not 0<=value<=1: raise r.RecordError('Selection values must be fractions')
    grid=selection['grid']
    if grid is None: _text(config['unknown_reasons'].get('selection_policy.grid'),'Unresolved threshold grid')
    elif not isinstance(grid,list) or not grid or any(type(v) not in (int,float) or not 0<=v<=1 for v in grid) or len(set(grid))!=len(grid):
        raise r.RecordError('Threshold grid must contain distinct finite fractions')
    _keys(config['timeouts'],{'operation_seconds'},'timeouts')
    if type(config['timeouts']['operation_seconds']) is not int or config['timeouts']['operation_seconds']<1: raise r.RecordError('Invalid timeout')
    if config['predecessor'] is not None: r.validate_id(config['predecessor'])
    resolved=json.loads(r.canonical_json_bytes(config))
    for key in ('repo_root','output_root','m0_archive'):
        resolved[key]=str(r.resolve_config_path(config[key],config_file,kind='directory' if key=='repo_root' else None))
    if config['data_root'] is not None: resolved['data_root']=str(r.resolve_config_path(config['data_root'],config_file))
    repo=Path(resolved['repo_root']); output=Path(resolved['output_root'])
    # Normalize .. lexically before checking links; resolve() alone could conceal one.
    lexical=Path(config['output_root']).expanduser()
    if not lexical.is_absolute(): lexical=Path(config_file).resolve().parent/lexical
    lexical=Path(os.path.abspath(lexical))
    anchor=repo/'artifacts/research/m1'
    if not lexical.is_relative_to(anchor) or not output.is_relative_to(anchor): raise r.RecordError('M1 output must stay under repository artifacts/research/m1')
    r.safe_child(repo,lexical.relative_to(repo).as_posix())
    protected=[resolved['m0_archive'],repo.parent/'checkpoint']
    if resolved['data_root']: protected.append(resolved['data_root'])
    r.reject_overlap(output,protected)
    return resolved


def config_identity(config):
    """Semantic policy/config identity excludes only physical locating roots."""
    return r.semantic_sha256({key:value for key,value in config.items() if key not in ('repo_root','data_root','output_root','m0_archive')},domain='m1.config')


class VerifiedEntry:
    """Read-only verified metadata. Constructor never accepts copied approval flags."""
    __slots__=('_root','_proof')

    def __init__(self,archive,*,expected_index_sha256,expected_receipt_sha256):
        from .runner import verify_run
        archive=Path(archive).resolve()
        _digest(expected_index_sha256); _digest(expected_receipt_sha256)
        expected={'acceptance/evidence_index.json':expected_index_sha256,'acceptance/receipt.json':expected_receipt_sha256}
        for path,digest in expected.items():
            if r.file_sha256(r.safe_child(archive,path,must_exist=True))!=digest: raise r.IntegrityError('M0 expected pin differs: '+path)
        verified=verify_run(archive)  # exit_code 1 may mean valid but scientifically blocked.
        gate=verified['gate']
        handoff=r.read_record(archive,'acceptance/m1_handoff.json',expected_type='m1_handoff')['data']
        eligibility=r.read_record(archive,'governance/component_eligibility.json')['items']
        scope=r.read_record(archive,'inputs/pilot_scope.json')['data']
        components=r.read_record(archive,'audit/duplicate_components.json')['items']
        protocol=r.read_record(archive,'governance/protocol.initial.json')['data']
        exposure=r.read_record(archive,'governance/exposure_summary.json')['data']
        bindings=dict(m0_run_id=verified['run_id'],m0_index_sha256=expected_index_sha256,m0_receipt_sha256=expected_receipt_sha256,
            source_snapshot_sha256=handoff['source_snapshot_sha256'],inventory_sha256=handoff['inventory_sha256'],
            source_instance_id=exposure['source_instance_id'])
        for name,path in [('components_sha256','audit/duplicate_components.json'),('exposure_sha256','governance/exposure_register.json'),
                          ('eligibility_sha256','governance/component_eligibility.json'),('protocol_sha256','governance/protocol.initial.json'),
                          ('decisions_sha256','governance/decisions.json')]:
            bindings[name]=r.file_sha256(r.safe_child(archive,path,must_exist=True))
        proof=dict(bindings=bindings,restrictions=gate['restrictions'],engineering_allowed=gate['m1_engineering_allowed'] and bool(handoff['allowed_engineering']),
            design_allowed=gate['dimensions']['confirmatory_design']['status'] in ('ready','restricted') and handoff['run_kind']=='real',
            gpu_ready=gate['dimensions']['gpu']['status']=='ready',component_ids=sorted(c['component_id'] for c in components),
            candidate_ids=sorted({item['component_id'] for item in eligibility if item['status']=='candidate_untouched_with_reviewed_coverage'}-set(scope['component_ids'])),
            protocol=protocol,parent_run_kind=handoff['run_kind'])
        for path,digest in expected.items():
            if r.file_sha256(archive/path)!=digest: raise r.IntegrityError('M0 moved during entry verification')
        object.__setattr__(self,'_root',str(archive)); object.__setattr__(self,'_proof',r.canonical_json_bytes(proof))

    def __setattr__(self,name,value): raise AttributeError('Verified entry is immutable')

    @property
    def snapshot(self): return json.loads(self._proof)

    @property
    def bindings(self): return self.snapshot['bindings']

    def reverify(self):
        """Recheck before an operation boundary, never once per image/batch."""
        from .runner import verify_run
        pins=self.bindings
        for relative,key in [('acceptance/evidence_index.json','m0_index_sha256'),('acceptance/receipt.json','m0_receipt_sha256')]:
            if r.file_sha256(r.safe_child(self._root,relative,must_exist=True))!=pins[key]: raise r.IntegrityError('M0 entry changed since verification')
        verify_run(self._root)


def inspect_m0(archive):
    """Discover pins only after full verification; returned dictionaries grant no access."""
    archive=Path(archive).resolve()
    entry=VerifiedEntry(archive,expected_index_sha256=r.file_sha256(archive/'acceptance/evidence_index.json'),
                        expected_receipt_sha256=r.file_sha256(archive/'acceptance/receipt.json'))
    return entry.bindings


def authorize(entry,*,mode,source_kind,source_instance_id,operation,role=None,component_ids=(),device='cpu'):
    if not isinstance(entry,VerifiedEntry): raise PermissionDenied('A verified M0 entry object is required, not copied flags')
    if mode not in ('engineering','diagnostic','real_design') or operation not in OPERATIONS or (role is not None and role not in ROLES):
        raise PermissionDenied('Unknown operation, mode or role')
    _text(source_instance_id,'source instance'); _ids(list(component_ids))
    if source_kind not in ('synthetic','real') or device not in ('cpu','cuda'): raise PermissionDenied('Invalid source/device')
    proof=entry.snapshot
    entry.reverify()
    if not proof['engineering_allowed']: raise PermissionDenied('M0 engineering entry is blocked')
    if (mode=='engineering')!=(source_kind=='synthetic'): raise PermissionDenied('Mode/source identity mismatch')
    if source_kind=='synthetic' and source_instance_id==proof['bindings']['source_instance_id']:
        raise PermissionDenied('Synthetic identity collides with the real M0 source instance')
    if source_kind=='synthetic' and set(component_ids)&set(proof['component_ids']):
        raise PermissionDenied('Synthetic component scope collides with real M0 metadata')
    if source_kind=='real' and source_instance_id!=proof['bindings']['source_instance_id']:
        raise PermissionDenied('Real source instance differs from verified M0')
    if source_kind=='real' and not set(component_ids)<=set(proof['component_ids']): raise PermissionDenied('Unknown real component boundary')
    if mode=='real_design' and (not proof['design_allowed'] or not component_ids or not set(component_ids)<=set(proof['candidate_ids'])):
        raise PermissionDenied('Real design needs reviewed candidate boundary and eligible G0')
    if operation in ('read_metadata','diagnose'):
        if role is not None: raise PermissionDenied('Metadata access has no image role')
    elif operation=='plan_folds':
        if mode=='diagnostic' or role is not None: raise PermissionDenied('Diagnostics cannot create active assignments')
    elif operation=='validate_oof_fixture':
        if mode!='engineering' or role not in (None,'outer_test'): raise PermissionDenied('Only stored synthetic OOF fixtures may be validated')
    elif operation=='benchmark':
        if mode!='engineering' or role is not None: raise PermissionDenied('M1 benchmarks use synthetic resident tensors, not datasets')
        if device=='cuda' and not proof['gpu_ready']: raise PermissionDenied('GPU execution unvalidated')
    else:
        if mode!='engineering' or role in (None,'outer_test','excluded','development_only'):
            raise PermissionDenied('M1 denies real images/fits and outer-test data access')
        allowed={'fit':{'inner_train'},'calibrate_threshold':{'development_calibration'},
                 'select_epoch':{'development_selection'},'select_candidate':{'development_selection','shortlist_confirmation'},
                 'export_predictions':{'development_selection','development_calibration','shortlist_confirmation'}}
        if operation in allowed and role not in allowed[operation]: raise PermissionDenied('Operation uses the wrong development role')
        if device=='cuda' and not proof['gpu_ready']: raise PermissionDenied('GPU execution unvalidated; CPU engineering remains available')
    return dict(allowed=True,operation=operation,role=role,mode=mode,source_kind=source_kind,
                source_instance_id=source_instance_id,restrictions=proof['restrictions'],bindings=proof['bindings'],
                scientific_authorization=False)


def guarded_call(entry,*,factory,mode,source_kind,source_instance_id,operation,role=None,component_ids=(),device='cpu'):
    """Check before invoking caller's dataset/iterator/export adapter; no outer loader is constructed."""
    authorize(entry,mode=mode,source_kind=source_kind,source_instance_id=source_instance_id,
              operation=operation,role=role,component_ids=component_ids,device=device)
    if not callable(factory): raise r.RecordError('Adapter must be callable')
    return factory()


def open_entry(config):
    for key in _PINS: _digest(config[key])
    entry=VerifiedEntry(config['m0_archive'],expected_index_sha256=config['m0_index_sha256'],expected_receipt_sha256=config['m0_receipt_sha256'])
    for key in ('protocol_sha256','decisions_sha256'):
        if config[key]!=entry.bindings[key]: raise r.IntegrityError('Pinned governance differs: '+key)
    authorize(entry,mode=config['mode'],source_kind=config['source']['kind'],source_instance_id=config['source']['instance_id'],
              operation='read_metadata',component_ids=config['component_ids'])
    if config['mode']=='real_design':
        evaluation=next(item for item in entry.snapshot['protocol']['topics'] if item['topic']=='evaluation')
        if config['fold_policy']['status']!='agreed' or config['fold_policy']['outer_folds']!=evaluation['value']['outer_folds']:
            raise PermissionDenied('Real fold choices must match the agreed parent policy')
    return entry


def create_run(config,run_id):
    """Create contracts only, not M1 orchestration, folds or G1. Config must be normalized."""
    r.validate_id(run_id)
    # Revalidate with absolute paths so hand-built config cannot bypass confinement.
    config=validate_config(config,config_file=Path(config['repo_root'])/'m1.config.json',require_pins=True)
    entry=open_entry(config)
    protected=[config['m0_archive'],Path(config['repo_root']).parent/'checkpoint']
    if config['data_root']: protected.append(config['data_root'])
    run=r.create_run_directory(config['output_root'],run_id,protected_roots=protected)
    try:
        write_record(run,'run.json',new_record('m1.run',run_id,dict(spec_id=SPEC_ID,spec_version=SPEC_VERSION,state='created',
            mode=config['mode'],config_identity=config_identity(config),predecessor=config['predecessor'],
            current_source=dict(status='pending',reason='Current M1 capture/orchestration belongs to M1.6'),scientific_authorization=False)))
        write_record(run,'config.resolved.json',new_record('m1.config',run_id,config))
        write_record(run,'inputs/m0.entry.json',new_record('m1.entry',run_id,entry.snapshot))
    except Exception as error:
        write_record(run,'failure.json',new_record('m1.failure',run_id,dict(error_type=type(error).__name__,message=str(error),state='failed')))
        if (run/'run.json').exists():
            record=read_record(run,'run.json',expected_type='m1.run'); record['data']['state']='failed'
            write_record(run,'run.json',record,overwrite=True)
        raise
    return run,entry
