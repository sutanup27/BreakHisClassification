"""M0 orchestration and read-only evidence verification; no scientific fits."""
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import re
import sys
import tempfile
import time

from . import records as r, m0, protocol as p, exposure as e
from .data_readiness import lineage_record, _components
from .gates import evaluate_gate

RECORDS = {
    'run.json':'run', 'config.resolved.json':'config.resolved',
    'baseline/git.json':'git', 'baseline/source_inventory.json':'source_inventory',
    'baseline/consistency.json':'git', 'baseline/final_consistency.json':'git',
    'baseline/environment.cpu.json':'environment', 'baseline/environment.cuda.json':'environment',
    'baseline/checks.json':'checks', 'logs/operations.json':'operations',
    'audit/audit.json':'audit', 'audit/manifest.json':'manifest',
    'audit/duplicate_components.json':'duplicate_components', 'audit/group_summary.json':'group_summary',
    'audit/lineage.json':'lineage', 'audit/near_duplicate_review.json':'near_duplicate_review',
    'inputs/lineage.json':'lineage','inputs/exposure.json':'exposure.input',
    'inputs/protocol.json':'protocol.initial','inputs/decisions.json':'decisions',
    'inputs/literature.json':'literature_matrix','inputs/pilot_scope.json':'pilot_scope',
    'governance/exposure_register.json':'exposure_register',
    'governance/exposure_summary.json':'exposure_summary',
    'governance/component_eligibility.json':'component_eligibility',
    'governance/protocol.initial.json':'protocol.initial','governance/decisions.json':'decisions',
    'governance/literature_matrix.json':'literature_matrix','governance/protocol_review.json':'protocol_review',
    'acceptance/tests.json':'acceptance_tests', 'failure.json':'failure',
}
FINALS = {'acceptance/evidence_index.json','acceptance/gate_g0.json',
          'acceptance/M0_REPORT.md','acceptance/m1_handoff.json','acceptance/receipt.json'}
OPTIONAL = {'failure.json'}
MAPPING = {
 'M0-T01':'test_t01_clean_reconstructible_and_stable_identity',
 'M0-T02':'test_t02_staged_unstaged_untracked_deleted_and_binary',
 'M0-T03':'test_t03_excludes_data_weights_environments_generated_and_personal',
 'M0-T04':'test_t04_repeated_movement_fails_and_retains_evidence',
 'M0-T05':'test_existing_symlink_and_broken_symlink_rejected_when_supported',
 'M0-T06':'test_t06_full_suffix_all_magnifications_dimensions_and_input_unchanged',
 'M0-T07':'test_t07_alternate_encoding_exact_link_and_stable_representative',
 'M0-T08':'test_t08_duplicate_label_conflict_blocks_without_silent_label_fix',
 'M0-T09':'test_t09_corrupt_bad_name_folder_and_unreadable_are_all_accounted',
 'M0-T10':'test_t10_missing_and_empty_sources_have_blocked_records',
 'M0-T11':'test_t11_exact_log_filter_propagates_all_linked_magnifications',
 'M0-T12':'test_t12_no_per_image_log_is_conservative_selected_partition_superset',
 'M0-T13':'test_t13_missing_manifest_blocks_all_candidate_untouched_claims',
 'M0-T14':'test_t14_synthetic_realistic_names_do_not_quarantine_real_components',
 'M0-T15':'test_t15_unknown_lineage_and_identity_stays_unknown',
 'M0-T16':'test_t16_template_preserves_proposals_unknown_numbers_and_draft_lifecycle',
 'M0-T17':'test_t17_missing_owner_deadline_or_evidence_rejected',
 'M0-T18':'test_t18_abstract_code_and_unavailable_cannot_be_full_method_reviews',
 'M0-T19':'test_t19_cpu_pass_gpu_failure_separate_restrictions',
 'M0-T20':'test_t20_timed_out_regression_blocks_engineering',
 'M0-T21':'test_t21_tampering_and_incomplete_index_rejected',
 'M0-T22':'test_t22_relocation_is_read_only_and_preserves_verification',
 'M0-T23':'test_t23_isolated_orchestration_and_scientific_blockers',
 'M0-T24':'test_t24_verifier_never_executes_or_writes',
 'M0-T25':'test_all_supported_commands_offer_help',
}


def _now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def _blob(run, relative, data):
    path=r.safe_child(run,relative)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name); stream.write(data); stream.flush(); os.fsync(stream.fileno())
        r.safe_child(run,relative)
        os.link(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def _get(run,relative): return r.read_record(run,relative,expected_type=RECORDS[relative])[r.RECORD_SHAPES[RECORDS[relative]]]


def _put(run,relative,run_id,payload,overwrite=False):
    return r.write_record(run,relative,r.new_record(RECORDS[relative],run_id,payload),overwrite=overwrite)


def load_inputs(config_file):
    config_file=Path(config_file).resolve()
    original=r.read_json(config_file)
    config=r.validate_m0_config(original,config_file=config_file)
    result={}
    for name in ('lineage','exposure','protocol','decisions','literature'):
        path=Path(config[name+'_path'])
        result[name]=r.read_json(path)
    lineage_record(result['lineage']); p.validate_protocol(result['protocol'])
    p.validate_decisions(result['decisions'],result['protocol']); p.validate_literature(result['literature'])
    history=result['exposure']
    fields={'schema_version','source_instance_id','history_root','events','history_coverage','equivalences','unknown_reason'}
    if not isinstance(history,dict) or set(history)-{'pilot_scope'}!=fields or type(history['schema_version']) is not int or history['schema_version']!=1:
        raise r.RecordError('Invalid exposure input scaffold')
    if not isinstance(history['events'],list) or not isinstance(history['equivalences'],list): raise r.RecordError('Invalid history collections')
    for event in history['events']: e.validate_event(event)
    if history['source_instance_id'] is None or history['history_root'] is None:
        if history['events'] or history['history_coverage'] is not None or history['equivalences']:
            raise r.RecordError('Historical assertions require explicit source instance and metadata root')
        if not isinstance(history['unknown_reason'],str) or not history['unknown_reason'].strip(): raise r.RecordError('Unknown history needs reason')
    else:
        e._text(history['source_instance_id'],'Source instance')
        r.resolve_config_path(history['history_root'],config['exposure_path'],kind='directory')
    result['pilot_scope']=history.get('pilot_scope',dict(status='unreviewed',reviewer=None,component_ids=[],
        lineage_exception=False,claims=[],evidence_refs=[],rationale='No reviewed real-fit scope supplied',training_device=None))
    _scope(result['pilot_scope'])
    # Capture actual bytes, and refuse inputs that moved while being decoded.
    raw={'config':config_file.read_bytes()}
    if json.loads(raw['config'].decode('utf-8-sig'),object_pairs_hook=r._unique_object)!=original: raise r.IntegrityError('Config changed while loading')
    for name in ('lineage','exposure','protocol','decisions','literature'):
        raw[name]=Path(config[name+'_path']).read_bytes()
        if json.loads(raw[name].decode('utf-8-sig'),object_pairs_hook=r._unique_object)!=result[name]: raise r.IntegrityError('Input changed while loading')
    return config,result,raw


def _scope(scope):
    fields={'status','reviewer','component_ids','lineage_exception','claims','evidence_refs','rationale','training_device'}
    if not isinstance(scope,dict) or set(scope)-{'university_pilot'}!=fields or scope['status'] not in ('reviewed','unreviewed'):
        raise r.RecordError('Invalid pilot scope declaration')
    if type(scope['lineage_exception']) is not bool or not isinstance(scope['component_ids'],list) or not all(isinstance(c,str) for c in scope['component_ids']): raise r.RecordError('Invalid pilot component scope')
    if len(set(scope['component_ids']))!=len(scope['component_ids']): raise r.RecordError('Duplicate pilot component')
    if not isinstance(scope['claims'],list) or not all(isinstance(v,str) for v in scope['claims']) or not isinstance(scope['evidence_refs'],list): raise r.RecordError('Invalid scope evidence/claims')
    if scope['training_device'] not in (None,'cpu','cuda'): raise r.RecordError('Invalid scope training device')
    if scope['status']=='reviewed':
        e._text(scope['reviewer'],'Pilot reviewer'); e._text(scope['rationale'],'Pilot scope rationale')
        if not scope['component_ids'] or not scope['evidence_refs'] or scope['training_device'] not in ('cpu','cuda') or not {'exploratory-only','filename-group-only','derivative-only'}<=set(scope['claims']):
            raise r.RecordError('Reviewed pilot requires explicit components, device, claims and evidence')
    if 'university_pilot' in scope:
        declaration=scope['university_pilot']
        expected={'version':1,'dataset_source':'researcher-declared-breakhis-website',
                  'prior_use':'acknowledged','workflow':'fresh-no-reuse',
                  'evaluation_claim':'within-study-held-out','protocol':'university-pilot-v1'}
        if not isinstance(declaration,dict) or declaration!=expected or type(declaration.get('version')) is not int:
            raise r.RecordError('Invalid university pilot v1 declaration')
        if scope['status']!='reviewed' or not scope['lineage_exception']:
            raise r.RecordError('University pilot requires reviewed declared-lineage scope')
    return scope


def _history_refs(history,scope):
    refs=[]
    for event in history['events']:
        refs.append(event['artifact'])
        if event['source_kind']=='real': refs.append(event['manifest'])
    coverage=history['history_coverage']
    if isinstance(coverage,dict): refs.extend(coverage.get('evidence_refs',[]))
    for bridge in history['equivalences']: refs.extend(bridge.get('evidence_refs',[]))
    refs.extend(scope['evidence_refs'])
    return refs


def _trace(log, real_run):
    rows=[dict(test_id=key,status='pass' if re.search(r'^'+re.escape(method)+r' \([^\n]+\) \.\.\. ok\s*$',log,re.M) else 'not_run',
               test_method=method,evidence_ref='logs/regressions.stderr.log') for key,method in MAPPING.items()]
    rows.append(dict(test_id='M0-T26',status='pass' if real_run else 'not_run',test_method='fresh real local pipeline',
                     evidence_ref='audit/audit.json'))
    return rows


def _final_source(run,config):
    source=_get(run,'baseline/git.json')
    state=m0._source_state(Path(config['repo_root']),run)
    unchanged=state['entries']==_get(run,'baseline/source_inventory.json')
    unchanged &= all(state[key]==source[key] for key in ('head','ref','status_porcelain_z','index_sha256','unresolved_lfs','unresolved_submodules'))
    unchanged &= state['patch']==(run/'baseline/tracked.patch').read_bytes()
    unchanged &= state['staged_patch']==(run/'baseline/staged.patch').read_bytes()
    return dict(status='passed' if unchanged else 'failed',reason=None if unchanged else 'Source moved during M0',checked_at_utc=_now())


def build_index(run,run_id):
    entries=[]
    for path in sorted(run.rglob('*')):
        relative=path.relative_to(run).as_posix()
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)(): raise r.IntegrityError('Evidence links forbidden')
        if not path.is_file() or relative in FINALS: continue
        kind=RECORDS.get(relative,'blob')
        producer='baseline' if relative.startswith('baseline/') else 'data-audit' if relative.startswith(('audit/','logs/data-audit.')) else 'governance' if relative.startswith('governance/') else 'm0-evidence'
        entries.append(r.artifact_entry(run,relative,record_type=kind,operation_id=producer,
            schema_version=1 if relative in RECORDS else None,
            status='partial' if relative.startswith(('baseline/attempt-1/','baseline/attempt-2/')) else 'complete',
            row_record_type='image_inventory_row' if relative in ('audit/inventory.jsonl','audit/exclusions.jsonl') else None))
    index=r.new_record('evidence_index',run_id,entries)
    r.write_record(run,'acceptance/evidence_index.json',index)
    r.verify_evidence_index(run,index,require_complete=False)
    return r.file_sha256(run/'acceptance/evidence_index.json')


def _facts(run):
    config=_get(run,'config.resolved.json')
    source=_get(run,'baseline/git.json')
    checks=_get(run,'baseline/checks.json')
    cpu=[check for check in checks if check['device']=='cpu']
    baseline=bool(cpu) and all(check['status']=='passed' and check['return_code']==0 for check in cpu)
    consistency=all(_get(run,path)['status']=='passed' for path in ('baseline/consistency.json','baseline/final_consistency.json'))
    consistency &= not source['unresolved_lfs'] and not source['unresolved_submodules']
    audit=_get(run,'audit/audit.json'); lineage=_get(run,'audit/lineage.json')
    review=_get(run,'governance/exposure_summary.json'); protocol=_get(run,'governance/protocol.initial.json')
    literature=p.validate_literature(_get(run,'governance/literature_matrix.json'))
    scope=_get(run,'inputs/pilot_scope.json')
    components=_get(run,'audit/duplicate_components.json')
    ids={c['component_id'] for c in components}
    scoped=scope['status']=='reviewed' and bool(scope['component_ids']) and set(scope['component_ids'])<=ids
    for ref in scope['evidence_refs']:
        try: e._reference(ref,run/'inputs/history',run,allow_archive=True)
        except (OSError,r.RecordError): scoped=False
    eligible=_get(run,'governance/component_eligibility.json')
    candidates={row['component_id'] for row in eligible if row['status']=='candidate_untouched_with_reviewed_coverage'}-set(scope['component_ids'])
    labels=Counter(c['class_labels'][0] for c in components if c['component_id'] in candidates and len(c['class_labels'])==1)
    topics={item['topic']:item for item in protocol['topics']}
    folds=topics['evaluation']['value']['outer_folds']
    training_text=str(topics['hardware']['value']['training']).casefold()
    planned_device='cuda' if any(word in training_text for word in ('cuda','gpu','rtx')) else 'cpu' if 'cpu' in training_text else None
    if scoped and planned_device and scope['training_device']!=planned_device: scoped=False
    critical_lineage=('source_release','derivative_parent','transformation_parameters','transformation_code_history','usage_terms')
    facts=dict(baseline_passed=baseline,source_consistent=bool(consistency),evidence_complete=True,
        data_ready=audit['data_integrity_ready'],lineage_known=all(lineage['facts'][field]['status']=='verified' for field in critical_lineage),
        lineage_scope_approved=scoped and scope['lineage_exception'],history_ready=review['history_review_ready'],
        scope_reviewed=scoped,protocol_pilot_ready=protocol['lifecycle'] in ('pilot_ready','initial_design_reviewed'),
        protocol_design_ready=protocol['lifecycle']=='initial_design_reviewed',
        owners_assigned=protocol['reviewer'] not in (None,'research_owner','reviewer') and all(t['owner'] not in ('research_owner','reviewer') for t in protocol['topics']),
        literature_complete=literature['novelty_review_complete'],
        gpu_ready=any(check['check_id']=='environment-cuda' and check['status']=='passed' and check['return_code']==0 for check in checks),
        training_device=scope['training_device'] if scoped else planned_device,fold_support=type(folds) is int and min(labels.get('benign',0),labels.get('malignant',0))>=folds,
        real_input=config['run_kind']=='real',acceptance_complete=all(row['status']=='pass' for row in _get(run,'acceptance/tests.json')))
    # Opt-in only: legacy archives recompute their original gate without new fields.
    # The profile is indexed, schema checked and hash-bound to preserved evidence.
    if 'university_pilot' in scope:
        _scope(scope)
        facts['university_pilot_reviewed']=scoped and scope['reviewer'] not in ('research_owner','reviewer')
    return facts


def _handoff(run,gate,index_hash):
    config=_get(run,'config.resolved.json'); source=_get(run,'baseline/git.json') if (run/'baseline/git.json').exists() else {}
    engineering=gate['m1_engineering_allowed']
    audit=_get(run,'audit/audit.json') if (run/'audit/audit.json').exists() else {}
    groups=_get(run,'audit/group_summary.json') if (run/'audit/group_summary.json').exists() else {}
    log=(run/'logs/regressions.stderr.log').read_text(encoding='utf-8',errors='replace') if (run/'logs/regressions.stderr.log').exists() else ''
    count=re.search(r'Ran (\d+) tests? in ([0-9.]+)s',log)
    operations=_get(run,'logs/operations.json') if (run/'logs/operations.json').exists() else []
    versions={path:r.file_sha256(run/path) for path in ('audit/duplicate_components.json','governance/component_eligibility.json',
        'governance/protocol.initial.json','governance/decisions.json','governance/literature_matrix.json') if (run/path).exists()}
    return dict(evidence_index_sha256=index_hash,source_snapshot_sha256=source.get('source_snapshot_sha256'),
        source_head=source.get('head'),source_dirty=source.get('dirty'),
        inventory_sha256=r.file_sha256(run/'audit/inventory.jsonl') if (run/'audit/inventory.jsonl').exists() else None,
        allowed_engineering=['CPU unit/interface development','Frozen inventory diagnostics; no untouched assignment inferred'] if engineering else [],
        allowed_scientific_component_ids=_get(run,'inputs/pilot_scope.json')['component_ids'] if gate['real_fit_scope_allowed'] else [],
        research_mode=config['mode'],overall_status=gate['overall_status'],restrictions=gate['restrictions'],
        run_kind=config['run_kind'],data_root=config['data_root'],data_counts=audit.get('counts'),filename_groups=groups.get('filename_groups'),
        regressions=dict(tests=int(count[1]),seconds=float(count[2])) if count else dict(reason='Regression stage did not complete'),
        operations=[dict(operation_id=item['operation_id'],status=item['status']) for item in operations],governance_sha256=versions,
        claims=['filename-group terminology only','No established clinical safety, novelty or independent-data generalization'],
        evidence_refs=['governance/component_eligibility.json','governance/protocol.initial.json','governance/decisions.json'])


def _report(run_id,gate,handoff):
    lines=[f'# M0 report: {run_id}', '',f"Requested mode: {gate['requested_mode']}. Overall: **{gate['overall_status']}**.",'',
        'Evidence generation and scientific eligibility are separate. No model fitting or held-out inference was launched.', '',
        '| Dimension | Status | Reasons |','| --- | --- | --- |']
    for key,value in sorted(gate['dimensions'].items()): lines.append(f"| {key} | {value['status']} | {', '.join(value['reason_codes']) or 'none'} |")
    lines += ['', '## M1 handoff','',f"Engineering allowed: {gate['m1_engineering_allowed']}.",
        f"Real fit scope allowed: {gate['real_fit_scope_allowed']}.",f"Source snapshot: `{handoff['source_snapshot_sha256']}`.",
        '', '## Restrictions','']+[f'- {code}' for code in gate['restrictions']]
    lines += ['', '## Measured evidence','',f"Run kind: {handoff['run_kind']}. Data root: `{handoff['data_root']}`.",
        f"Image counts: {handoff['data_counts'] if handoff['data_counts'] is not None else 'unavailable (audit did not complete)' }.",
        f"Filename groups: {handoff['filename_groups'] if handoff['filename_groups'] is not None else 'unavailable' }.",
        f"Current regression result: {handoff['regressions']}.",f"Source HEAD: `{handoff['source_head']}`; dirty: {handoff['source_dirty']}.",
        '', '## Operations','']+[f"- {item['operation_id']}: {item['status']}" for item in handoff['operations']]
    lines += ['', 'Clinical identity, novelty and scientific agreement are not established by successful software checks.',
              'See `tests.json`, `m1_handoff.json`, `gate_g0.json` and `evidence_index.json` in this directory for traceability.','']
    return '\n'.join(lines)


def run_m0(config_file,run_id):
    r.validate_id(run_id)
    config,inputs,raw=load_inputs(config_file)
    run=r.create_run_directory(config['output_root'],run_id,protected_roots=[config['data_root']])
    operations=[]
    run_tick=time.monotonic()
    descriptor=dict(state='created',started_at_utc=_now(),ended_at_utc=None,spec_id=r.SPEC_ID,spec_version=r.SPEC_VERSION,
        runner_source_root=str(Path(__file__).resolve().parents[2]),
        spec_sha256=None,config_file=str(Path(config_file).resolve()),requested_mode=config['mode'],run_kind=config['run_kind'],
        config_sha256=__import__('hashlib').sha256(raw['config']).hexdigest(),predecessor=None,
        gate_ref='acceptance/gate_g0.json',report_ref='acceptance/M0_REPORT.md',input_files=[])
    _put(run,'run.json',run_id,descriptor)
    _put(run,'config.resolved.json',run_id,config)
    try:
        for name,data in raw.items():
            relative=f'inputs/{name}.input'; _blob(run,relative,data)
            descriptor['input_files'].append(dict(path=relative,sha256=r.file_sha256(run/relative)))
        for name in ('lineage','exposure','protocol','decisions','literature','pilot_scope'):
            _put(run,f'inputs/{name}.json',run_id,inputs[name])
        spec=Path(__file__).resolve().parents[2]/'docs/specs/m0/M0_SPEC.md'
        _blob(run,'inputs/M0_SPEC.md',spec.read_bytes()); descriptor['spec_sha256']=r.file_sha256(run/'inputs/M0_SPEC.md')
        history=inputs['exposure']
        history_root=r.resolve_config_path(history['history_root'],config['exposure_path'],kind='directory') if history['history_root'] else Path(config['exposure_path']).parent
        r.safe_child(run,'inputs/history').mkdir()
        _blob(run,'inputs/history/.m0-history-anchor',b'Frozen history metadata. An empty history is not evidence of no exposure.\n')
        for ref in _history_refs(history,inputs['pilot_scope']):
            try:
                path=e._reference(ref,history_root,run)
                relative='inputs/history/'+ref['path']
                if not r.safe_child(run,relative).exists(): _blob(run,relative,path.read_bytes())
                if r.file_sha256(run/relative)!=ref['sha256']: raise r.IntegrityError('History moved during capture')
            except (OSError,r.RecordError):
                # The resolver records unresolved evidence; no empty successful substitute.
                continue
        descriptor['state']='running'; _put(run,'run.json',run_id,descriptor,True)
        started=_now(); tick=time.monotonic()
        operations.append(dict(operation_id='baseline',function='capture_baseline',started_at_utc=started,status='running'))
        _put(run,'logs/operations.json',run_id,operations,True)
        baseline=m0.capture_baseline(config['repo_root'],run,run_id,devices=config['check_devices'],
            environment_seconds=config['timeouts']['environment_seconds'],regression_seconds=config['timeouts']['regression_seconds'])
        operations[-1].update(ended_at_utc=_now(),elapsed_seconds=time.monotonic()-tick,status='passed' if baseline['cpu_engineering_ready'] else 'failed')
        _put(run,'logs/operations.json',run_id,operations,True)
        if any(check['status']=='interrupted' for check in baseline['checks']):
            raise InterruptedError('Baseline interrupted by operator; audit was not started')
        worker=Path(__file__).resolve().parents[2]/'tools/m0_audit_worker.py'
        command=[sys.executable,str(worker),'--run-dir',str(run)]
        audit_op=m0._execute(run,'data-audit',command,config['repo_root'],'cpu',config['timeouts']['audit_seconds'],dict(os.environ,PYTHONUTF8='1'))
        audit_op['input_hashes'].update({path:r.file_sha256(run/path) for path in ('config.resolved.json','inputs/lineage.json')})
        operations.append(audit_op); _put(run,'logs/operations.json',run_id,operations,True)
        if audit_op['status']!='passed': raise r.IntegrityError('Data audit failed/interrupted/timed out; retained worker logs')
        instance=history['source_instance_id'] or 'unreviewed-'+r.semantic_sha256(config['data_root'],domain='m0.unreviewed_source')[:20]
        started=_now(); tick=time.monotonic()
        e.resolve_exposure(run,run_id,history_root=run/'inputs/history',source_instance_id=instance,
            events=history['events'],history_coverage=history['history_coverage'],equivalences=history['equivalences'],_archive=True)
        p.write_protocol_records(run,run_id,protocol=inputs['protocol'],decisions=inputs['decisions'],literature=inputs['literature'])
        operations.append(dict(operation_id='governance',function='resolve_exposure/write_protocol_records',started_at_utc=started,
            ended_at_utc=_now(),elapsed_seconds=time.monotonic()-tick,status='passed'))
        _put(run,'logs/operations.json',run_id,operations,True)
        _put(run,'baseline/final_consistency.json',run_id,_final_source(run,config))
        log=(run/'logs/regressions.stderr.log').read_text(encoding='utf-8',errors='replace')
        _put(run,'acceptance/tests.json',run_id,_trace(log,config['run_kind']=='real'))
        descriptor.update(state='completed',ended_at_utc=_now())
        operations.append(dict(operation_id='m0-evidence',function='run_m0',started_at_utc=descriptor['started_at_utc'],
            ended_at_utc=descriptor['ended_at_utc'],elapsed_seconds=time.monotonic()-run_tick,status='passed'))
        _put(run,'logs/operations.json',run_id,operations,True)
        _put(run,'run.json',run_id,descriptor,True)
        verify_semantics(run)
        index_hash=build_index(run,run_id)
        gate=evaluate_gate(_facts(run),config['mode'],index_hash)
    except (Exception,KeyboardInterrupt) as error:
        for operation in operations:
            if operation['status']=='running': operation.update(status='failed',ended_at_utc=_now(),elapsed_seconds=time.monotonic()-tick)
        descriptor.update(state='failed',ended_at_utc=_now())
        operations.append(dict(operation_id='m0-evidence',function='run_m0',started_at_utc=descriptor['started_at_utc'],
            ended_at_utc=descriptor['ended_at_utc'],elapsed_seconds=time.monotonic()-run_tick,status='failed'))
        _put(run,'logs/operations.json',run_id,operations,True)
        _put(run,'run.json',run_id,descriptor,True)
        _put(run,'failure.json',run_id,dict(error_type=type(error).__name__,message=str(error) or 'Interrupted by operator',owner='research_owner',
            resolution='Repair the failure and use a new run ID; partial runs are never resumed'))
        if not (run/'logs/operations.json').exists(): _put(run,'logs/operations.json',run_id,operations)
        # A generation failure is never converted into a successful empty stage.
        if (run/'acceptance/evidence_index.json').exists():
            (run/'acceptance/evidence_index.json').rename(run/'acceptance/evidence_index.failed.json')
        index_hash=build_index(run,run_id)
        facts={key:False for key in ('baseline_passed','source_consistent','evidence_complete','data_ready','lineage_known',
            'lineage_scope_approved','history_ready','scope_reviewed','protocol_pilot_ready','protocol_design_ready',
            'owners_assigned','literature_complete','gpu_ready','fold_support','real_input','acceptance_complete')}
        facts['training_device']=None
        gate=evaluate_gate(facts,config['mode'],index_hash)
    handoff=_handoff(run,gate,index_hash)
    r.write_record(run,'acceptance/gate_g0.json',r.new_record('gate_g0',run_id,gate))
    r.write_record(run,'acceptance/m1_handoff.json',r.new_record('m1_handoff',run_id,handoff))
    _blob(run,'acceptance/M0_REPORT.md',_report(run_id,gate,handoff).encode('utf-8'))
    receipt={relative:r.file_sha256(run/relative) for relative in sorted(FINALS-{'acceptance/receipt.json'})}
    r.write_record(run,'acceptance/receipt.json',r.new_record('completion_receipt',run_id,dict(artifacts=receipt,state=descriptor['state'])))
    verified=verify_run(run)['validated'] if descriptor['state']=='completed' else False
    return dict(run_dir=str(run),state=descriptor['state'],validated=verified,gate=gate,exit_code=1 if descriptor['state']!='completed' or gate['overall_status']=='blocked' else 0)


def verify_semantics(run):
    """Reconcile stored domain records using only indexed/local bytes, without IO writes."""
    run=Path(run)
    for relative in set(RECORDS)-OPTIONAL:
        if not r.safe_child(run,relative).is_file(): raise r.IntegrityError('Required evidence missing: '+relative)
    descriptor=_get(run,'run.json'); run_id=r.read_record(run,'run.json')['run_id']
    if descriptor['state']!='completed': raise r.IntegrityError('Run is partial or failed')
    config=_get(run,'config.resolved.json')
    if descriptor['requested_mode']!=config['mode'] or descriptor['run_kind']!=config['run_kind']:
        raise r.IntegrityError('Run/config mode mismatch')
    if descriptor['spec_id']!=r.SPEC_ID or descriptor['spec_version']!=r.SPEC_VERSION or descriptor['spec_sha256']!=r.file_sha256(run/'inputs/M0_SPEC.md'):
        raise r.IntegrityError('Spec identity mismatch')
    for entry in descriptor['input_files']:
        if r.file_sha256(r.safe_child(run,entry['path'],must_exist=True))!=entry['sha256']: raise r.IntegrityError('Original input hash mismatch')
    if descriptor['config_sha256']!=r.file_sha256(run/'inputs/config.input'): raise r.IntegrityError('Original config mismatch')
    original=r.read_json(run/'inputs/config.input')
    if original.get('run_kind','real')!=config['run_kind']: raise r.IntegrityError('Run kind changed after configuration')
    for field in ('mode','sources','audit_workers','audit_seed','check_devices','baseline_checks','timeouts'):
        if original[field]!=config[field]: raise r.IntegrityError('Resolved configuration changed: '+field)
    for name in ('lineage','exposure','protocol','decisions','literature'):
        if r.read_json(run/f'inputs/{name}.input')!=_get(run,f'inputs/{name}.json'): raise r.IntegrityError('Input snapshot mismatch: '+name)
    source=_get(run,'baseline/git.json'); inventory=_get(run,'baseline/source_inventory.json')
    captured=[row for row in inventory if row['capture_status']=='captured']
    for row in inventory:
        path=r.safe_child(run/'baseline/source_snapshot',row['path'])
        if row['capture_status']=='deleted':
            if path.exists(): raise r.IntegrityError('Deleted source reappears in snapshot')
        elif row['capture_status']!='captured' or path.stat().st_size!=row['bytes'] or r.file_sha256(path)!=row['sha256']:
            raise r.IntegrityError('Source reconstruction differs')
    actual={path.relative_to(run/'baseline/source_snapshot').as_posix() for path in (run/'baseline/source_snapshot').rglob('*') if path.is_file()}
    if actual!={row['path'] for row in captured}: raise r.IntegrityError('Snapshot has extra/missing source')
    if r.snapshot_sha256(captured,source['capture_policy'])!=source['source_snapshot_sha256']: raise r.IntegrityError('Snapshot identity differs')
    core={key:source[key] for key in ('head','ref','status_porcelain_z','dirty','index_sha256','unresolved_lfs','unresolved_submodules')}
    if source['source_identity']!=r.semantic_sha256({'git':core,'snapshot':source['source_snapshot_sha256']},domain='m0.source_identity'):
        raise r.IntegrityError('Dirty source identity differs')
    checks=_get(run,'baseline/checks.json')
    if len({check['check_id'] for check in checks})!=len(checks): raise r.IntegrityError('Duplicate check IDs')
    python=next(check['argv'][0] for check in checks if check['check_id']=='imports')
    expected=m0.baseline_plan(python,config['check_devices'],environment_seconds=config['timeouts']['environment_seconds'],regression_seconds=config['timeouts']['regression_seconds'])
    by_id={check['check_id']:check for check in checks}
    if set(by_id)!={item[0] for item in expected}|{'gpu-driver'}: raise r.IntegrityError('Incomplete/unexpected baseline command plan')
    for name,argv,device,timeout in expected:
        item=by_id[name]
        if item['argv']!=argv or item['device']!=device or item['timeout_seconds']!=timeout or item['cwd']!=config['repo_root']:
            raise r.IntegrityError('Non-allowlisted baseline command')
        if item['status']=='passed' and item['return_code']!=0: raise r.IntegrityError('Successful check has nonzero return code')
    for item in checks:
        for log in ('stdout','stderr'): r.verify_artifact(run,item[log])
        for path,digest in item['input_hashes'].items():
            if r.file_sha256(r.safe_child(run,path,must_exist=True))!=digest: raise r.IntegrityError('Check input identity changed')
    for device in ('cpu','cuda'):
        selected=[item for item in checks if item['device']==device and item['check_id']!='gpu-driver']
        expected_status='not_requested' if not selected else 'passed' if all(item['status']=='passed' for item in selected) else 'failed'
        environment=_get(run,f'baseline/environment.{device}.json')
        if environment['status']!=expected_status or environment['check_ids']!=[item['check_id'] for item in selected]:
            raise r.IntegrityError('Device record contradicts actual checks')
        observation=by_id.get('environment-'+device)
        if observation and observation['status']=='passed':
            text=r.safe_child(run,observation['stdout']['path']).read_text(encoding='utf-8')
            observed=json.loads(text[text.index('{'):])
            if environment['observed']!=observed or observed.get('device_checked')!=device:
                raise r.IntegrityError('Environment observation contradicts device/log')
    regression=(run/'logs/regressions.stderr.log').read_text(encoding='utf-8',errors='replace')
    if by_id['regressions']['status']=='passed':
        match=re.search(r'Ran (\d+) tests? in ',regression)
        if not match or int(match[1])<1 or 'skipped=' in regression or not re.search(r'^OK\s*$',regression,re.M):
            raise r.IntegrityError('Regression success not supported by current logs')
    expected_trace=_trace(regression,config['run_kind']=='real')
    if _get(run,'acceptance/tests.json')!=expected_trace: raise r.IntegrityError('Acceptance trace contradicts actual test logs')
    operations=_get(run,'logs/operations.json')
    audit_op=next((item for item in operations if item['operation_id']=='data-audit'),None)
    if not audit_op or audit_op['status']!='passed' or audit_op['return_code']!=0: raise r.IntegrityError('Audit has no successful operation evidence')
    # No process is launched here. The original command is checked against its frozen source location.
    flavor=PureWindowsPath if PureWindowsPath(descriptor['runner_source_root']).drive else PurePosixPath
    worker=str(flavor(descriptor['runner_source_root'])/'tools/m0_audit_worker.py')
    original_run=str(flavor(config['output_root'])/run_id)
    if audit_op['argv']!=[python,worker,'--run-dir',original_run] or audit_op['timeout_seconds']!=config['timeouts']['audit_seconds']:
        raise r.IntegrityError('Unexpected audit command')
    for log in ('stdout','stderr'): r.verify_artifact(run,audit_op[log])
    for path,digest in audit_op['input_hashes'].items():
        if r.file_sha256(r.safe_child(run,path,must_exist=True))!=digest: raise r.IntegrityError('Audit input identity changed')
    rows,components,audit,identity=e._inventory(run,run_id)
    if any(row['source']!='cancer_binary' or row['image_id']!=r.content_id('m0.image',{'source':row['source'],'relative_path':row['relative_path']}) for row in rows):
        raise r.IntegrityError('Image path/source identity differs')
    exclusions=r.read_jsonl(run/'audit/exclusions.jsonl')
    if exclusions!=[row for row in rows if row['status']=='excluded']: raise r.IntegrityError('Exclusion accounting differs')
    counts=dict(discovered=len(rows),included=sum(row['status']=='included' for row in rows),excluded=len(exclusions))
    if audit['counts']!=counts or _get(run,'audit/group_summary.json')['counts']!=counts: raise r.IntegrityError('Image counts do not reconcile')
    replay=copy.deepcopy(rows)
    for row in replay:
        if row['exclusion_reason_code']=='EXACT_PIXEL_DUPLICATE': row.update(status='included',exclusion_reason_code=None,exclusion_detail=None)
    if _components(replay)!=components or any(a['status']!=b['status'] or a['representative_image_id']!=b['representative_image_id'] for a,b in zip(rows,replay)):
        raise r.IntegrityError('Duplicate representative/linkage policy differs')
    if audit['data_integrity_ready']!=(not bool(audit['blockers'])): raise r.IntegrityError('Data status contradicts blockers')
    if any(c['label_conflict'] for c in components) and 'DATA_LABEL_CONFLICT' not in audit['blockers']: raise r.IntegrityError('Label conflict hidden')
    if _get(run,'audit/lineage.json')!=lineage_record(_get(run,'inputs/lineage.json')): raise r.IntegrityError('Lineage was silently promoted')
    near=_get(run,'audit/near_duplicate_review.json')
    if near['status']!='incomplete' or near['algorithm']!='not_run' or near['automatic_similar_image_exclusions']:
        raise r.IntegrityError('Unimplemented near-duplicate review claimed complete')
    history=_get(run,'inputs/exposure.json')
    instance=history['source_instance_id'] or 'unreviewed-'+r.semantic_sha256(config['data_root'],domain='m0.unreviewed_source')[:20]
    replay=e.resolve_exposure(run,run_id,history_root=run/'inputs/history',source_instance_id=instance,
        events=history['events'],history_coverage=history['history_coverage'],equivalences=history['equivalences'],_archive=True,_publish=False)
    actual=_get(run,'governance/exposure_register.json')
    # Error wording may contain the original absolute root; all decision-bearing fields are deterministic.
    for key in ('coverage','confidence','restriction','image_ids','filename_group_ids','duplicate_component_ids','historical_paths','source_kind'):
        if [item[key] for item in actual]!=[item[key] for item in replay['register']]: raise r.IntegrityError('Exposure resolution differs: '+key)
    if _get(run,'governance/component_eligibility.json')!=replay['eligibility']: raise r.IntegrityError('Component eligibility differs')
    review=_get(run,'governance/exposure_summary.json')
    for key in ('history_review_ready','eligibility_counts','inventory_sha256','status','confirmatory_design_ready'):
        if review[key]!=replay['summary'][key]: raise r.IntegrityError('Exposure summary differs')
    protocol=_get(run,'governance/protocol.initial.json'); decisions=_get(run,'governance/decisions.json'); literature=_get(run,'governance/literature_matrix.json')
    p.validate_protocol(protocol); p.validate_decisions(decisions,protocol); literature_result=p.validate_literature(literature)
    initial=_get(run,'inputs/protocol.json')
    if any(protocol[key]!=initial[key] for key in ('protocol_id','revision','lifecycle','reviewer','topics')): raise r.IntegrityError('Protocol choices were promoted')
    if decisions!=_get(run,'inputs/decisions.json') or literature!=_get(run,'inputs/literature.json'): raise r.IntegrityError('Governance inputs changed')
    expected_bindings=dict(source_snapshot_sha256=source['source_snapshot_sha256'],audit_inventory_sha256=r.file_sha256(run/'audit/inventory.jsonl'),
        exposure_register_sha256=r.file_sha256(run/'governance/exposure_register.json'),literature_input_sha256=r.semantic_sha256(literature,domain='m0.literature_input'),
        decision_input_sha256=r.semantic_sha256(decisions,domain='m0.decision_input'))
    if protocol['bindings']!=expected_bindings: raise r.IntegrityError('Protocol bindings are stale')
    protocol_review=_get(run,'governance/protocol_review.json')
    expected_review=p.validate_protocol(protocol)
    expected_review.update(literature=literature_result,history_review_ready=review['history_review_ready'],
        confirmatory_design_ready=False,scientific_authorization=False,protocol_input_sha256=r.semantic_sha256(initial,domain='m0.protocol_input'))
    if protocol_review!=expected_review:
        raise r.IntegrityError('Protocol/literature review overclaims readiness')
    _scope(_get(run,'inputs/pilot_scope.json'))
    return _facts(run)


def verify_run(run_dir):
    """Validate a frozen run read-only. Never executes checks, audits or repairs."""
    run=Path(run_dir).resolve()
    index=r.read_record(run,'acceptance/evidence_index.json',expected_type='evidence_index')
    run_id=index['run_id']
    r.verify_evidence_index(run,index,require_complete=False)
    files=set()
    for path in run.rglob('*'):
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)(): raise r.IntegrityError('Evidence contains links')
        if path.is_file(): files.add(path.relative_to(run).as_posix())
    indexed={entry['path'] for entry in index['items']}
    if files != indexed|FINALS: raise r.IntegrityError('Incomplete evidence graph or unexpected files')
    for relative in (set(RECORDS)-OPTIONAL)|{'audit/inventory.jsonl','audit/exclusions.jsonl','baseline/tracked.patch','baseline/staged.patch','inputs/M0_SPEC.md'}:
        if relative not in indexed: raise r.IntegrityError('Required artifact omitted from index: '+relative)
    for entry in index['items']:
        if entry['operation_id'] not in {item['operation_id'] for item in _get(run,'logs/operations.json')}:
            raise r.IntegrityError('Artifact has no producing operation')
        if entry['path'] in RECORDS and (entry['record_type']!=RECORDS[entry['path']] or entry['schema_version']!=1 or entry['status']!='complete'):
            raise r.IntegrityError('Required record is not indexed as complete typed evidence')
        if entry['path'] in ('audit/inventory.jsonl','audit/exclusions.jsonl') and entry.get('row_record_type')!='image_inventory_row':
            raise r.IntegrityError('JSONL row schema omitted from index')
    for relative,kind in [('acceptance/gate_g0.json','gate_g0'),('acceptance/m1_handoff.json','m1_handoff'),('acceptance/receipt.json','completion_receipt')]:
        record=r.read_record(run,relative,expected_type=kind)
        if record['run_id']!=run_id: raise r.IntegrityError('Final artifact belongs to another run')
    receipt=r.read_record(run,'acceptance/receipt.json')['data']
    if set(receipt['artifacts'])!=FINALS-{'acceptance/receipt.json'} or receipt['state']!='completed': raise r.IntegrityError('Incomplete completion receipt')
    for relative,digest in receipt['artifacts'].items():
        if r.file_sha256(r.safe_child(run,relative,must_exist=True))!=digest: raise r.IntegrityError('Final artifact differs: '+relative)
    facts=verify_semantics(run)
    config=_get(run,'config.resolved.json')
    index_hash=r.file_sha256(run/'acceptance/evidence_index.json')
    expected=evaluate_gate(facts,config['mode'],index_hash)
    gate=r.read_record(run,'acceptance/gate_g0.json')['data']
    if gate!=expected: raise r.IntegrityError('Gate does not follow evidence/policy')
    handoff=_handoff(run,gate,index_hash)
    if r.read_record(run,'acceptance/m1_handoff.json')['data']!=handoff: raise r.IntegrityError('M1 handoff differs from evidence')
    if (run/'acceptance/M0_REPORT.md').read_text(encoding='utf-8')!=_report(run_id,gate,handoff): raise r.IntegrityError('Human report contradicts gate')
    return dict(validated=True,run_id=run_id,artifacts=len(indexed),gate=gate,exit_code=1 if gate['overall_status']=='blocked' else 0)
