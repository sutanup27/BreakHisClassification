"""M0.4 metadata-only exposure resolution; never loads images, models or metrics.

History declarations require review. Missing logs never establish non-exposure.
"""
from collections import defaultdict
from pathlib import Path
import re

from . import records as r

_SHA = re.compile('[0-9a-f]{64}\\Z')
_ACCESS = {'train','selection','threshold','test_metric_inspection','test_prediction_inspection','synthetic_check','unknown'}
_EVENT = {'event_id','artifact','manifest','source','source_instance_id','source_kind',
          'selected_partition','magnification_filter','access_type','accessed_paths',
          'evidence_source','occurred_at','reviewer_note'}


def _text(value, label):
    if not isinstance(value,str) or not value.strip():
        raise r.RecordError(label+' must be nonempty text')


def _digest(value):
    if not isinstance(value,str) or not _SHA.fullmatch(value):
        raise r.RecordError('Expected SHA-256')


def _reference(ref, history_root, run):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
        raise r.RecordError('Metadata reference requires path and sha256')
    _digest(ref['sha256'])
    path=r.safe_child(history_root,ref['path'],must_exist=True)
    if path.is_relative_to(run) or path.suffix.lower() not in {'.json','.jsonl','.log','.txt','.md','.yaml','.yml'}:
        raise r.RecordError('Reference must be historical metadata, outside the current run')
    if r.file_sha256(path)!=ref['sha256']:
        raise r.IntegrityError('Historical evidence hash mismatch: '+ref['path'])
    return path


def validate_event(event):
    if not isinstance(event,dict) or set(event)!=_EVENT:
        raise r.RecordError('Exposure event has missing or unknown fields (metrics/predictions are not inputs)')
    r.canonical_json_bytes(event)
    r.validate_id(event['event_id'])
    for field in ('source','source_instance_id'):
        _text(event[field],field)
    if event['source_kind'] not in ('real','synthetic') or event['access_type'] not in _ACCESS:
        raise r.RecordError('Invalid source kind or access type')
    if (event['source_kind']=='synthetic') != (event['access_type']=='synthetic_check'):
        raise r.RecordError('Synthetic checks require separate synthetic source identity')
    if event['evidence_source'] not in ('config','log','manifest','operator_attestation'):
        raise r.RecordError('Invalid evidence source')
    if event['selected_partition'] not in ('train','validation','test','all'):
        raise r.RecordError('Invalid historical partition')
    mags=event['magnification_filter']
    if not isinstance(mags,list) or not mags or not all(isinstance(m,str) for m in mags) or len(set(mags))!=len(mags) or (mags!=['all'] and set(mags)-{'40X','100X','200X','400X'}):
        raise r.RecordError('Invalid magnification filter')
    paths=event['accessed_paths']
    if paths is not None:
        if not isinstance(paths,list) or not all(isinstance(path,str) for path in paths) or len(set(paths))!=len(paths):
            raise r.RecordError('Accessed paths must be distinct paths or null')
        for path in paths: r.portable_relative_path(path)
    if event['occurred_at'] is not None: _text(event['occurred_at'],'occurred_at')
    if event['reviewer_note'] is not None: _text(event['reviewer_note'],'reviewer_note')
    return event


def _inventory(run, run_id):
    audit=r.read_record(run,'audit/audit.json',expected_type='audit')
    components=r.read_record(run,'audit/duplicate_components.json',expected_type='duplicate_components')
    if audit['run_id']!=run_id or components['run_id']!=run_id:
        raise r.IntegrityError('Inventory belongs to another run')
    path=r.safe_child(run,'audit/inventory.jsonl',must_exist=True)
    rows=[]
    # Strict JSON loader also catches duplicate keys/nonfinite values in JSONL rows.
    import json
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line,object_pairs_hook=r._unique_object)
        r.canonical_json_bytes(row)
        if not isinstance(row,dict): raise r.RecordError('Inventory row must be an object')
        if row.get('schema_version')!=1 or type(row['schema_version']) is not int or row.get('record_type')!='image_inventory_row':
            raise r.RecordError('Unsupported inventory row schema')
        rows.append(row)
    if len({row['image_id'] for row in rows})!=len(rows):
        raise r.IntegrityError('Duplicate inventory identity')
    expected=r.semantic_sha256([dict(path=row['relative_path'],sha256=row['raw_sha256']) for row in rows],domain='m0.data_inventory')
    if expected!=audit['data']['input_snapshot_sha256']:
        raise r.IntegrityError('Inventory identity differs from audit')
    seen=set()
    by_id={row['image_id']:row for row in rows}
    for component in components['items']:
        ids=component['image_ids']
        if component['component_id']!=r.content_id('m0.component.v1',sorted(ids)):
            raise r.IntegrityError('Invalid component identity')
        for image_id in ids:
            if image_id in seen or image_id not in by_id or by_id[image_id]['duplicate_component_id']!=component['component_id']:
                raise r.IntegrityError('Inconsistent component membership')
            seen.add(image_id)
    if seen!=set(by_id): raise r.IntegrityError('Incomplete component membership')
    from .data_readiness import _components
    import copy
    if _components(copy.deepcopy(rows))!=components['items']:
        raise r.IntegrityError('Components do not follow the declared linkage policy')
    return rows,components['items'],audit['data'],r.semantic_sha256(rows,domain='m0.exposure_inventory')


def _manifest_rows(path, event):
    manifest=r.read_json(path)
    if not isinstance(manifest,dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version']!=1 or not isinstance(manifest.get('rows'),list):
        raise r.RecordError('Expected historical manifest v1')
    selected=[]
    seen=set()
    for row in manifest['rows']:
        if not isinstance(row,dict): raise r.RecordError('Invalid manifest row')
        if row.get('source')!=event['source']: continue
        relative=r.portable_relative_path(row.get('path'))
        if relative in seen: raise r.IntegrityError('Duplicate historical path')
        seen.add(relative)
        if row.get('split') not in ('train','validation','test') or row.get('magnification') not in ('40X','100X','200X','400X'):
            raise r.RecordError('Invalid historical partition/magnification')
        _text(row.get('patient_id'),'Historical filename group')
        for key in ('sha256','pixel_sha256'): _digest(row.get(key))
        if (event['selected_partition']=='all' or row['split']==event['selected_partition']) and (event['magnification_filter']==['all'] or row['magnification'] in event['magnification_filter']):
            selected.append(row)
    if not selected: raise r.IntegrityError('Historical filter selects no attributable rows')
    return selected


def inventory_scope(run_dir, run_id):
    """Read-only identity to bind a separately reviewed history declaration."""
    _,components,_,digest=_inventory(Path(run_dir).resolve(),run_id)
    return dict(inventory_sha256=digest,component_ids=sorted(c['component_id'] for c in components))


def resolve_exposure(run_dir, run_id, *, history_root, source_instance_id, events,
                     history_coverage=None, equivalences=None):
    """Resolve explicit metadata declarations against the current M0.3 audit.

    Opaque artifact logs are hashed, never parsed for scores/errors/probabilities.
    Missing evidence yields unresolved records; malformed input fails before IO.
    No directory-wide discovery, model inference or held-out group selection.
    """
    r.validate_id(run_id)
    _text(source_instance_id,'Current source instance')
    if not isinstance(events,list): raise r.RecordError('Events must be a list')
    for event in events: validate_event(event)
    if len({event['event_id'] for event in events})!=len(events):
        raise r.RecordError('Duplicate exposure event ID')
    run,history=Path(run_dir).resolve(),Path(history_root).resolve()
    if not history.is_dir(): raise r.RecordError('Historical metadata root missing')
    rows,components,audit,inventory_hash=_inventory(run,run_id)
    component_ids={component['component_id'] for component in components}
    # Explicit reviewed equivalences supplement exact hash matches across instances.
    bridges=defaultdict(set)
    equivalence_records=[]
    if equivalences is not None and not isinstance(equivalences,list):
        raise r.RecordError('Equivalences must be a list')
    for item in equivalences or []:
        fields={'source_instance_id','filename_group_id','current_component_id','reviewer','evidence_refs'}
        if not isinstance(item,dict) or set(item)!=fields: raise r.RecordError('Invalid equivalence declaration')
        for field in fields-{'evidence_refs'}: _text(item[field],field)
        if item['current_component_id'] not in component_ids: raise r.RecordError('Unknown equivalent component')
        if not isinstance(item['evidence_refs'],list) or not item['evidence_refs']: raise r.RecordError('Equivalence needs evidence')
        for ref in item['evidence_refs']: _reference(ref,history,run)
        bridges[(item['source_instance_id'],item['filename_group_id'])].add(item['current_component_id'])
        equivalence_records.append(item)
    by_group,by_raw,by_pixel=defaultdict(set),defaultdict(set),defaultdict(set)
    by_path={row['relative_path']:row for row in rows}
    for row in rows:
        for field,index in [('filename_group_id',by_group),('raw_sha256',by_raw),('decoded_pixel_sha256',by_pixel)]:
            if row[field]: index[row[field]].add(row['duplicate_component_id'])
    register=[]
    exposed=defaultdict(set)
    unresolved=[]
    for event in events:
        result=dict(exposure_id=event['event_id'],artifact_ref=event['artifact'],artifact_sha256=None,
            manifest_ref_and_hash=event['manifest'],occurred_at_or_unknown=event['occurred_at'],
            occurred_at_reason=None if event['occurred_at'] else 'Historical time not supplied',
            evidence_source=event['evidence_source'],source=event['source'],source_instance_id=event['source_instance_id'],
            source_kind=event['source_kind'],selected_partition=event['selected_partition'],
            magnification_filter=event['magnification_filter'],access_type=event['access_type'],
            image_ids=[],filename_group_ids=[],duplicate_component_ids=[],historical_paths=[],
            coverage='unresolved',confidence='unresolved',restriction='unresolved',reviewer_note=event['reviewer_note'],
            issues=[],owner='research_owner',freeze_by='M1',resolution='Review attributable historical metadata and coverage')
        matched=set()
        try:
            _reference(event['artifact'],history,run)
            result['artifact_sha256']=event['artifact']['sha256']
            if event['source_kind']=='synthetic':
                if event['source_instance_id']==source_instance_id:
                    raise r.RecordError('Synthetic and real instances must differ')
                result.update(coverage='exact',confidence='documented',restriction='synthetic_not_real')
            else:
                manifest_path=_reference(event['manifest'],history,run)
                selected=_manifest_rows(manifest_path,event)
                paths=event['accessed_paths']
                exact=paths is not None and event['evidence_source']=='log'
                if exact and not paths:
                    result['issues'].append('EMPTY_PER_IMAGE_ACCESS_LOG')
                if paths is not None and set(paths)-{row['path'] for row in selected}:
                    result['issues'].append('PER_IMAGE_LOG_CONTRADICTS_FILTER')
                chosen=[row for row in selected if row['path'] in paths] if exact and not result['issues'] else selected
                result['historical_paths']=sorted(row['path'] for row in chosen)
                for historical_row in chosen:
                    matches=by_raw[historical_row['sha256']] | by_pixel[historical_row['pixel_sha256']]
                    if event['source_instance_id']==source_instance_id:
                        matches |= by_group[historical_row['patient_id']]
                        current=by_path.get(historical_row['path'])
                        if current and (current['raw_sha256']!=historical_row['sha256'] or current['decoded_pixel_sha256']!=historical_row['pixel_sha256']):
                            result['issues'].append('HISTORICAL_CONTENT_CONTRADICTION')
                    matches |= bridges[(event['source_instance_id'],historical_row['patient_id'])]
                    if not matches: result['issues'].append('HISTORICAL_IDENTITY_UNRESOLVED')
                    matched |= matches
                if event['access_type']=='unknown': result['issues'].append('ACCESS_TYPE_UNKNOWN')
                if not result['issues']:
                    result.update(coverage='exact' if exact else 'conservative_superset',
                                  confidence='self_reported' if event['evidence_source']=='operator_attestation' else 'documented',
                                  restriction='development_only')
        except (OSError,r.RecordError) as error:
            result['issues'].append(str(error))
        if event['source_kind']=='real':
            for component in matched: exposed[component].add(event['event_id'])
            affected=[row for row in rows if row['duplicate_component_id'] in matched]
            result.update(image_ids=sorted(row['image_id'] for row in affected),
                          filename_group_ids=sorted({row['filename_group_id'] for row in affected if row['filename_group_id']}),
                          duplicate_component_ids=sorted(matched))
            if result['coverage']=='unresolved': unresolved.append(event['event_id'])
        elif result['coverage']=='unresolved':
            # Contradictory synthetic identity cannot silently establish real non-exposure.
            unresolved.append(event['event_id'])
        result['issues']=sorted(set(result['issues']))
        register.append(result)
    # Recheck every supplied reference before granting any candidate status. A
    # moving historical file cannot be represented by its earlier hash alone.
    for event in events:
        try:
            _reference(event['artifact'],history,run)
            if event['source_kind']=='real': _reference(event['manifest'],history,run)
        except (OSError,r.RecordError):
            if event['event_id'] not in unresolved: unresolved.append(event['event_id'])
            next(item for item in register if item['exposure_id']==event['event_id'])['issues'].append('HISTORICAL_REFERENCE_UNAVAILABLE_OR_CHANGED')
            next(item for item in register if item['exposure_id']==event['event_id']).update(coverage='unresolved',restriction='unresolved',confidence='unresolved')
    for bridge in equivalence_records:
        try:
            for ref in bridge['evidence_refs']: _reference(ref,history,run)
        except (OSError,r.RecordError):
            for event in events:
                if event['source_kind']=='real' and event['source_instance_id']==bridge['source_instance_id']:
                    if event['event_id'] not in unresolved: unresolved.append(event['event_id'])
                    item=next(item for item in register if item['exposure_id']==event['event_id'])
                    item['issues'].append('EQUIVALENCE_EVIDENCE_UNAVAILABLE_OR_CHANGED')
                    item.update(coverage='unresolved',restriction='unresolved',confidence='unresolved')
    covered=set()
    coverage_issues=[]
    if history_coverage is None:
        coverage_issues.append('HISTORY_COVERAGE_MISSING')
    else:
        required={'schema_version','status','reviewer','inventory_sha256','source_instance_id',
                  'component_ids','events_complete','evidence_refs','reviewer_note'}
        if not isinstance(history_coverage,dict) or set(history_coverage)!=required:
            raise r.RecordError('Invalid history-coverage declaration')
        try:
            if type(history_coverage['schema_version']) is not int or history_coverage['schema_version']!=1: raise r.RecordError('Unsupported coverage schema')
            _text(history_coverage['reviewer'],'Coverage reviewer')
            _text(history_coverage['reviewer_note'],'Coverage scope/review note')
            if history_coverage['status']!='reviewed' or history_coverage['events_complete'] is not True:
                raise r.RecordError('History coverage is not explicitly reviewed/complete')
            if history_coverage['source_instance_id']!=source_instance_id or history_coverage['inventory_sha256']!=inventory_hash:
                raise r.IntegrityError('Coverage belongs to another inventory/instance')
            ids=history_coverage['component_ids']
            if not isinstance(ids,list) or not all(isinstance(c,str) for c in ids) or len(set(ids))!=len(ids) or set(ids)-component_ids:
                raise r.RecordError('Invalid coverage components')
            refs=history_coverage['evidence_refs']
            if not isinstance(refs,list) or not refs: raise r.RecordError('Coverage requires evidence references')
            for ref in refs: _reference(ref,history,run)
            covered=set(ids)
        except (OSError,r.RecordError) as error:
            coverage_issues.append(str(error))
    eligibility=[]
    for component in sorted(component_ids):
        known=bool(exposed[component])
        candidate=component in covered and not unresolved and audit['data_integrity_ready'] and not known
        eligibility.append(dict(component_id=component,
            status='development_only' if known else 'candidate_untouched_with_reviewed_coverage' if candidate else 'exposure_unknown',
            event_ids=sorted(exposed[component]),coverage_reviewed=component in covered,
            scope_inventory_sha256=inventory_hash,source_instance_id=source_instance_id,
            restriction='Never use as untouched confirmation' if known else
                        'Candidate within declared review boundary; fold/protocol approval still required' if candidate else
                        'History unresolved; cannot claim untouched',
            owner='research_owner',freeze_by='M1',resolution='Review history coverage and downstream design before scientific use'))
    blockers=([dict(reason='EXPOSURE_UNRESOLVED',event_ids=unresolved)] if unresolved else [])
    if covered!=component_ids or coverage_issues:
        blockers.append(dict(reason='HISTORY_COVERAGE_INCOMPLETE',issues=coverage_issues))
    if not audit['data_integrity_ready']: blockers.append(dict(reason='DATA_INTEGRITY_BLOCKED'))
    summary=dict(source_instance_id=source_instance_id,inventory_sha256=inventory_hash,
        history_review_ready=not blockers,confirmatory_design_ready=False,
        status='blocked' if blockers else 'restricted',blockers=blockers,
        eligibility_counts={status:sum(item['status']==status for item in eligibility) for status in
            ('development_only','exposure_unknown','candidate_untouched_with_reviewed_coverage')},
        coverage_declaration=history_coverage,coverage_issues=coverage_issues,equivalences=equivalence_records,
        owner='research_owner',freeze_by='M1',restrictions=['Review declarations are not automated proof of complete history',
            'Reseeding never erases exposure','No fold selection or model/test inference performed',
            'Final Gate G0, lineage, protocol and fold support remain downstream'])
    destinations=['governance/exposure_register.json','governance/component_eligibility.json','governance/exposure_summary.json']
    for relative in destinations:
        if r.safe_child(run,relative).exists(): raise FileExistsError(relative)
    for relative,kind,payload in zip(destinations,['exposure_register','component_eligibility','exposure_summary'],[register,eligibility,summary]):
        r.write_record(run,relative,r.new_record(kind,run_id,payload))
    return summary
