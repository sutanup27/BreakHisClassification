"""M0.5 declarative protocol/decision/literature validation; never trains or searches."""
import copy
from datetime import date
import re

from . import records as r

TOPICS = {
    'task_source': ('task','source','data_root','dataset_scope','input_resolution','provenance_boundary'),
    'grouping': ('rule','clinical_identity','exact_duplicates','exposure'),
    'aggregation': ('primary','secondary','missing_magnifications'),
    'hardware': ('training','primary_timing','secondary_timing','cpu_threads'),
    'architectures': ('primary','compact','standard','surgery'),
    'evaluation': ('outer_folds','inner_roles','calibration_role','exposure_constraints'),
    'seeds': ('namespaces','values'),
    'selection': ('epoch','candidate','threshold','calibration_data'),
    'safeguards': ('sensitivity_margin','specificity_margin','absolute_performance','clinical_noninferiority'),
    'budgets': ('dense_epochs','adaptation_epochs','recovery_epochs','candidates','latency_budgets','matched_costs'),
    'compute_gate': ('day','cost_estimation','stop_rule'),
    'hypotheses': ('O1','O2','O3'),
    'comparisons': ('A','B','C','D','E','controls','negative_route'),
    'statistics': ('unit','uncertainty','fold_seed_reporting','multiplicity'),
    'endpoints': ('primary','secondary','missing_support','timing_sessions'),
    'claims': ('clinical','novelty','generalization','confirmatory_freeze'),
    'pretraining': ('recipe','weights','normalization','augmentation'),
}
_STATUSES={'agreed','proposed','deferred','blocked'}
_BINDINGS={'source_snapshot_sha256','audit_inventory_sha256','exposure_register_sha256',
           'literature_input_sha256','decision_input_sha256'}
_ENTRY={'topic','value','status','rationale','evidence_refs','owner','freeze_by'}
_CRITICAL={'task_source','grouping','aggregation','hardware','evaluation','seeds','selection','safeguards','budgets','pretraining'}
_SHA=re.compile(r'[0-9a-f]{64}\Z')
LITERATURE_FIELDS={'literature_id','title','authors','year','venue','publication_status','version_read','url','doi',
    'access_date','review_date','access_level','review_status','categories','central','task_dataset','grouping',
    'ranking_allocation','class_sample_balancing','selection_threshold','search_adaptation_recovery',
    'runtime_target','code_url','license','comparator_role','distinction_overlap','reviewer','owner',
    'freeze_by','open_questions','unknown_reasons'}


def _text(value, label):
    if not isinstance(value,str) or not value.strip(): raise r.RecordError(label+' requires nonempty text')


def _strings(value,label,nonempty=True):
    if not isinstance(value,list) or (nonempty and not value) or not all(isinstance(v,str) and v.strip() for v in value):
        raise r.RecordError(label+' requires a list of nonempty strings')


def _version(value):
    if type(value) is not int or value!=1: raise r.RecordError('Unsupported input schema version')


def _entry(entry):
    if not isinstance(entry,dict) or set(entry)!=_ENTRY: raise r.RecordError('Invalid protocol topic fields')
    if entry['topic'] not in TOPICS or entry['status'] not in _STATUSES: raise r.RecordError('Invalid topic/status')
    if not isinstance(entry['value'],dict) or set(entry['value'])!=set(TOPICS[entry['topic']]):
        raise r.RecordError('Missing or unknown policy fields for '+entry['topic'])
    for field in ('rationale','owner'): _text(entry[field],field)
    _strings(entry['evidence_refs'],'evidence_refs')
    if entry['freeze_by'] not in {f'M{n}' for n in range(8)}: raise r.RecordError('Invalid freeze milestone')
    if entry['status']=='agreed' and _contains_null(entry['value']):
        raise r.RecordError('Agreed policy cannot contain unresolved null values')
    values=entry['value']
    if entry['topic']=='seeds':
        _strings(values['namespaces'],'seed namespaces')
        if not {'split','training','pruning','search'}<=set(values['namespaces']) or len(set(values['namespaces']))!=len(values['namespaces']):
            raise r.RecordError('Independent split/training/pruning/search namespaces required')
        if values['values'] is not None:
            seeds=values['values']
            if not isinstance(seeds,dict) or set(seeds)!=set(values['namespaces']) or any(type(v) is not int or not 0<=v<2**32 for v in seeds.values()):
                raise r.RecordError('Seed values must be named uint32 integers')
    if entry['topic']=='safeguards':
        for field in ('sensitivity_margin','specificity_margin'):
            value=values[field]
            if value is not None and (type(value) not in (int,float) or not 0<=value<=1): raise r.RecordError('Margin must be null or a fraction in [0,1]')
    if entry['topic'] in ('budgets','hardware','evaluation'):
        fields={'budgets':('dense_epochs','adaptation_epochs','recovery_epochs','candidates'),
                'hardware':('cpu_threads',),'evaluation':('outer_folds',)}[entry['topic']]
        for field in fields:
            value=values[field]
            if value is not None and (type(value) is not int or value<(2 if field=='outer_folds' else 1)):
                raise r.RecordError('Invalid positive integer policy: '+field)
        if entry['topic']=='budgets' and values['latency_budgets'] is not None:
            limits=values['latency_budgets']
            if not isinstance(limits,list) or not limits or any(type(v) not in (int,float) or v<=0 for v in limits): raise r.RecordError('Latency budgets must be positive numbers')


def _contains_null(value):
    return value is None or (isinstance(value,dict) and any(_contains_null(v) for v in value.values())) or (isinstance(value,list) and any(_contains_null(v) for v in value))


def validate_protocol(protocol):
    fields={'schema_version','protocol_id','revision','lifecycle','reviewer','bindings','topics'}
    if not isinstance(protocol,dict) or set(protocol)!=fields: raise r.RecordError('Invalid initial protocol fields')
    _version(protocol['schema_version'])
    r.canonical_json_bytes(protocol)
    r.validate_id(protocol['protocol_id'])
    if type(protocol['revision']) is not int or protocol['revision']<1: raise r.RecordError('Invalid protocol revision')
    if protocol['lifecycle'] not in ('draft','pilot_ready','initial_design_reviewed'):
        raise r.RecordError('M0 cannot emit confirmatory_frozen or unknown lifecycle')
    if protocol['reviewer'] is not None: _text(protocol['reviewer'],'reviewer')
    bindings=protocol['bindings']
    if not isinstance(bindings,dict) or set(bindings)!=_BINDINGS: raise r.RecordError('Missing/unknown protocol binding')
    for digest in bindings.values():
        if digest is not None and (not isinstance(digest,str) or not _SHA.fullmatch(digest)): raise r.RecordError('Invalid binding digest')
    entries=protocol['topics']
    if not isinstance(entries,list): raise r.RecordError('Topics must be a list')
    for entry in entries: _entry(entry)
    if len(entries)!=len(TOPICS) or {e['topic'] for e in entries}!=set(TOPICS):
        raise r.RecordError('All mandatory protocol topics required exactly once')
    restrictions=['DECISION_PENDING:'+e['topic'] for e in entries if e['status']!='agreed']
    restrictions += ['OWNER_ROLE_NOT_NAMED:'+e['topic'] for e in entries if e['owner'] in ('research_owner','reviewer')]
    if any(value is None for value in bindings.values()): restrictions.append('PROTOCOL_UNBOUND')
    if protocol['lifecycle']!='draft':
        _text(protocol['reviewer'],'Named lifecycle reviewer')
        required=set(TOPICS) if protocol['lifecycle']=='initial_design_reviewed' else _CRITICAL
        if any(e['topic'] in required and e['status']!='agreed' for e in entries) or 'PROTOCOL_UNBOUND' in restrictions:
            raise r.RecordError('Lifecycle requires bound evidence and agreed policies')
    return dict(structure_valid=True,lifecycle=protocol['lifecycle'],restrictions=restrictions,
                scientific_authorization=False,confirmatory_frozen=False)


def validate_decisions(decisions, protocol):
    validate_protocol(protocol)
    if not isinstance(decisions,list): raise r.RecordError('Decisions must be a list')
    r.canonical_json_bytes(decisions)
    latest={}
    chains={}
    for item in decisions:
        extra={'decision_id','revision','supersedes','affected_claims'}
        if not isinstance(item,dict) or set(item)!=_ENTRY|extra: raise r.RecordError('Invalid decision revision fields')
        _entry({key:item[key] for key in _ENTRY})
        r.validate_id(item['decision_id'])
        _strings(item['affected_claims'],'affected_claims')
        revision=item['revision']
        if type(revision) is not int or revision<1: raise r.RecordError('Invalid decision revision')
        previous=chains.get(item['decision_id'])
        if revision!=(previous['revision']+1 if previous else 1): raise r.RecordError('Decision revisions must be consecutive and preserved')
        expected=f"{item['decision_id']}@{revision-1}" if previous else None
        if item['supersedes']!=expected or (previous and previous['topic']!=item['topic']): raise r.RecordError('Invalid supersedes chain/topic')
        if item['topic'] in latest and latest[item['topic']]['decision_id']!=item['decision_id']:
            raise r.RecordError('One stable decision ID per topic required')
        chains[item['decision_id']]=item
        latest[item['topic']]=item
    if set(latest)!=set(TOPICS): raise r.RecordError('Decision register must cover every protocol topic')
    for entry in protocol['topics']:
        if any(latest[entry['topic']][key]!=entry[key] for key in _ENTRY):
            raise r.IntegrityError('Latest decision disagrees with protocol: '+entry['topic'])
    return dict(revisions=len(decisions),topics=len(latest))


def validate_literature(matrix):
    if not isinstance(matrix,list) or not matrix: raise r.RecordError('Literature matrix cannot be empty')
    r.canonical_json_bytes(matrix)
    seen=set()
    coverage=set()
    gaps=[]
    for row in matrix:
        if not isinstance(row,dict) or set(row)!=LITERATURE_FIELDS: raise r.RecordError('Invalid literature columns')
        r.validate_id(row['literature_id'])
        if row['literature_id'] in seen: raise r.RecordError('Duplicate literature ID')
        seen.add(row['literature_id'])
        for field in ('title','publication_status','version_read','url','comparator_role','distinction_overlap','owner'):
            _text(row[field],field)
        if not row['url'].startswith('https://'): raise r.RecordError('Literature requires primary HTTPS source URL')
        if type(row['central']) is not bool: raise r.RecordError('Central must be boolean')
        _strings(row['categories'],'categories')
        if set(row['categories'])-{'hardware_aware','class_aware','sample_medical','adjacent'}: raise r.RecordError('Unknown literature category')
        if row['central']: coverage.update(row['categories'])
        structured={'authors','year','categories','central','open_questions','unknown_reasons'}
        for field in LITERATURE_FIELDS-structured:
            if row[field] is not None: _text(row[field],field)
        if row['access_level'] not in ('full_methods','abstract_only','code_only','unavailable'): raise r.RecordError('Invalid literature access')
        if row['review_status'] not in ('pending','reviewed_methods'): raise r.RecordError('Invalid literature review status')
        if row['freeze_by'] not in {f'M{n}' for n in range(8)}: raise r.RecordError('Literature needs freeze milestone')
        _strings(row['open_questions'],'open_questions',nonempty=False)
        if not isinstance(row['unknown_reasons'],dict): raise r.RecordError('Unknown reasons must be an object')
        for field,value in row.items():
            if value is None and not (isinstance(row['unknown_reasons'].get(field),str) and row['unknown_reasons'][field].strip()):
                raise r.RecordError('Unavailable literature field requires reason: '+field)
        if row['authors'] is not None: _strings(row['authors'],'authors')
        if row['year'] is not None and (type(row['year']) is not int or not 1900<=row['year']<=2100): raise r.RecordError('Invalid citation year')
        for field in ('access_date','review_date'):
            if row[field] is not None:
                try: date.fromisoformat(row[field])
                except (TypeError,ValueError) as error: raise r.RecordError('Invalid review/access date') from error
        if row['review_status']=='reviewed_methods':
            if row['access_level']!='full_methods' or not row['reviewer'] or not row['review_date']:
                raise r.RecordError('Abstract/code/unavailable access cannot count as reviewed full methods')
            if any(row[field] is None for field in ('authors','year','venue')): raise r.RecordError('Full-method review needs an attributable citation')
            for field in ('task_dataset','ranking_allocation','distinction_overlap'): _text(row[field],field)
        if row['central'] and (row['review_status']!='reviewed_methods' or row['open_questions'] or row['unknown_reasons']):
            gaps.append(dict(literature_id=row['literature_id'],owner=row['owner'],freeze_by=row['freeze_by'],
                             restriction='CENTRAL_METHOD_REVIEW_INCOMPLETE'))
    if not {'hardware_aware','class_aware','sample_medical'}<=coverage: raise r.RecordError('Missing nearest-method literature category')
    return dict(entries=len(matrix),full_method_reviews=sum(row['review_status']=='reviewed_methods' for row in matrix),
                central_gaps=gaps,novelty_review_complete=not gaps,novelty_claim='unverified; requires scientific review')


def write_protocol_records(run_dir,run_id,*,protocol,decisions,literature):
    """Bind declarations to current run artifacts without promoting any choices."""
    validate_protocol(protocol)
    validate_decisions(decisions,protocol)
    literature_result=validate_literature(literature)
    git=r.read_record(run_dir,'baseline/git.json',expected_type='git')
    exposure=r.read_record(run_dir,'governance/exposure_register.json',expected_type='exposure_register')
    review=r.read_record(run_dir,'governance/exposure_summary.json',expected_type='exposure_summary')
    if any(record['run_id']!=run_id for record in (git,exposure,review)) or git['data'].get('status')!='complete':
        raise r.IntegrityError('Protocol requires attributable source/exposure records for this run')
    from .exposure import inventory_scope
    scope=inventory_scope(run_dir,run_id)
    if review['data']['inventory_sha256']!=scope['inventory_sha256']: raise r.IntegrityError('Exposure review is stale')
    if protocol['lifecycle']!='draft' and not review['data']['history_review_ready']:
        raise r.RecordError('Reviewed protocol lifecycle cannot bypass unresolved history')
    bound=copy.deepcopy(protocol)
    bindings=dict(source_snapshot_sha256=git['data']['source_snapshot_sha256'],
        audit_inventory_sha256=r.file_sha256(r.safe_child(run_dir,'audit/inventory.jsonl',must_exist=True)),
        exposure_register_sha256=r.file_sha256(r.safe_child(run_dir,'governance/exposure_register.json',must_exist=True)),
        literature_input_sha256=r.semantic_sha256(literature,domain='m0.literature_input'),
        decision_input_sha256=r.semantic_sha256(decisions,domain='m0.decision_input'))
    for key,value in bindings.items():
        if bound['bindings'][key] is not None and bound['bindings'][key]!=value: raise r.IntegrityError('Stale protocol binding: '+key)
    bound['bindings']=bindings
    result=validate_protocol(bound)
    validate_decisions(decisions,bound)
    for path in ('governance/protocol.initial.json','governance/decisions.json','governance/literature_matrix.json','governance/protocol_review.json'):
        if r.safe_child(run_dir,path).exists(): raise FileExistsError(path)
    result.update(literature=literature_result,history_review_ready=review['data']['history_review_ready'],
                  confirmatory_design_ready=False,scientific_authorization=False,
                  protocol_input_sha256=r.semantic_sha256(protocol,domain='m0.protocol_input'))
    for path,kind,payload in [('governance/protocol.initial.json','protocol.initial',bound),
            ('governance/decisions.json','decisions',decisions),('governance/literature_matrix.json','literature_matrix',literature),
            ('governance/protocol_review.json','protocol_review',result)]:
        r.write_record(run_dir,path,r.new_record(kind,run_id,payload))
    return result
