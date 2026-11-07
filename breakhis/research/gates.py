"""Pure G0 policy: evidence completion, engineering and scientific scope differ."""
from . import records as r


def evaluate_gate(facts, requested_mode, index_sha256):
    if requested_mode not in ('pilot', 'confirmatory_design'):
        raise r.RecordError('Unknown requested mode')
    dimensions, blockers = {}, []
    def dimension(name, hard=(), limits=()):
        hard, limits = sorted(set(hard)), sorted(set(limits))
        status = 'blocked' if hard else 'restricted' if limits else 'ready'
        dimensions[name] = dict(status=status, reason_codes=hard+limits)
        for code in hard:
            blockers.append(dict(blocker_id=r.content_id('m0.blocker', [name, code]),
                dimension=name, reason_code=code, owner='research_owner',
                evidence_refs=['acceptance/evidence_index.json'],
                resolution='Repair or review the named condition and produce a fresh evidence run'))
    common = []
    if not facts['baseline_passed']: common.append('BASELINE_FAILED')
    if not facts['source_consistent']: common.append('SOURCE_INCONSISTENT')
    if not facts['evidence_complete']: common.append('EVIDENCE_INCOMPLETE')
    engineering_limits = []
    if not facts['data_ready']: engineering_limits.append('DATA_DEPENDENT_WORK_BLOCKED')
    if not facts['acceptance_complete']: engineering_limits.append('ACCEPTANCE_TRACE_INCOMPLETE')
    if not facts['real_input']: engineering_limits.append('SYNTHETIC_EVIDENCE_ONLY')
    dimension('engineering', common, engineering_limits)
    scientific = list(common)
    if not facts['data_ready']: scientific.append('DATA_INTEGRITY_BLOCKED')
    if not facts['real_input']: scientific.append('SYNTHETIC_INPUTS')
    if not facts['acceptance_complete']: scientific.append('ACCEPTANCE_TRACE_INCOMPLETE')
    pilot, pilot_limits = list(scientific), []
    if not facts['scope_reviewed']: pilot.append('PILOT_SCOPE_UNREVIEWED')
    if not facts['protocol_pilot_ready']: pilot.append('PROTOCOL_PENDING')
    if not facts['owners_assigned']: pilot.append('REVIEW_OWNER_UNASSIGNED')
    if not facts['lineage_known']:
        (pilot_limits if facts['lineage_scope_approved'] else pilot).append('LINEAGE_UNKNOWN')
    if not facts['history_ready']:
        (pilot_limits if facts['scope_reviewed'] else pilot).append('EXPOSURE_UNRESOLVED')
    if facts['training_device'] == 'cuda' and not facts['gpu_ready']: pilot.append('GPU_UNAVAILABLE')
    if not facts['literature_complete']: pilot_limits.append('LITERATURE_INCOMPLETE')
    dimension('exploratory', pilot, pilot_limits)
    design = list(scientific)
    if not facts['history_ready']: design.append('EXPOSURE_UNRESOLVED')
    if not facts['protocol_design_ready']: design.append('PROTOCOL_DESIGN_UNREVIEWED')
    if not facts['owners_assigned']: design.append('REVIEW_OWNER_UNASSIGNED')
    if not facts['fold_support']: design.append('FOLD_SUPPORT_UNREVIEWED')
    if not facts['literature_complete']: design.append('LITERATURE_INCOMPLETE')
    if not facts['lineage_known'] and not facts['lineage_scope_approved']: design.append('LINEAGE_UNKNOWN')
    if facts['training_device'] == 'cuda' and not facts['gpu_ready']: design.append('GPU_UNAVAILABLE')
    dimension('confirmatory_design', design, [] if facts['lineage_known'] else ['DERIVATIVE_FILENAME_GROUP_SCOPE_ONLY'])
    dimension('gpu', [] if facts['gpu_ready'] or facts['training_device'] != 'cuda' else ['GPU_UNAVAILABLE'],
              [] if facts['gpu_ready'] else ['GPU_NOT_VALIDATED'])
    dimension('novelty_review', (), [] if facts['literature_complete'] else ['LITERATURE_INCOMPLETE'])
    selected = dimensions['exploratory' if requested_mode == 'pilot' else 'confirmatory_design']['status']
    if dimensions['engineering']['status'] == 'blocked': selected = 'blocked'
    return dict(requested_mode=requested_mode, overall_status=selected, dimensions=dimensions,
        evidence_index_sha256=index_sha256, blockers=blockers,
        restrictions=sorted({code for d in dimensions.values() for code in d['reason_codes']}),
        m1_engineering_allowed=dimensions['engineering']['status'] != 'blocked',
        real_fit_scope_allowed=dimensions['exploratory']['status'] != 'blocked',
        novelty_claim='unverified; review completion never establishes novelty',
        clinical_claim='not established', confirmatory_frozen=False)
