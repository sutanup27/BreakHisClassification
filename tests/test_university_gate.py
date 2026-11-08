"""University pilot is explicit, scoped and cannot weaken historical gates."""
import copy
import unittest
from breakhis.research.gates import evaluate_gate
from breakhis.research.runner import _scope
from breakhis.research import records as r


class UniversityGateTests(unittest.TestCase):
    def facts(self, **changes):
        facts = {key: True for key in ('baseline_passed','source_consistent','evidence_complete',
            'data_ready','real_input','acceptance_complete','gpu_ready','scope_reviewed','lineage_scope_approved')}
        facts.update({key: False for key in ('protocol_pilot_ready','owners_assigned','lineage_known',
            'history_ready','literature_complete','protocol_design_ready','fold_support')})
        facts.update(training_device='cpu', university_pilot_reviewed=True)
        facts.update(changes)
        return facts

    def test_explicit_pilot_restricts_claims_without_documentation_block(self):
        gate = evaluate_gate(self.facts(), 'pilot', 'a'*64)
        self.assertEqual(gate['overall_status'], 'restricted')
        self.assertTrue(gate['real_fit_scope_allowed'])
        self.assertEqual(gate['dimensions']['confirmatory_design']['status'], 'blocked')
        self.assertIn('PILOT_PARTITION_VALIDATION_REQUIRED', gate['restrictions'])
        self.assertFalse(gate['confirmatory_frozen'])

    def test_legacy_gate_does_not_opt_in(self):
        facts = self.facts(); del facts['university_pilot_reviewed']
        gate = evaluate_gate(facts, 'pilot', 'a'*64)
        self.assertEqual(gate['overall_status'], 'blocked')
        self.assertNotIn('policy_profile', gate)

    def test_integrity_scope_source_trace_and_gpu_failures_still_block(self):
        for field in ('baseline_passed','source_consistent','evidence_complete','data_ready',
                      'real_input','acceptance_complete','scope_reviewed','lineage_scope_approved'):
            with self.subTest(field=field):
                gate = evaluate_gate(self.facts(**{field:False}), 'pilot', 'a'*64)
                self.assertFalse(gate['real_fit_scope_allowed'])
        gate = evaluate_gate(self.facts(training_device='cuda',gpu_ready=False), 'pilot', 'a'*64)
        self.assertEqual(gate['overall_status'], 'blocked')

    def test_profile_schema_requires_review_owner_scope_and_exact_claim(self):
        scope = dict(status='reviewed', reviewer='Sutanu', component_ids=['a'*64],
            lineage_exception=True, claims=['exploratory-only','filename-group-only','derivative-only'],
            evidence_refs=[dict(path='UNIVERSITY_PILOT_PROTOCOL.md',sha256='b'*64)],
            rationale='User-declared fresh university pilot', training_device='cuda',
            university_pilot=dict(version=1,dataset_source='researcher-declared-breakhis-website',
                prior_use='acknowledged',workflow='fresh-no-reuse',
                evaluation_claim='within-study-held-out',protocol='university-pilot-v1'))
        self.assertEqual(_scope(scope),scope)
        for change in ({'version':True},{'evaluation_claim':'historically-untouched'},{'extra':True}):
            changed=copy.deepcopy(scope); changed['university_pilot'].update(change)
            with self.assertRaises(r.RecordError): _scope(changed)
        changed=copy.deepcopy(scope); changed['status']='unreviewed'
        with self.assertRaises(r.RecordError): _scope(changed)


if __name__ == '__main__': unittest.main()
