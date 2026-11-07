"""M0.5 semantics: M0-T16--T18; declarations are not scientific truth tests."""
import copy
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from breakhis.research import protocol as p, records as r
from breakhis.research.data_readiness import audit_data
from breakhis.research.exposure import resolve_exposure

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol=r.read_json(ROOT/'configs/research/protocol.initial.json')
        self.decisions=r.read_json(ROOT/'configs/research/decisions.initial.json')
        self.literature=r.read_json(ROOT/'configs/research/literature.initial.json')

    def topic(self,name): return next(item for item in self.protocol['topics'] if item['topic']==name)

    def test_t16_template_preserves_proposals_unknown_numbers_and_draft_lifecycle(self):
        before=copy.deepcopy(self.protocol)
        result=p.validate_protocol(self.protocol)
        self.assertEqual(self.protocol,before)
        self.assertEqual(result['lifecycle'],'draft')
        self.assertFalse(result['scientific_authorization'])
        self.assertTrue(all(topic['status']=='proposed' for topic in self.protocol['topics']))
        self.assertIsNone(self.topic('safeguards')['value']['specificity_margin'])
        self.assertIn('PROTOCOL_UNBOUND',result['restrictions'])
        self.assertEqual(p.validate_decisions(self.decisions,self.protocol)['topics'],17)

    def test_t17_missing_mandatory_topic_rejected(self):
        self.protocol['topics'].pop()
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_t17_missing_owner_deadline_or_evidence_rejected(self):
        original=copy.deepcopy(self.protocol)
        for field,value in [('owner',''),('freeze_by',None),('evidence_refs',[])]:
            self.protocol=copy.deepcopy(original)
            self.topic('aggregation')[field]=value
            with self.subTest(field=field),self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_missing_subpolicy_duplicate_topic_and_unknown_fields_rejected(self):
        self.topic('selection')['value'].pop('threshold')
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)
        self.setUp()
        self.protocol['topics'].append(copy.deepcopy(self.protocol['topics'][0]))
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)
        self.setUp()
        self.protocol['novelty_established']=True
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_agreed_cannot_hide_unresolved_null_numbers(self):
        self.topic('safeguards')['status']='agreed'
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_no_confirmatory_freeze_or_unsupported_lifecycle(self):
        for lifecycle in ('confirmatory_frozen','complete'):
            self.protocol['lifecycle']=lifecycle
            with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_lifecycle_cannot_skip_evidence_review_and_pending_choices(self):
        for lifecycle in ('pilot_ready','initial_design_reviewed'):
            self.protocol['lifecycle']=lifecycle
            self.protocol['reviewer']='fixture reviewer'
            with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_numeric_limits_and_seed_namespaces_validated(self):
        self.topic('safeguards')['value']['specificity_margin']=1.5
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)
        self.setUp()
        self.topic('hardware')['value']['cpu_threads']=True
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)
        self.setUp()
        self.topic('seeds')['value']['namespaces']=['split']
        with self.assertRaises(r.RecordError): p.validate_protocol(self.protocol)

    def test_decision_latest_must_match_protocol(self):
        self.decisions[0]['status']='blocked'
        with self.assertRaises(r.IntegrityError): p.validate_decisions(self.decisions,self.protocol)

    def test_revision_history_preserved_and_supersedes_checked(self):
        old=copy.deepcopy(self.decisions[0])
        latest=copy.deepcopy(old)
        latest.update(revision=2,supersedes=old['decision_id']+'@1',rationale='Revised rationale with retained history')
        self.protocol['topics'][0]['rationale']=latest['rationale']
        self.decisions.append(latest)
        self.assertEqual(p.validate_decisions(self.decisions,self.protocol)['revisions'],18)
        self.decisions.pop(0)
        with self.assertRaises(r.RecordError): p.validate_decisions(self.decisions,self.protocol)

    def test_unknown_decision_fields_and_nonfinite_history_rejected(self):
        self.decisions[0]['value']['input_resolution']=float('nan')
        with self.assertRaises(r.RecordError): p.validate_decisions(self.decisions,self.protocol)

    def test_t18_abstract_code_and_unavailable_cannot_be_full_method_reviews(self):
        for level in ('abstract_only','code_only','unavailable'):
            matrix=copy.deepcopy(self.literature)
            matrix[0].update(access_level=level,review_status='reviewed_methods',reviewer='fixture',review_date='2026-11-07')
            with self.subTest(level=level),self.assertRaises(r.RecordError): p.validate_literature(matrix)

    def test_curated_matrix_has_actual_levels_and_central_limits(self):
        result=p.validate_literature(self.literature)
        self.assertEqual(result['entries'],7)
        self.assertEqual(result['full_method_reviews'],1)
        self.assertFalse(result['novelty_review_complete'])
        self.assertTrue(all(gap['owner'] and gap['freeze_by'] for gap in result['central_gaps']))

    def test_unavailable_citation_fields_require_reasons(self):
        row=next(row for row in self.literature if row['literature_id']=='channelzip')
        row['unknown_reasons'].pop('authors')
        with self.assertRaises(r.RecordError): p.validate_literature(self.literature)

    def test_literature_missing_nearest_category_or_unknown_columns_rejected(self):
        matrix=[row for row in self.literature if 'class_aware' not in row['categories']]
        with self.assertRaises(r.RecordError): p.validate_literature(matrix)
        self.literature[0]['novelty_established']=True
        with self.assertRaises(r.RecordError): p.validate_literature(self.literature)

    def fixture(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo=Path(tmp.name).resolve()/'repo'; repo.mkdir()
        data=Path(tmp.name).resolve()/'data'
        image=data/'40X/benign/SOB_B_A-14-100AB-40-001.png'
        image.parent.mkdir(parents=True)
        Image.new('RGB',(12,8),(1,2,3)).save(image)
        run=r.create_run_directory(repo/'artifacts/research/m0','fixture')
        r.write_record(run,'baseline/git.json',r.new_record('git','fixture',dict(status='complete',source_snapshot_sha256='a'*64)))
        audit_data(data,run,'fixture',repo_root=repo)
        history=Path(tmp.name).resolve()/'history'; history.mkdir()
        resolve_exposure(run,'fixture',history_root=history,source_instance_id='fixture-source',events=[])
        return run

    def test_binding_preserves_topics_history_and_literature_statuses(self):
        run=self.fixture()
        before=copy.deepcopy(self.protocol)
        result=p.write_protocol_records(run,'fixture',protocol=self.protocol,decisions=self.decisions,literature=self.literature)
        bound=r.read_record(run,'governance/protocol.initial.json')['data']
        self.assertTrue(all(bound['bindings'].values()))
        self.assertEqual(bound['topics'],before['topics'])
        self.assertEqual(self.protocol,before)
        self.assertEqual(bound['lifecycle'],'draft')
        self.assertFalse(result['scientific_authorization'])
        self.assertFalse(result['history_review_ready'])
        self.assertEqual(r.read_record(run,'governance/decisions.json')['items'],self.decisions)
        with self.assertRaises(FileExistsError):
            p.write_protocol_records(run,'fixture',protocol=self.protocol,decisions=self.decisions,literature=self.literature)

    def test_stale_binding_refused_before_any_protocol_output(self):
        run=self.fixture()
        self.protocol['bindings']['audit_inventory_sha256']='0'*64
        with self.assertRaises(r.IntegrityError):
            p.write_protocol_records(run,'fixture',protocol=self.protocol,decisions=self.decisions,literature=self.literature)
        self.assertFalse((run/'governance/protocol.initial.json').exists())

    def test_supported_reviewed_lifecycle_remains_planning_not_scientific_authorization(self):
        self.topic('seeds')['value']['values']={name:index for index,name in enumerate(self.topic('seeds')['value']['namespaces'])}
        self.topic('safeguards')['value'].update(sensitivity_margin=0.01,specificity_margin=0.01)
        self.topic('budgets')['value'].update(dense_epochs=1,adaptation_epochs=1,recovery_epochs=1,candidates=1,latency_budgets=[1.0])
        self.topic('pretraining')['value'].update(recipe='synthetic fixture recipe',weights='from scratch fixture')
        for entry in self.protocol['topics']: entry['status']='agreed'
        self.protocol.update(lifecycle='initial_design_reviewed',reviewer='synthetic fixture reviewer',
                             bindings={key:'a'*64 for key in self.protocol['bindings']})
        result=p.validate_protocol(self.protocol)
        self.assertFalse(result['scientific_authorization'])
        self.assertFalse(result['confirmatory_frozen'])

    def test_example_config_matches_current_foundation_schema(self):
        run=self.fixture()
        repo=run.parents[3]
        config=r.read_json(ROOT/'configs/research/m0.example.json')
        config.update(repo_root=str(repo),data_root=str(repo.parent/'data'),output_root=str(repo/'artifacts/research/m0'))
        result=r.validate_m0_config(config,config_file=ROOT/'configs/research/m0.example.json')
        self.assertEqual(result['check_devices'],['cpu'])
        self.assertEqual(result['sources'],['cancer_binary'])


if __name__=='__main__': unittest.main()
