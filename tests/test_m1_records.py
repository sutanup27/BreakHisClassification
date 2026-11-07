"""M1.1 acceptance for M1-T01--T03. Synthetic metadata only; no M1 fit/fold generation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from breakhis.research import records as r, m1_records as m

ROOT=Path(__file__).resolve().parents[1]


def row(path='40X/benign/example.png',image='a'*64):
    return dict(schema_version=1,record_type='m1.input_row',source_instance_id='synthetic-source',
        image_id=image,relative_path=path,raw_sha256='b'*64,decoded_pixel_sha256='c'*64,
        filename_group_id='14-100AB',component_id='d'*64,class_label='benign',subtype='adenosis',
        magnification='40X',status='included',unavailable_reasons={})


class M1ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name).resolve()/'repo'; self.repo.mkdir()
        self.config=r.read_json(ROOT/'configs/research/m1.example.json')
        self.config.update(repo_root=str(self.repo),output_root=str(self.repo/'artifacts/research/m1'),m0_archive=str(self.repo/'artifacts/research/m0/parent'))
        self.config_file=self.repo/'configs/m1.json'

    def test_import_is_stdlib_only_and_has_no_io(self):
        code='from pathlib import Path\nfrom unittest.mock import patch\nimport sys\nwith patch.object(Path,"mkdir",side_effect=AssertionError("mkdir")), patch.object(Path,"open",side_effect=AssertionError("open")):\n import breakhis.research.m1_records\nassert "torch" not in sys.modules\n'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_distinct_registries_without_mutating_m0_defaults(self):
        original=dict(r.RECORD_SHAPES)
        for kind,shape in m.RECORD_SHAPES.items():
            record=m.new_record(kind,'m1-test',{} if shape=='data' else [])
            m.validate_record(record,expected_type=kind)
            with self.assertRaises(r.RecordError): r.validate_record(record)
        self.assertEqual(r.RECORD_SHAPES,original)
        with self.assertRaises(r.RecordError): m.validate_record(r.new_record('run','m0',{}))

    def test_t02_record_versions_extra_fields_and_nonfinite_values_rejected(self):
        for change in (dict(schema_version=2),dict(unexpected=True),dict(record_type='m1.unknown')):
            record=m.new_record('m1.run','m1-test',{}); record.update(change)
            with self.assertRaises(r.RecordError): m.validate_record(record)
        with self.assertRaises(r.RecordError): m.new_record('m1.run','m1-test',{'bad':float('nan')})

    def test_t02_atomic_roundtrip_no_clobber_paths_and_failure_cleanup(self):
        run=r.create_run_directory(self.repo/'artifacts/research/m1','test')
        record=m.new_record('m1.run','test',{'state':'created'})
        m.write_record(run,'run.json',record)
        self.assertEqual(m.read_record(run,'run.json'),record)
        with self.assertRaises(FileExistsError): m.write_record(run,'run.json',record)
        with self.assertRaises(r.RecordError): m.write_record(run,'../escape.json',record)
        with patch.object(r.os,'link',side_effect=OSError('injected publication failure')):
            with self.assertRaises(OSError): m.write_record(run,'new.json',record)
        self.assertFalse((run/'new.json').exists())
        self.assertFalse(list(run.glob('*.tmp')))

    def test_t02_template_is_inactive_until_archive_pins_are_supplied(self):
        validated=m.validate_config(self.config,config_file=self.config_file)
        self.assertEqual(validated['mode'],'engineering')
        self.assertFalse((self.repo/'artifacts').exists())
        with self.assertRaises(m.PermissionDenied): m.validate_config(self.config,config_file=self.config_file,require_pins=True)

    def test_t02_unknown_config_invalid_namespaces_and_unsafe_outputs(self):
        for change in (dict(unknown='shell string'),dict(spec_id='M0-SPEC-001'),dict(output_root=str(self.repo/'outside')),
                       dict(mode='training'),dict(device='auto'),dict(deterministic=1),dict(loader={'batch_size':2,'num_workers':2})):
            config=copy.deepcopy(self.config); config.update(change)
            with self.subTest(change=change),self.assertRaises(r.RecordError): m.validate_config(config,config_file=self.config_file)
        self.config['seeds']['loader']=True
        with self.assertRaises(r.RecordError): m.validate_config(self.config,config_file=self.config_file)

    def test_t02_overlap_and_role_policy_fail_before_writes(self):
        self.config['data_root']=str(self.repo/'artifacts/research/m1')
        with self.assertRaises(r.RecordError): m.validate_config(self.config,config_file=self.config_file)
        self.config['data_root']=None; self.config['fold_policy']['inner_roles'].append('outer_test')
        with self.assertRaises(r.RecordError): m.validate_config(self.config,config_file=self.config_file)
        self.assertFalse((self.repo/'artifacts').exists())

    def test_t03_jsonl_identity_order_relocation_policy_and_content(self):
        rows=[row(),row('100X/benign/second.png','e'*64)]
        policy={'hash_version':1,'grouping':'full suffix'}
        expected=m.identity_sha256(rows,policy)
        self.assertEqual(m.identity_sha256(list(reversed(rows)),policy),expected)
        run=r.create_run_directory(self.repo/'artifacts/research/m1','one')
        m.write_input_rows(run,'inputs/images.jsonl',rows,policy=policy)
        archive=self.repo/'relocated'; shutil.copytree(run,archive)
        self.assertEqual(m.identity_sha256(m.read_input_rows(archive,'inputs/images.jsonl'),policy),expected)
        changed=copy.deepcopy(rows); changed[0]['raw_sha256']='f'*64
        self.assertNotEqual(m.identity_sha256(changed,policy),expected)
        self.assertNotEqual(m.identity_sha256(rows,{'hash_version':2}),expected)
        with self.assertRaises(r.RecordError): m.identity_sha256([rows[0],rows[0]],policy)

    def test_t02_jsonl_unknown_schema_type_case_collision_and_nonfinite_rejected(self):
        for change in (dict(schema_version=2),dict(record_type='m1.prediction_row'),dict(extra='unknown'),dict(raw_sha256=None)):
            value=row(); value.update(change)
            with self.assertRaises(r.RecordError): m.validate_input_row(value)
        value=row('40X/benign/EXAMPLE.png','e'*64)
        with self.assertRaises(r.RecordError): m.identity_sha256([row(),value],{})

    def test_t03_config_semantics_excludes_locator_roots_not_policy(self):
        first=m.validate_config(self.config,config_file=self.config_file)
        second=copy.deepcopy(first)
        for key in ('repo_root','output_root','data_root','m0_archive'): second[key]='different physical location'
        self.assertEqual(m.config_identity(first),m.config_identity(second))
        second['seeds']['training']+=1
        self.assertNotEqual(m.config_identity(first),m.config_identity(second))

    def test_t02_output_junction_cannot_be_hidden_by_path_resolution(self):
        anchor=self.repo/'artifacts/research/m1'; target=anchor/'actual'; target.mkdir(parents=True)
        alias=anchor/'alias'; junction=False
        try:
            alias.symlink_to(target,target_is_directory=True)
        except OSError:
            if os.name!='nt': raise
            self.assertTrue(alias.absolute().is_relative_to(self.repo))
            self.assertTrue(target.resolve().is_relative_to(self.repo))
            result=subprocess.run(['cmd','/c','mklink','/J',str(alias),str(target)],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr); junction=True
        try:
            self.config['output_root']=str(alias)
            with self.assertRaises(r.RecordError): m.validate_config(self.config,config_file=self.config_file)
        finally:
            if junction: os.rmdir(alias)
            else: alias.unlink()


class M1EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the actual isolated M0 CLI/worker fixture instead of inventing a ready Boolean.
        import test_m0_runner as fixture
        cls.fixture=fixture.RunnerTests
        cls.fixture.setUpClass()
        cls.archive=cls.fixture.evidence_run
        cls.pins=m.inspect_m0(cls.archive)

    @classmethod
    def tearDownClass(cls): cls.fixture.tearDownClass()

    def entry(self,archive=None):
        return m.VerifiedEntry(archive or self.archive,expected_index_sha256=self.pins['m0_index_sha256'],expected_receipt_sha256=self.pins['m0_receipt_sha256'])

    def cloned(self):
        target=Path(tempfile.mkdtemp(dir=self.fixture.tmp.name))/'archive'
        shutil.copytree(self.archive,target)
        return target

    def config(self):
        config=r.read_json(ROOT/'configs/research/m1.example.json')
        config.update(repo_root=str(self.fixture.repo),output_root=str(self.fixture.repo/'artifacts/research/m1'),m0_archive=str(self.archive))
        for key in m._PINS: config[key]=self.pins[key]
        return m.validate_config(config,config_file=self.fixture.config_file)

    def call(self,entry,**kwargs):
        request=dict(mode='engineering',source_kind='synthetic',source_instance_id='independent-m1-fixture',operation='read_metadata')
        request.update(kwargs)
        return m.authorize(entry,**request)

    def test_t01_verified_blocked_science_parent_still_allows_synthetic_engineering(self):
        result=self.call(self.entry())
        self.assertTrue(result['allowed']); self.assertFalse(result['scientific_authorization'])
        self.assertIn('PROTOCOL_PENDING',result['restrictions'])

    def test_t01_wrong_pins_missing_gate_and_changed_handoff_rejected(self):
        with self.assertRaises(r.IntegrityError): m.VerifiedEntry(self.archive,expected_index_sha256='0'*64,expected_receipt_sha256=self.pins['m0_receipt_sha256'])
        for relative in ('acceptance/gate_g0.json','acceptance/m1_handoff.json','audit/duplicate_components.json'):
            archive=self.cloned(); (archive/relative).unlink()
            with self.subTest(relative=relative),self.assertRaises((r.RecordError,OSError)): self.entry(archive)

    def test_t01_copied_flags_mutable_snapshots_and_late_tampering_do_not_grant_access(self):
        with self.assertRaises(m.PermissionDenied): self.call({'engineering_allowed':True})
        archive=self.cloned(); entry=self.entry(archive)
        copy_of_proof=entry.snapshot; copy_of_proof['design_allowed']=True
        self.assertFalse(entry.snapshot['design_allowed'])
        with self.assertRaises(AttributeError): entry._proof=b'{}'
        (archive/'audit/audit.json').write_text('{}')
        with self.assertRaises(r.RecordError): self.call(entry)

    def test_role_guards_reject_outer_construction_iteration_and_export_before_factory(self):
        entry=self.entry()
        for operation in ('construct_dataset','iterate_images','export_predictions'):
            factory=Mock(side_effect=AssertionError('Outer adapter invoked'))
            with self.subTest(operation=operation),self.assertRaises(m.PermissionDenied):
                m.guarded_call(entry,factory=factory,mode='engineering',source_kind='synthetic',source_instance_id='m1-fixture',operation=operation,role='outer_test')
            factory.assert_not_called()

    def test_synthetic_dev_role_callback_can_run_but_wrong_calibration_role_cannot(self):
        entry=self.entry(); factory=Mock(return_value='synthetic adapter')
        self.assertEqual(m.guarded_call(entry,factory=factory,mode='engineering',source_kind='synthetic',source_instance_id='m1-fixture',operation='construct_dataset',role='development_selection'),'synthetic adapter')
        with self.assertRaises(m.PermissionDenied): self.call(entry,operation='calibrate_threshold',role='development_selection')
        self.assertTrue(self.call(entry,operation='calibrate_threshold',role='development_calibration')['allowed'])
        with self.assertRaises(m.PermissionDenied): self.call(entry,operation='export_predictions',role='inner_train')

    def test_diagnostic_reads_metadata_only_and_real_design_remains_blocked(self):
        entry=self.entry(); source=entry.bindings['source_instance_id']
        self.assertTrue(self.call(entry,mode='diagnostic',source_kind='real',source_instance_id=source,operation='diagnose')['allowed'])
        for operation,role in (('plan_folds',None),('fit','inner_train'),('iterate_images','inner_train'),('export_predictions','development_selection')):
            with self.assertRaises(m.PermissionDenied): self.call(entry,mode='diagnostic',source_kind='real',source_instance_id=source,operation=operation,role=role)
        with self.assertRaises(m.PermissionDenied): self.call(entry,mode='real_design',source_kind='real',source_instance_id=source,operation='plan_folds',component_ids=entry.snapshot['component_ids'])

    def test_synthetic_instance_collision_unknown_role_and_gpu_denied(self):
        entry=self.entry()
        with self.assertRaises(m.PermissionDenied): self.call(entry,source_instance_id=entry.bindings['source_instance_id'])
        with self.assertRaises(m.PermissionDenied): self.call(entry,component_ids=entry.snapshot['component_ids'])
        with self.assertRaises(m.PermissionDenied): self.call(entry,operation='construct_dataset',role='test')
        with self.assertRaises(m.PermissionDenied): self.call(entry,operation='benchmark',device='cuda')
        self.assertTrue(self.call(entry,operation='validate_oof_fixture',role='outer_test')['allowed'])

    def test_t02_run_creation_never_marks_complete_and_rejects_stale_governance(self):
        config=self.config()
        run,entry=m.create_run(config,'m1-created')
        self.assertEqual(m.read_record(run,'run.json')['data']['state'],'created')
        self.assertFalse(m.read_record(run,'run.json')['data']['scientific_authorization'])
        with self.assertRaises(r.RecordError): m.create_run(config,'m1-created')
        config['protocol_sha256']='0'*64
        with self.assertRaises(r.IntegrityError): m.create_run(config,'m1-no-stale')
        self.assertFalse((Path(config['output_root'])/'m1-no-stale').exists())

    def test_t02_publication_failure_preserves_partial_failure_record(self):
        config=self.config(); original=m.write_record
        def fail(run,relative,record,**kwargs):
            if relative=='config.resolved.json': raise OSError('injected config write failure')
            return original(run,relative,record,**kwargs)
        with patch.object(m,'write_record',side_effect=fail):
            with self.assertRaises(OSError): m.create_run(config,'m1-partial')
        run=Path(config['output_root'])/'m1-partial'
        self.assertEqual(m.read_record(run,'run.json')['data']['state'],'failed')
        self.assertTrue((run/'failure.json').exists())


if __name__=='__main__': unittest.main()
