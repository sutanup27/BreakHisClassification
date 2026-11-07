"""M0.6: isolated synthetic orchestration, read-only verification and G0 precedence."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from breakhis.research import records as r
from breakhis.research import runner as x
from breakhis.research.gates import evaluate_gate

ROOT=Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def facts(self,**kwargs):
        facts={key:True for key in ('baseline_passed','source_consistent','evidence_complete','data_ready','lineage_known',
            'lineage_scope_approved','history_ready','scope_reviewed','protocol_pilot_ready','protocol_design_ready',
            'owners_assigned','literature_complete','gpu_ready','fold_support','real_input','acceptance_complete')}
        facts.update(training_device='cpu'); facts.update(kwargs)
        return facts

    def test_ready_design_is_not_confirmatory_freeze_or_novelty_established(self):
        gate=evaluate_gate(self.facts(),'confirmatory_design','a'*64)
        self.assertEqual(gate['overall_status'],'ready')
        self.assertFalse(gate['confirmatory_frozen'])
        self.assertIn('unverified',gate['novelty_claim'])

    def test_hard_failures_override_all_scientific_eligibility(self):
        for field in ('baseline_passed','source_consistent','evidence_complete'):
            gate=evaluate_gate(self.facts(**{field:False}),'pilot','a'*64)
            self.assertEqual(gate['overall_status'],'blocked')
            self.assertFalse(gate['m1_engineering_allowed'])

    def test_cpu_engineering_survives_gpu_failure(self):
        gate=evaluate_gate(self.facts(gpu_ready=False,training_device='cuda'),'pilot','a'*64)
        self.assertEqual(gate['dimensions']['engineering']['status'],'ready')
        self.assertEqual(gate['dimensions']['gpu']['status'],'blocked')
        self.assertEqual(gate['overall_status'],'blocked')

    def test_explicit_derivative_development_scope_is_restricted_not_confirmation(self):
        gate=evaluate_gate(self.facts(lineage_known=False,history_ready=False,literature_complete=False),'pilot','a'*64)
        self.assertEqual(gate['overall_status'],'restricted')
        self.assertEqual(gate['dimensions']['confirmatory_design']['status'],'blocked')

    def test_unknown_scope_cannot_silently_pass_pilot(self):
        gate=evaluate_gate(self.facts(scope_reviewed=False,lineage_known=False,lineage_scope_approved=False,history_ready=False),'pilot','a'*64)
        self.assertEqual(gate['overall_status'],'blocked')

    def test_data_and_trace_failures_keep_only_restricted_engineering(self):
        gate=evaluate_gate(self.facts(data_ready=False,acceptance_complete=False),'pilot','a'*64)
        self.assertEqual(gate['dimensions']['engineering']['status'],'restricted')
        self.assertEqual(gate['overall_status'],'blocked')


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.repo=Path(cls.tmp.name).resolve()/'repo'; cls.repo.mkdir()
        def put(path,text):
            path=cls.repo/path; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding='utf-8')
        put('.gitignore','artifacts/\n__pycache__/\n*.pyc\n')
        for path in ('Train.py','Main.py','Test.py','VGG.py','breakhis/__init__.py','breakhis/cli.py','breakhis/training.py','breakhis/checkpoints.py'):
            put(path,'"""Synthetic orchestration fixture module, not production training."""\n')
        put('breakhis/__main__.py','import argparse\np=argparse.ArgumentParser(); p.add_argument("command"); p.parse_args()\n')
        put('tools/check_environment.py','import sys,json\nprint(json.dumps({"device_checked":sys.argv[-1],"synthetic_fixture":True}))\n')
        put('tests/test_fixture.py','import unittest\nclass Fixture(unittest.TestCase):\n def test_fixture_identity(self):\n  self.assertEqual("synthetic-only".split("-"),["synthetic","only"])\n')
        inputs=cls.repo/'configs/research'; inputs.mkdir(parents=True)
        for path in (ROOT/'configs/research').glob('*.json'): shutil.copyfile(path,inputs/path.name)
        data=Path(cls.tmp.name).resolve()/'data'
        for label,code,slide,color in [('benign','B_A','100AB',(1,2,3)),('malignant','M_DC','200CD',(4,5,6))]:
            path=data/f'40X/{label}/SOB_{code}-14-{slide}-40-001.png'; path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(12,8),color).save(path)
        config=r.read_json(inputs/'m0.example.json')
        config.update(repo_root=str(cls.repo),data_root=str(data),output_root=str(cls.repo/'artifacts/research/m0'),run_kind='synthetic')
        cls.config_file=inputs/'m0.example.json'; cls.config_file.write_text(json.dumps(config),encoding='utf-8')
        for args in [('init','-q'),('config','user.name','Fixture'),('config','user.email','fixture@example.invalid'),('add','.'),('commit','-qm','synthetic fixture')]:
            subprocess.run(['git','-C',str(cls.repo),*args],check=True,capture_output=True)
        cls.result=x.run_m0(cls.config_file,'synthetic-complete')
        cls.evidence_run=Path(cls.result['run_dir'])
        evidence=os.environ.get('BREAKHIS_M0_EVIDENCE')
        if evidence:
            target=(ROOT/evidence).resolve()
            if not target.is_relative_to(ROOT/'artifacts'): raise AssertionError('CI evidence must stay under artifacts')
            target.mkdir(parents=True,exist_ok=True)
            shutil.copytree(cls.evidence_run,target/Path(cls.tmp.name).name)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def clone(self):
        target=Path(tempfile.mkdtemp(dir=self.tmp.name))/'archive'
        shutil.copytree(self.evidence_run,target)
        return target

    def test_t23_isolated_orchestration_and_scientific_blockers(self):
        self.assertEqual(self.result['state'],'completed')
        self.assertTrue(self.result['validated'])
        self.assertEqual(self.result['exit_code'],1)
        self.assertEqual(self.result['gate']['overall_status'],'blocked')
        self.assertFalse(self.result['gate']['real_fit_scope_allowed'])
        self.assertEqual(len(r.read_record(self.evidence_run,'acceptance/tests.json')['items']),26)

    def test_t24_verifier_never_executes_or_writes(self):
        before={p.relative_to(self.evidence_run).as_posix():(r.file_sha256(p),p.stat().st_mtime_ns) for p in self.evidence_run.rglob('*') if p.is_file()}
        with patch('subprocess.Popen',side_effect=AssertionError('Verifier launched command')),patch.object(Path,'mkdir',side_effect=AssertionError('Verifier wrote directory')):
            first=x.verify_run(self.evidence_run); second=x.verify_run(self.evidence_run)
        self.assertEqual(first,second)
        after={p.relative_to(self.evidence_run).as_posix():(r.file_sha256(p),p.stat().st_mtime_ns) for p in self.evidence_run.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_t21_tampering_and_incomplete_index_rejected(self):
        archive=self.clone()
        (archive/'audit/inventory.jsonl').write_text('{}\n')
        with self.assertRaises(r.RecordError): x.verify_run(archive)
        archive=self.clone()
        (archive/'logs/regressions.stderr.log').unlink()
        with self.assertRaises(r.RecordError): x.verify_run(archive)
        archive=self.clone()
        index=r.read_record(archive,'acceptance/evidence_index.json')
        index['items']=[entry for entry in index['items'] if entry['path']!='audit/audit.json']
        r.write_record(archive,'acceptance/evidence_index.json',index,overwrite=True)
        with self.assertRaises(r.RecordError): x.verify_run(archive)

    def test_t22_relocation_is_read_only_and_preserves_verification(self):
        archive=self.clone()
        self.assertEqual(x.verify_run(archive),x.verify_run(self.evidence_run))

    def test_gate_tampering_and_semantically_forged_ready_rejected(self):
        archive=self.clone()
        gate=r.read_record(archive,'acceptance/gate_g0.json')
        gate['data']['overall_status']='ready'
        r.write_record(archive,'acceptance/gate_g0.json',gate,overwrite=True)
        receipt=r.read_record(archive,'acceptance/receipt.json')
        receipt['data']['artifacts']['acceptance/gate_g0.json']=r.file_sha256(archive/'acceptance/gate_g0.json')
        r.write_record(archive,'acceptance/receipt.json',receipt,overwrite=True)
        with self.assertRaises(r.IntegrityError): x.verify_run(archive)

    def test_unknown_schema_extra_files_and_row_declarations_rejected(self):
        archive=self.clone(); (archive/'unindexed.txt').write_text('extra')
        with self.assertRaises(r.IntegrityError): x.verify_run(archive)
        archive=self.clone(); index=r.read_record(archive,'acceptance/evidence_index.json'); index['schema_version']=99
        (archive/'acceptance/evidence_index.json').write_text(json.dumps(index))
        with self.assertRaises(r.RecordError): x.verify_run(archive)
        archive=self.clone(); index=r.read_record(archive,'acceptance/evidence_index.json')
        row=next(item for item in index['items'] if item['path']=='audit/inventory.jsonl')
        row.pop('row_schema_version'); row.pop('row_record_type')
        r.write_record(archive,'acceptance/evidence_index.json',index,overwrite=True)
        with self.assertRaises(r.RecordError): x.verify_run(archive)

    def test_nonempty_run_and_bad_config_refused_before_checks(self):
        with self.assertRaises(r.RecordError): x.run_m0(self.config_file,'synthetic-complete')
        bad=r.read_json(self.config_file); bad['unexpected']='shell command'
        path=self.repo/'bad.json'; path.write_text(json.dumps(bad))
        with self.assertRaises(r.RecordError): x.run_m0(path,'should-not-exist')
        self.assertFalse((self.repo/'artifacts/research/m0/should-not-exist').exists())

    def test_cli_help_and_config_error_exit_codes(self):
        script=ROOT/'tools/run_m0.py'
        result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,str(script),'--config',str(self.config_file),'--run-id','../escape'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2,result.stderr)

    def test_audit_timeout_retains_partial_evidence_and_never_reports_ready(self):
        from breakhis.research import m0
        original=m0._execute
        def execute(run,check_id,argv,cwd,device,timeout,env):
            if check_id=='data-audit':
                argv=[sys.executable,'-c','import time; print("partial audit fixture",flush=True); time.sleep(30)']
                timeout=1
            return original(run,check_id,argv,cwd,device,timeout,env)
        with patch.object(m0,'_execute',side_effect=execute):
            result=x.run_m0(self.config_file,'timeout-fixture')
        run=Path(result['run_dir'])
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['exit_code'],1)
        self.assertFalse(result['validated'])
        self.assertIn('partial audit fixture',(run/'logs/data-audit.stdout.log').read_text())
        self.assertTrue((run/'failure.json').is_file())
        with self.assertRaises(r.RecordError): x.verify_run(run)

    def test_historical_references_are_frozen_and_verify_without_original_files(self):
        history=Path(self.tmp.name)/'history-snapshot'; history.mkdir()
        rows=r.read_jsonl(self.evidence_run/'audit/inventory.jsonl')
        manifest=dict(schema_version=1,rows=[dict(source='cancer_binary',path=row['relative_path'],
            patient_id=row['filename_group_id'],magnification=row['magnification'],sha256=row['raw_sha256'],
            pixel_sha256=row['decoded_pixel_sha256'],split='test') for row in rows])
        (history/'manifest.json').write_text(json.dumps(manifest)); (history/'access.log').write_text('Synthetic metadata fixture only')
        exposure=r.read_json(ROOT/'configs/research/exposure.initial.json')
        exposure.update(source_instance_id='fixture-history',history_root=str(history),events=[dict(event_id='fixture-eval',
            artifact=dict(path='access.log',sha256=r.file_sha256(history/'access.log')),
            manifest=dict(path='manifest.json',sha256=r.file_sha256(history/'manifest.json')),
            source='cancer_binary',source_instance_id='fixture-history',source_kind='real',selected_partition='test',
            magnification_filter=['40X'],access_type='test_metric_inspection',accessed_paths=None,
            evidence_source='log',occurred_at=None,reviewer_note='Synthetic fixture modeling real access')])
        path=self.repo/'configs/history-exposure.json'; path.write_text(json.dumps(exposure))
        config=r.validate_m0_config(r.read_json(self.config_file),config_file=self.config_file)
        config['exposure_path']=str(path)
        config_file=self.repo/'configs/history-config.json'; config_file.write_text(json.dumps(config))
        result=x.run_m0(config_file,'history-fixture')
        self.assertTrue(result['validated'])
        (history/'access.log').unlink(); (history/'manifest.json').unlink()
        self.assertTrue(x.verify_run(result['run_dir'])['validated'])

    def test_jsonl_schema_version_is_explicit_and_validated(self):
        path=self.evidence_run/'audit/inventory.jsonl'
        entry=r.artifact_entry(self.evidence_run,'audit/inventory.jsonl',record_type='blob',operation_id='audit',row_record_type='image_inventory_row')
        self.assertEqual(entry['row_schema_version'],1)
        entry['row_schema_version']=99
        with self.assertRaises(r.RecordError): r.validate_artifact_entry(entry)
        self.assertTrue(r.read_jsonl(path))

    def test_operator_interrupt_stops_before_audit_and_preserves_failed_run(self):
        with patch.object(x.m0,'capture_baseline',side_effect=KeyboardInterrupt), \
             patch.object(x.m0,'_execute',side_effect=AssertionError('Audit started after interrupt')):
            result=x.run_m0(self.config_file,'interrupted-fixture')
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['exit_code'],1)
        self.assertFalse(result['validated'])
        run=Path(result['run_dir'])
        self.assertFalse((run/'audit').exists())
        self.assertEqual(r.read_record(run,'failure.json')['data']['error_type'],'KeyboardInterrupt')


if __name__=='__main__': unittest.main()
