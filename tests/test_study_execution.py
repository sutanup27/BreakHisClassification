"""M2.4 execution, retry, accounting, capacity and saved-only report acceptance."""
import contextlib
import copy
import io
import subprocess
import unittest
from unittest.mock import patch
import torch
from torch import nn
from breakhis import baseline_plan as b,study_execution as x,experiments as e,checkpoints as c,cli,training
from breakhis.models.vgg import VGG
import test_baseline_plan as fixtures


class StudyExecution(unittest.TestCase):
    save=fixtures.BaselinePlanTests.save
    save_study=fixtures.BaselinePlanTests.save_study
    def setUp(self):
        fixtures.BaselinePlanTests.setUp(self)
        self.config['training'].update(device='cpu',epochs=1,batch_size=2);self.save()
        self.study.update(dense_epochs=1,recovery_epochs=1,max_fit_jobs=100,available_wall_hours=8)
        self.study['evaluation_profiles']=self.study['evaluation_profiles'][:1]
        self.study['evaluation_profiles'][0].update(warmup=0,repeats=1)
        self.save_study();self.original_run=subprocess.run
        old_threads=torch.get_num_threads();torch.set_num_threads(2);self.addCleanup(torch.set_num_threads,old_threads)
        patch.object(VGG,'ARCH',[4,4,'M',4,4,'M',4,4,'M',4,4,'M']).start()
        def factory(arch,classes=2,channels=None,**kwargs):
            return VGG(classes,channels) if arch=='vgg' else nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,classes))
        patch.object(c,'create_model',side_effect=factory).start()
        def child(argv,**kwargs):
            if '-m' not in argv:return self.original_run(argv,**kwargs)
            cli.main(argv[argv.index('breakhis')+1:]);return subprocess.CompletedProcess(argv,0)
        patch('breakhis.development.subprocess.run',side_effect=child).start()
        self.root_study=b.prepare(self.study_file)

    def stage(self,name,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):return x.execute(self.study_file,name,**kwargs)

    def test_t14_all_public_stages_no_test_pixels_and_report_no_fit_or_inference(self):
        for row in self.rows:
            if row['split']=='test':(self.images/row['path']).unlink()
        self.stage('pilot');self.stage('parents');result=self.stage('candidates')
        self.assertTrue(result['complete_required']);self.assertEqual(len(result['rows']),28)
        ledger=e.read_json(self.root_study/'execution/ledger.json');self.assertEqual(len(ledger['selections']),3)
        for row in self.rows:
            if row['split']!='test':(self.images/row['path']).unlink()
        with patch.object(c,'create_model',side_effect=AssertionError('model factory in report')),patch.object(training,'train',side_effect=AssertionError('fit in report')):
            reported=self.stage('report')
        self.assertEqual(result['rows'],reported['rows']);self.assertFalse(result['scientific_approval'])
        for job in ledger['jobs'].values():
            self.assertFalse(job['unknown_work']);self.assertEqual(job['actual_optimizer_steps'],job['committed_optimizer_steps'])

    def test_t15_child_failure_and_retry_reuses_completed_pilots(self):
        real=e.run_experiment;count=0
        def fail(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:raise RuntimeError('simulated CUDA out of memory')
            return real(*args,**kwargs)
        with patch.object(e,'run_experiment',side_effect=fail),self.assertRaisesRegex(RuntimeError,'out of memory'):self.stage('pilot')
        ledger=e.read_json(self.root_study/'execution/ledger.json');self.assertEqual(sum(r['status']=='completed' for r in ledger['jobs'].values()),1)
        with self.assertRaisesRegex(ValueError,'resume'):self.stage('pilot')
        with patch.object(e,'run_experiment',wraps=real) as fit:self.stage('pilot',resume=True)
        self.assertEqual(fit.call_count,5)
        ledger=e.read_json(self.root_study/'execution/ledger.json');failed=next(r for r in ledger['jobs'].values() if len(r['attempts'])==2)
        self.assertEqual(failed['attempts'][0]['status'],'failed');self.assertEqual(failed['status'],'completed')

    def test_t15_interrupted_epoch_replay_is_charged(self):
        original=training.train;first=True
        def interrupt(*args,**kwargs):
            nonlocal first
            value=original(*args,**kwargs)
            if first:first=False;raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=interrupt),self.assertRaises(InterruptedError):self.stage('pilot',max_jobs=1)
        self.stage('pilot',resume=True,max_jobs=1)
        ledger=e.read_json(self.root_study/'execution/ledger.json');record=next(iter(ledger['jobs'].values()))
        self.assertEqual(record['actual_optimizer_steps'],2*record['committed_optimizer_steps'])
        self.assertEqual(record['replayed_optimizer_steps'],record['committed_optimizer_steps'])
        self.assertEqual(record['image_presentations'],2*record['committed_image_presentations'])

    def test_t15_real_nonzero_child_exit_is_recorded_and_retryable(self):
        import sys
        fallback=subprocess.run;first=True
        def child_exit(argv,**kwargs):
            nonlocal first
            if first and '-m' in argv and argv[argv.index('breakhis')+1]=='train':
                first=False
                return self.original_run([sys.executable,'-c','raise SystemExit(7)'],**kwargs)
            return fallback(argv,**kwargs)
        with patch('breakhis.development.subprocess.run',side_effect=child_exit),self.assertRaisesRegex(RuntimeError,'exit 7'):self.stage('pilot',max_jobs=1)
        result=self.stage('pilot',resume=True,max_jobs=1)
        self.assertEqual(len(result['rows']),1)
        record=next(iter(e.read_json(self.root_study/'execution/ledger.json')['jobs'].values()))
        self.assertEqual(len(record['attempts']),2);self.assertEqual(record['status'],'completed')

    def test_t15_prune_publication_failure_resumes_without_parent_refit(self):
        self.stage('pilot');self.stage('parents')
        before={k:r['selection_sha256'] for k,r in e.read_json(self.root_study/'execution/ledger.json')['jobs'].items()}
        original=cli.write_json;first=True
        def fail(path,*args,**kwargs):
            nonlocal first
            if first and str(path).endswith('prune.validation.json'):
                first=False;raise OSError('injected study pruning publication failure')
            return original(path,*args,**kwargs)
        with patch.object(cli,'write_json',side_effect=fail),self.assertRaises(OSError):self.stage('candidates',max_jobs=1)
        # Resume via the public session adapter reconciles only provably zero-fit
        # failures; unknown partial training remains protected by capacity checks.
        from tools.reconcile_study_accounting import reconcile
        corrected=reconcile(self.study_file);self.assertEqual(len(corrected),1)
        self.stage('candidates',resume=True,max_jobs=1)
        ledger=e.read_json(self.root_study/'execution/ledger.json')
        self.assertTrue(all(ledger['jobs'][k]['selection_sha256']==v for k,v in before.items()))
        child=next(r for k,r in ledger['jobs'].items() if k not in before)
        self.assertEqual(child['status'],'completed');self.assertEqual(len(child['attempts']),2)
        self.assertEqual(child['actual_optimizer_steps'],child['committed_optimizer_steps'])

    def test_t16_t18_capacity_and_session_boundary(self):
        self.stage('pilot',max_jobs=1)
        ledger=e.read_json(self.root_study/'execution/ledger.json');self.assertEqual(len(ledger['jobs']),1)
        self.assertEqual(ledger['sessions'][-1]['status'],'paused_at_job_boundary')
        with self.assertRaisesRegex(ValueError,'pilots required'):self.stage('parents')
        self.stage('pilot',resume=True)
        result=self.stage('parents',max_jobs=1)
        ledger=e.read_json(self.root_study/'execution/ledger.json')
        row=next(r for r in result['rows'] if r['kind']=='parent');record=ledger['jobs'][row['job_id']]
        self.assertEqual(row['incremental_steps'],record['actual_optimizer_steps'])
        self.assertTrue(all(r['cpu_peak_rss_bytes'] is None for r in record['resource_attempts']))
        costs=e.read_json(self.root_study/'execution/measured_cost_inputs.json');plan=e.read_json(self.root_study/'jobs.json')
        too_small=copy.deepcopy(plan);too_small['study']['max_fit_jobs']=1;costs['plan_sha256']=b.sha(too_small)
        with self.assertRaisesRegex(ValueError,'capacity'):b.require_capacity(too_small,costs)

    def test_t15_completed_artifact_or_frozen_plan_changes_refuse(self):
        self.stage('pilot',max_jobs=1)
        ledger=e.read_json(self.root_study/'execution/ledger.json');key=next(iter(ledger['jobs']))
        results=e.RUNS/key/'development/results.json';results.write_bytes(results.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.stage('pilot',resume=True)
        path=self.root_study/'configs'/next(iter(e.read_json(self.root_study/'jobs.json')['fit_jobs']))['job_id']
        frozen=self.root_study/'template.json';frozen.write_bytes(frozen.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):self.stage('report')


if __name__=='__main__':unittest.main()
