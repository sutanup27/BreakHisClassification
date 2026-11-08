"""M1-T09--T14: numerical resume, publication recovery and development isolation."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch
from breakhis import experiments as e, paths
from breakhis import checkpoints as c, cli, training, fit_boundary as boundary
import test_experiments as fixtures


class ExecutionTests(unittest.TestCase):
    save=fixtures.ExperimentTests.save

    def setUp(self):
        fixtures.ExperimentTests.setUp(self)
        threads=torch.get_num_threads(); torch.set_num_threads(2); self.addCleanup(torch.set_num_threads,threads)
        self.config['training'].update(epochs=4,batch_size=2,device='cpu',seed=22,optimizer='adam',scheduler='none')
        self.config['evaluation'].update(warmup=0,repeats=1)
        self.save()
        self.addCleanup(patch.stopall)
        patch.object(paths,'CHECKPOINT_ROOT',self.root/'checkpoint').start()
        patch.object(e,'RUNS',self.root/'runs').start()
        patch.object(c,'create_model',side_effect=self.model).start()
        self.real_run=subprocess.run

    @staticmethod
    def model(*args,**kwargs):
        return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
            torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,2))

    def args(self,identifier='fit',**options):
        result=['train','--dataset','breakhis','--manifest',str(self.root/'manifest.json'),
            '--data-root',str(self.images),'--device','cpu','--epochs','4','--batch-size','2',
            '--deterministic','--seed','22','--optimizer','adam','--scheduler','none',
            '--selection-metric','group_balanced_accuracy','--experiment-id',identifier]
        for key,value in options.items():
            flag='--'+key.replace('_','-')
            result[result.index(flag)+1]=str(value)
        return result

    def directory(self,identifier='fit'):
        return self.root/f'checkpoint/binary/40X/vgg/{identifier}'

    def fit(self,identifier='fit',**options):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            cli.main(self.args(identifier,**options))
        return c.read_checkpoint(self.directory(identifier)/'train-last.pth')

    def resume(self,identifier='fit',extra=()):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            cli.main(['train','--dataset','breakhis','--resume',str(self.directory(identifier)/'train-last.pth'),*extra])
        return c.read_checkpoint(self.directory(identifier)/'train-last.pth')

    @staticmethod
    def numerical(state):
        state=copy.deepcopy(state)
        for row in state['history']:
            row.pop('training_seconds'); row.pop('validation_seconds')
        return state

    def child(self,argv,**kwargs):
        if '-m' not in argv: return self.real_run(argv,**kwargs)
        cli.main(argv[argv.index('breakhis')+1:])
        return subprocess.CompletedProcess(argv,0)

    def run_dev(self,identifier='dev',resume=False):
        with patch('breakhis.development.subprocess.run',side_effect=self.child), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return e.run_experiment(self.file,run_id=identifier,resume=resume)

    def test_t09_matching_fresh_seeds_reproduce_and_changed_seed_changes_weights(self):
        a=self.fit('a'); b=self.fit('b'); changed=self.fit('changed',seed=23)
        self.assertTrue(boundary.same(a['state_dict'],b['state_dict']))
        self.assertFalse(boundary.same(a['state_dict'],changed['state_dict']))
        self.assertEqual(changed['config']['seed'],23)
        self.assertTrue(changed['training_state']['random_state']['deterministic_algorithms'])

    def test_t10_sgd_cosine_and_adam_constant_exact_resume_with_ties(self):
        # Force score/loss ties and a worse epoch: epoch 2 wins, epoch 4 must not replace it.
        from breakhis import reporting
        real_collect=reporting.collect_predictions
        for optimizer,scheduler in [('sgd','cosine'),('adam','none')]:
            counts={'epoch':0}
            def collect(*args,**kwargs):
                result=list(real_collect(*args,**kwargs)); counts['epoch']+=1
                result[0]=dict(result[0],loss=[.9,.8,.7,.8][counts['epoch']-1]); return tuple(result)
            def groups(*args,**kwargs):
                return dict(available=True,balanced_accuracy=[.5,.5,.4,.5][counts['epoch']-1])
            with patch.object(reporting,'collect_predictions',side_effect=collect), \
                 patch.object(reporting,'magnification_balanced_metrics',side_effect=groups):
                full=self.fit('full-'+optimizer,optimizer=optimizer,scheduler=scheduler)
                counts['epoch']=0
                original=training.train
                def stop(*args,**kwargs):
                    value=original(*args,**kwargs)
                    if kwargs['epoch']==2: raise KeyboardInterrupt
                    return value
                with patch.object(training,'train',side_effect=stop):
                    partial=self.fit('resume-'+optimizer,optimizer=optimizer,scheduler=scheduler)
                self.assertEqual(partial['epoch'],1)
                counts['epoch']=1
                resumed=self.resume('resume-'+optimizer)
            self.assertTrue(boundary.same(full['state_dict'],resumed['state_dict']))
            self.assertTrue(boundary.same(self.numerical(full['training_state']),self.numerical(resumed['training_state'])))
            self.assertEqual(resumed['training_state']['best']['epoch'],2)
            best=c.read_checkpoint(self.directory('resume-'+optimizer)/'train.pth')
            self.assertEqual(best['epoch'],2)
            rows=resumed['training_state']['history']
            self.assertEqual([r['optimizer_steps'] for r in rows],[1]*4)
            self.assertEqual(rows[0]['learning_rate'],.001)
            self.assertEqual(rows[1]['learning_rate'],rows[0]['next_learning_rate'])
            self.assertTrue(all(r['validation_seconds']>=0 and r['training_seconds']>=0 for r in rows))

    def interrupted(self,identifier='fit',epoch=2):
        original=training.train
        def stop(*args,**kwargs):
            value=original(*args,**kwargs)
            if kwargs['epoch']==epoch: raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=stop): return self.fit(identifier)

    def test_t11_epoch_zero_restart_matches_fresh(self):
        full=self.fit('full'); partial=self.interrupted(epoch=1)
        self.assertEqual(partial['epoch'],0)
        resumed=self.resume()
        self.assertTrue(boundary.same(full['state_dict'],resumed['state_dict']))
        self.assertTrue(boundary.same(self.numerical(full['training_state']),self.numerical(resumed['training_state'])))

    def test_t11_missing_best_partial_weights_changed_runtime_and_config_refused(self):
        self.interrupted()
        directory=self.directory(); best=directory/'train.pth'; saved=best.read_bytes(); best.unlink()
        with self.assertRaises(SystemExit): self.resume()
        best.write_bytes(saved)
        with self.assertRaises(SystemExit): self.resume(extra=['--seed','99'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            cli.main(['train','--dataset','breakhis','--resume',str(directory/'train-interrupted.pth')])
        latest=directory/'train-last.pth'; payload=c.read_checkpoint(latest)
        payload['runtime']['python']='changed'; c.save_payload(latest,payload)
        with self.assertRaises(SystemExit): self.resume()

    def test_t11_changed_source_and_device_refused(self):
        self.interrupted(); latest=self.directory()/'train-last.pth'; original=c.read_checkpoint(latest)
        for field,value in [('source_sha256',{}),('resolved_device','cuda'),('cpu_threads',999)]:
            payload=copy.deepcopy(original); payload['config'][field]=value; c.save_payload(latest,payload)
            with self.subTest(field=field),self.assertRaises(SystemExit): self.resume()

    def test_t13_publication_after_commit_repairs_best_and_history(self):
        full=self.fit('full')
        original=boundary.reconcile
        def fail(output,payload):
            if payload['epoch']==2: raise OSError('injected best/history publication failure')
            return original(output,payload)
        with patch.object(boundary,'reconcile',side_effect=fail),self.assertRaises(OSError): self.fit()
        latest=c.read_checkpoint(self.directory()/'train-last.pth'); self.assertEqual(latest['epoch'],2)
        self.assertTrue((self.directory()/'train.epoch-pending.json').exists())
        resumed=self.resume()
        self.assertTrue(boundary.same(full['state_dict'],resumed['state_dict']))
        self.assertFalse((self.directory()/'train.epoch-pending.json').exists())

    def test_t13_publication_before_commit_rolls_back_to_previous_boundary(self):
        full=self.fit('full')
        save_latest=c.save_checkpoint
        def fail(path,*args,**kwargs):
            if Path(path).name=='train-last.pth' and args[3]==2:
                raise OSError('injected boundary publication failure')
            return save_latest(path,*args,**kwargs)
        with patch.object(c,'save_checkpoint',side_effect=fail),self.assertRaises(OSError): self.fit()
        self.assertEqual(c.read_checkpoint(self.directory()/'train-last.pth')['epoch'],1)
        resumed=self.resume(); self.assertTrue(boundary.same(full['state_dict'],resumed['state_dict']))

    def test_t14_duplicate_stale_missing_and_corrupt_history_refused_without_journal(self):
        self.interrupted(); path=self.directory()/'train.history.json'; original=path.read_bytes()
        for document in ({'epochs':[]},{'epochs':[{'epoch':1},{'epoch':1}]},None,'corrupt'):
            path.write_bytes(original)
            if document is None: path.unlink()
            elif document=='corrupt': path.write_text('{')
            else: e.write_json(path,document)
            with self.subTest(document=document),self.assertRaises(SystemExit): self.resume()

    def test_t12_default_pruning_development_never_constructs_test(self):
        from breakhis import data
        self.config['pruning']=dict(method='unstructured',sparsity=.5,finetune_epochs=1); self.save()
        for row in self.rows:
            if row['split']=='test': (self.images/row['path']).unlink()
        original=data.get_split_loaders; requested=[]
        def loaders(*args,**kwargs):
            roles=kwargs['splits']; self.assertNotIn('test',roles); requested.append(roles)
            return original(*args,**kwargs)
        with patch.object(data,'get_split_loaders',side_effect=loaders): run=self.run_dev()
        state=e.read_json(run/'run.json')
        self.assertEqual(state['status'],'development_completed')
        self.assertEqual(state['phases']['final']['status'],'not_requested')
        self.assertFalse((run/'final').exists()); self.assertIn(('validation',),requested)
        selection=e.read_json(run/'development/selection.json')
        self.assertEqual(set(selection['roles']),{'dense','pruned','finetuned'})
        self.assertEqual(selection['roles']['pruned']['criterion'],'fixed_recipe')
        self.assertEqual(selection['roles']['pruned']['epoch'],0)
        for role in selection['roles']:
            self.assertEqual(e.read_json(run/f'development/{role}.metrics.json')['split'],'validation')
        self.assertIsNone(e.read_json(run/'results.json')['phases']['final'])

    def test_t13_child_nonzero_and_ctrl_c_leave_honest_states(self):
        for identifier,effect in [('nonzero',subprocess.CompletedProcess([],7)),('interrupt',KeyboardInterrupt())]:
            def child(argv,**kwargs):
                if '-m' not in argv: return self.real_run(argv,**kwargs)
                if isinstance(effect,BaseException): raise effect
                return effect
            with patch('breakhis.development.subprocess.run',side_effect=child), \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaises((RuntimeError,KeyboardInterrupt)):
                e.run_experiment(self.file,run_id=identifier)
            state=e.read_json(self.root/f'runs/{identifier}/run.json')
            expected='interrupted' if identifier=='interrupt' else 'failed'
            self.assertEqual(state['phases']['development']['status'],expected)
            self.assertEqual(state['stages']['train']['status'],expected)
            self.assertTrue(Path(state['stages']['train']['log']).exists())

    def test_t13_interrupted_child_fit_resumes_completed_boundary(self):
        original=training.train
        def stop(*args,**kwargs):
            value=original(*args,**kwargs)
            if kwargs['epoch']==2: raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=stop),self.assertRaises(InterruptedError): self.run_dev()
        state=e.read_json(self.root/'runs/dev/run.json'); self.assertEqual(state['status'],'interrupted')
        self.assertEqual(c.read_checkpoint(self.directory('dev')/'train-last.pth')['epoch'],1)
        run=self.run_dev(resume=True); self.assertEqual(e.read_json(run/'run.json')['status'],'development_completed')

    def test_t14_completed_fit_recovery_and_missing_secondary_artifact_refusal(self):
        original=boundary.reconcile
        # Fail after final boundary committed, before stage success is recorded.
        def fail(output,payload):
            value=original(output,payload)
            if payload['epoch']==4: raise OSError('stage publication interruption')
            return value
        with patch.object(boundary,'reconcile',side_effect=fail),self.assertRaises(OSError): self.run_dev()
        run=self.run_dev(resume=True)
        state=e.read_json(run/'run.json'); self.assertTrue(state['stages']['train']['recovered_completed_boundary'])
        # Simulate an interrupted report after training was successfully committed.
        state['status']='failed'; state['phases']['development']['status']='failed'
        e.write_json(run/'run.json',state)
        (self.directory('dev')/'train.config.json').unlink()
        with self.assertRaisesRegex(ValueError,'Artifact missing'): self.run_dev(resume=True)

    def test_t14_missing_prediction_blocks_skip_and_success(self):
        run=self.run_dev(); state=e.read_json(run/'run.json')
        state['status']='failed'; state['phases']['development']['status']='failed'
        e.write_json(run/'run.json',state); (run/'development/dense.predictions.json').unlink()
        with self.assertRaisesRegex(ValueError,'Artifact missing'): self.run_dev(resume=True)
        self.assertEqual(e.read_json(run/'run.json')['phases']['development']['status'],'failed')

    def test_t11_completed_boundary_recovery_rejects_runtime_drift(self):
        original=boundary.reconcile
        def fail(output,payload):
            value=original(output,payload)
            if payload['epoch']==4: raise OSError('interrupted stage publication')
            return value
        with patch.object(boundary,'reconcile',side_effect=fail),self.assertRaises(OSError): self.run_dev()
        latest=self.directory('dev')/'train-last.pth'; payload=c.read_checkpoint(latest)
        payload['runtime']['python']='different'; c.save_payload(latest,payload)
        with self.assertRaisesRegex(ValueError,'identity/config/runtime'): self.run_dev(resume=True)

    def test_t10_legacy_accuracy_keeps_earlier_equal_epoch(self):
        original=training.evaluate; calls={'n':0}
        def evaluate(*args,**kwargs):
            original(*args,**kwargs); calls['n']+=1
            return 50., 1./calls['n']
        with patch.object(training,'evaluate',side_effect=evaluate):
            result=self.fit(selection_metric='accuracy')
        self.assertEqual(result['training_state']['best']['epoch'],1)

    def test_t13_results_publication_failure_can_retry_completed_stages(self):
        original=e.write_json
        def fail(path,*args,**kwargs):
            if Path(path).as_posix().endswith('development/results.json'): raise OSError('disk failure')
            return original(path,*args,**kwargs)
        with patch.object(e,'write_json',side_effect=fail),self.assertRaises(OSError): self.run_dev()
        state=e.read_json(self.root/'runs/dev/run.json'); self.assertEqual(state['status'],'failed')
        self.assertIsNone(e.read_json(self.root/'runs/dev/results.json')['phases']['development'])
        run=self.run_dev(resume=True); self.assertEqual(e.read_json(run/'run.json')['status'],'development_completed')


if __name__=='__main__': unittest.main()
