"""M1-T15--T21: fixed final inference, complete predictions and saved comparisons."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from breakhis import experiments as e, checkpoints as c, cli, reporting as r
from breakhis import phase_artifacts as a, experiment_setup as setup
from breakhis.final_evaluation import run_final
import test_m1_execution as fixtures


class PhaseTests(unittest.TestCase):
    save=fixtures.ExecutionTests.save
    model=staticmethod(fixtures.ExecutionTests.model)
    child=fixtures.ExecutionTests.child
    run_dev=fixtures.ExecutionTests.run_dev
    directory=fixtures.ExecutionTests.directory

    def setUp(self):
        fixtures.ExecutionTests.setUp(self)
        self.config['training']['epochs']=1; self.save()

    def final(self,identifier='dev',resume=False):
        with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return run_final(run_id=identifier,resume=resume)

    def refresh(self,run,path):
        """Simulate a logically inconsistent document whose publication hash is current."""
        selection_path=run/'development/selection.json'
        selection=e.read_json(selection_path)
        changed=False
        for binding in selection['roles'].values():
            for key in ('report','predictions'):
                ref=binding[key]
                if run/ref['path']==path:
                    binding[key]=setup.artifact_reference(path,root=run,kind='run'); changed=True
        if changed: e.write_json(selection_path,selection)
        state=e.read_json(run/'run.json'); inputs=e.read_json(run/'inputs.json')
        for record in state['stages'].values():
            for i,ref in enumerate(record.get('artifacts',[])):
                root=run if ref['root']=='run' else Path(inputs['checkpoint_directory'])
                if root/ref['path']==path or changed and root/ref['path']==selection_path:
                    record['artifacts'][i]=setup.artifact_reference(root/ref['path'],root=root,kind=ref['root'])
        e.write_json(run/'run.json',state)

    def test_t15_missing_development_refuses_before_loaders(self):
        setup.prepare_run(self.file,run_id='prepared')
        with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('test access')):
            with self.assertRaisesRegex(ValueError,'completed development'): self.final('prepared')

    def test_t15_changed_selection_checkpoint_config_and_manifest_refuse(self):
        run=self.run_dev(); inputs=e.read_json(run/'inputs.json')
        for path in (run/'development/selection.json',run/'config.json',run/'manifest.json',self.directory('dev')/'train.pth'):
            before=path.read_bytes(); path.write_bytes(before+b' ')
            with self.subTest(path=path),patch('breakhis.data.get_split_loaders',side_effect=AssertionError('test access')):
                with self.assertRaises((ValueError,OSError)): self.final()
            path.write_bytes(before)
        path=run/'development/selection.json'; value=e.read_json(path); value['threshold']='tuned'
        e.write_json(path,value); self.refresh(run,path)
        with self.assertRaisesRegex(ValueError,'policy'): self.final()

    def test_t16_cpu_final_of_cuda_origin_needs_no_fitting_gpu_or_training_files(self):
        # CUDA-origin declaration on portable saved weights; no CUDA operations performed.
        # A separate conditional device test exercises actual CUDA when available.
        self.config['training']['device']='cuda'; self.save()
        with patch.object(torch.cuda,'is_available',return_value=True):
            prepared=setup.prepare_run(self.file,run_id='origin')
        # Complete a CPU fixture with the same portable model, then mark its original
        # fitting device as CUDA in frozen identities (simulating a copied CUDA run).
        self.config['training']['device']='cpu'; self.save()
        run=self.run_dev(); config=e.read_json(run/'config.json'); config['training']['device']='cuda'
        e.write_json(run/'config.json',config)
        inputs=e.read_json(run/'inputs.json'); inputs['identity']['resolved_device']='cuda'
        inputs['identity']['config_sha256']=e.digest(run/'config.json'); e.write_json(run/'inputs.json',inputs)
        selection=e.read_json(run/'development/selection.json'); selection['config_sha256']=inputs['identity']['config_sha256']
        e.write_json(run/'development/selection.json',selection); self.refresh(run,run/'development/selection.json')
        for suffix in ('metrics','predictions'):
            path=run/f'development/dense.{suffix}.json'; value=e.read_json(path)
            value['config_sha256']=inputs['identity']['config_sha256']; e.write_json(path,value); self.refresh(run,path)
        for row in self.rows:
            if row['split']!='test': (self.images/row['path']).unlink()
        from breakhis import data
        original=data.get_split_loaders
        def loaders(*args,**kwargs):
            self.assertEqual(kwargs['splits'],('test',)); return original(*args,**kwargs)
        with patch.object(torch.cuda,'is_available',return_value=False), \
             patch('breakhis.training.train',side_effect=AssertionError('fitting')), \
             patch.object(data,'get_split_loaders',side_effect=loaders):
            self.final()
        state=e.read_json(run/'run.json'); self.assertEqual(state['status'],'final_completed')
        self.assertEqual(e.read_json(run/'final/dense.metrics.json')['evaluation_runtime']['device'],'cpu')

    def test_t16_final_uses_bound_all_roles_and_designated_recipe(self):
        self.config['pruning']=dict(method='unstructured',sparsity=.5,finetune_epochs=1); self.save()
        run=self.run_dev(); selection=(run/'development/selection.json').read_bytes()
        with patch('breakhis.training.train',side_effect=AssertionError('fitting')), \
             patch('breakhis.pruning.prune_weights',side_effect=AssertionError('pruning')), \
             patch.object(torch.optim,'Adam',side_effect=AssertionError('optimizer')):
            self.final()
        result=e.read_json(run/'final/results.json')
        self.assertEqual(result['designated_role'],'finetuned'); self.assertEqual(len(result['reports']),3)
        self.assertEqual(selection,(run/'development/selection.json').read_bytes())
        for role in result['reports']: a.validate_report(run,self.config,e.read_json(run/'inputs.json'),e.read_json(run/'development/selection.json'),role,'final')
        before=(run/'final/results.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already completed'): self.final()
        self.assertEqual(before,(run/'final/results.json').read_bytes())

    def test_t16_failed_final_preserves_development_and_retries_only_incomplete_roles(self):
        self.config['pruning']=dict(method='unstructured',sparsity=.5,finetune_epochs=1); self.save()
        run=self.run_dev(); before={p.name:p.read_bytes() for p in (run/'development').iterdir()}
        original=self.child; calls=[]
        def fail(argv,**kwargs):
            if '-m' in argv:
                role=argv[argv.index('--model-role')+1]; calls.append(role)
                if role=='pruned': return subprocess.CompletedProcess(argv,9)
            return original(argv,**kwargs)
        with patch('breakhis.final_evaluation.subprocess.run',side_effect=fail),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError): run_final(run_id='dev')
        self.assertEqual(e.read_json(run/'run.json')['phases']['final']['status'],'failed')
        self.assertEqual(before,{p.name:p.read_bytes() for p in (run/'development').iterdir()})
        dense=(run/'final/dense.metrics.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'--resume'): self.final()
        self.final(resume=True); self.assertEqual(dense,(run/'final/dense.metrics.json').read_bytes())

    def test_t16_final_results_publication_failure_is_retryable(self):
        run=self.run_dev(); original=e.write_json
        def fail(path,*args,**kwargs):
            if Path(path)==run/'final/results.json': raise OSError('publication failed')
            return original(path,*args,**kwargs)
        with patch.object(e,'write_json',side_effect=fail),self.assertRaises(OSError): self.final()
        self.assertIsNone(e.read_json(run/'results.json')['phases']['final'])
        with patch('breakhis.checkpoints.load_checkpoint',side_effect=AssertionError('completed stage repeated')): self.final(resume=True)

    def test_t17_hand_calculated_unequal_magnifications_and_tie(self):
        rows=[dict(patient_id='b',group_id='component-b',subtype='adenosis',magnification='40X')]*3+[
            dict(patient_id='b',group_id='component-b',subtype='adenosis',magnification='100X'),
            dict(patient_id='m',group_id='component-m',subtype='ductal_carcinoma',magnification='40X')]
        scores=[[.8,.2]]*3+[[.2,.8],[.3,.7]]; targets=[0]*4+[1]
        primary=r.magnification_balanced_metrics(rows,targets,scores,r.CLASSES)
        self.assertEqual(primary['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(primary['balanced_accuracy'],1)
        self.assertAlmostEqual(primary['groups'][0]['probabilities'][0],.5)
        self.assertAlmostEqual(primary['groups'][0]['probabilities'][1],.5)
        self.assertEqual(primary['groups'][0]['prediction'],0)
        self.assertEqual(primary['groups'][0]['component_id'],'component-b')
        self.assertEqual(primary['groups'][0]['images'],4)
        secondary=r.grouped_patient_metrics(rows,targets,scores,r.CLASSES)
        self.assertAlmostEqual(secondary['groups'][0]['probabilities'][1],.35)
        absent=r.classification_metrics([0],[0],r.CLASSES)
        self.assertIsNone(absent['sensitivity']); self.assertIsNone(absent['balanced_accuracy'])
        self.assertIn('recall',absent['per_class']['malignant']['undefined'])
        self.assertEqual(absent['accuracy_unit'],'percent'); self.assertEqual(absent['rate_unit'],'fraction')

    def test_t18_complete_ordered_predictions_and_subtype_envelopes(self):
        run=self.run_dev(); self.final(); prediction=e.read_json(run/'final/dense.predictions.json')
        expected=[row for row in self.rows if row['split']=='test']
        self.assertEqual([v['path'] for v in prediction['rows']],[v['path'] for v in expected])
        self.assertTrue(all('subtype' in row and 'pixel_sha256' in row for row in prediction['rows']))
        self.assertEqual(prediction['config_sha256'],e.digest(run/'config.json'))
        self.assertEqual(prediction['classes'],r.CLASSES); self.assertEqual(prediction['phase'],'final')
        self.assertEqual(prediction['rows'][0]['group_id'],prediction['groups'][0]['component_id'])

    def test_t18_shuffle_drop_custom_sampler_duplicates_and_missing_rows_refuse(self):
        dataset=TensorDataset(torch.ones(3,2),torch.tensor([0,1,0])); dataset.classes=r.CLASSES
        dataset.rows=[dict(patient_id=str(i),path=str(i)) for i in range(3)]
        model=torch.nn.Identity()
        for loader in (DataLoader(dataset,batch_size=2,shuffle=True),DataLoader(dataset,batch_size=2,drop_last=True)):
            with self.assertRaises(ValueError): r.collect_predictions(model,loader)
        class Duplicate(SequentialSampler):
            def __iter__(self): return iter([0,0,2])
        with self.assertRaises(ValueError): r.collect_predictions(model,DataLoader(dataset,sampler=Duplicate(dataset)))
        dataset.rows[1]['path']='0'
        with self.assertRaisesRegex(ValueError,'Duplicate'): r.collect_predictions(model,DataLoader(dataset))
        dataset.rows.pop()
        with self.assertRaisesRegex(ValueError,'coverage'): r.collect_predictions(model,DataLoader(dataset))

    def test_t18_config_alias_protected_before_loading(self):
        run=self.run_dev(); checkpoint=self.directory('dev')/'train.pth'
        before=(run/'config.json').read_bytes()
        with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('loaded pixels')), \
             contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            cli.main(['evaluate','--dataset','breakhis','--checkpoint',str(checkpoint),
                '--manifest',str(run/'manifest.json'),'--experiment-config',str(run/'config.json'),
                '--output',str(run/'config.json')])
        self.assertEqual(before,(run/'config.json').read_bytes())

    def pair(self,final=False):
        first=self.run_dev('a'); second=self.run_dev('b')
        if final: self.final('a'); self.final('b')
        return first,second

    def test_t19_comparison_reads_only_selected_phase_without_model_loading(self):
        pair=self.pair(True)
        with patch.object(c,'load_checkpoint',side_effect=AssertionError('model load')), \
             patch.object(torch,'load',side_effect=AssertionError('weights loaded')):
            dev=e.compare_runs(pair,self.root/'dev.json')
            final=e.compare_runs(pair,self.root/'final.json',phase='final')
        self.assertEqual(e.read_json(dev)['rows'][0]['split'],'validation')
        self.assertEqual(e.read_json(final)['rows'][0]['split'],'test')
        row=e.read_json(final)['rows'][0]
        self.assertEqual(row['completed_dense_epochs'],1); self.assertEqual(row['dense_optimizer_steps'],1)
        self.assertTrue((self.root/'dev.csv').exists())

    def test_t19_incomplete_final_never_falls_back_to_development(self):
        pair=self.pair()
        with self.assertRaisesRegex(ValueError,'two completed requested'): e.compare_runs(pair,self.root/'bad.json',phase='final')
        self.assertFalse((self.root/'bad.json').exists())

    def test_t19_missing_corrupt_or_mixed_phase_outputs_refuse(self):
        pair=self.pair(); path=pair[1]/'development/dense.predictions.json'; before=path.read_bytes()
        for value in (None,b'{',before+b' '):
            if value is None: path.unlink()
            else: path.write_bytes(value)
            with self.assertRaises((ValueError,OSError)): e.compare_runs(pair,self.root/'bad.json')
            path.write_bytes(before)
        report=e.read_json(path); report['phase']='final'; e.write_json(path,report); self.refresh(pair[1],path)
        with self.assertRaisesRegex(ValueError,'phase or identity'): e.compare_runs(pair,self.root/'bad.json')

    def test_t19_failed_final_keeps_development_comparable(self):
        pair=self.pair(); state=e.read_json(pair[0]/'run.json'); state['status']='failed'
        state['phases']['final']['status']='failed'; e.write_json(pair[0]/'run.json',state)
        self.assertTrue(e.compare_runs(pair,self.root/'dev.json').exists())

    def test_t20_selection_difference_requires_explicit_declaration(self):
        first=self.run_dev('a'); self.config['training']['selection_metric']='accuracy'; self.save(); second=self.run_dev('b')
        with self.assertRaisesRegex(ValueError,'--vary'): e.compare_runs([first,second],self.root/'bad.json')
        output=e.compare_runs([first,second],self.root/'varied.json',vary=['training.selection_metric'])
        self.assertIn('training.selection_metric',e.read_json(output)['varied_factors'])
        with self.assertRaisesRegex(ValueError,'invariants cannot vary'): e.compare_runs([first,second],self.root/'bad.json',vary=['data.manifest'])

    def test_t20_supported_seed_epoch_pruning_factors_and_budgets_listed(self):
        first=self.run_dev('a'); self.config['training'].update(seed=23,epochs=2)
        self.config['pruning']=dict(method='unstructured',sparsity=.5,finetune_epochs=1); self.save(); second=self.run_dev('b')
        output=e.read_json(e.compare_runs([first,second],self.root/'varied.json'))
        self.assertEqual(set(output['varied_factors']),{'training.seed','training.epochs','pruning'})
        fine=next(v for v in output['rows'] if v['role']=='finetuned')
        self.assertEqual(fine['completed_dense_epochs'],2); self.assertEqual(fine['completed_finetune_epochs'],1)
        self.assertEqual(fine['finetune_optimizer_steps'],1)

    def test_t20_benchmark_and_split_invariants_cannot_be_overridden(self):
        first=self.run_dev('a'); self.config['evaluation']['repeats']=2; self.save(); second=self.run_dev('b')
        with self.assertRaisesRegex(ValueError,'invariants'): e.compare_runs([first,second],self.root/'bad.json',vary=['training.selection_metric'])

    def test_t18_reordered_or_truncated_saved_predictions_fail_semantic_validation(self):
        run=self.run_dev(); path=run/'development/dense.predictions.json'; original=e.read_json(path)
        for rows in (list(reversed(original['rows'])),original['rows'][:-1]):
            value=copy.deepcopy(original); value['rows']=rows; e.write_json(path,value); self.refresh(run,path)
            with self.assertRaisesRegex(ValueError,'Prediction'): a.development(run)

    def test_t19_split_changes_and_legacy_runs_refuse(self):
        first=self.run_dev('a')
        manifest=self.root/'manifest.json'; manifest.write_bytes(manifest.read_bytes()+b' ')
        second=self.run_dev('b')
        with self.assertRaisesRegex(ValueError,'invariants'): e.compare_runs([first,second],self.root/'bad.json')
        state=e.read_json(second/'run.json'); state['schema_version']=1; e.write_json(second/'run.json',state)
        with self.assertRaisesRegex(ValueError,'Schema-1'): e.compare_runs([first,second],self.root/'bad.json')

    def test_t19_failed_run_diagnostics_and_atomic_comparison_publication(self):
        pair=self.pair(); prepared=setup.prepare_run(self.file,run_id='incomplete')
        output=e.compare_runs([*pair,prepared['run']],self.root/'diagnostics.json')
        result=e.read_json(output)
        self.assertEqual(len(result['diagnostics']),1); self.assertEqual(len(result['rows']),2)
        import breakhis.saved_comparison as comparison
        original=comparison.publish
        def fail(path,*args,**kwargs):
            if Path(path).suffix=='.json': raise OSError('comparison publication failure')
            return original(path,*args,**kwargs)
        with patch.object(comparison,'publish',side_effect=fail),self.assertRaises(OSError):
            e.compare_runs(pair,self.root/'failed.json')
        self.assertFalse((self.root/'failed.json').exists()); self.assertFalse((self.root/'failed.csv').exists())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available in CPU test environment')
    def test_t16_actual_cuda_fit_then_portable_cpu_final(self):
        self.config['training']['device']='cuda'; self.save(); run=self.run_dev()
        with patch.object(torch.cuda,'is_available',return_value=False): self.final()
        self.assertEqual(e.read_json(run/'final/dense.metrics.json')['evaluation_runtime']['device'],'cpu')

    def test_t21_seeded_resident_input_preserves_rng_and_is_image_independent(self):
        model=torch.nn.Linear(2,2).eval(); state=torch.get_rng_state().clone()
        one=r.benchmark_model(model,torch.ones(1,2),warmup=0,repeats=2)
        two=r.benchmark_model(model,torch.zeros(1,2),warmup=0,repeats=2)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(one['input_sha256'],two['input_sha256'])
        self.assertEqual(one['protocol'],'resident-forward-synthetic-v1')
        self.assertEqual(len(one['samples_ms']),2); self.assertTrue(all(v>0 for v in one['samples_ms']))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available in CPU test environment')
    def test_t21_cuda_timing_is_synchronized(self):
        model=torch.nn.Linear(2,2).cuda().eval()
        with patch.object(torch.cuda,'synchronize',wraps=torch.cuda.synchronize) as sync:
            report=r.benchmark_model(model,torch.ones(1,2),warmup=1,repeats=2)
        self.assertGreaterEqual(sync.call_count,5)
        self.assertEqual(report['runtime']['device'],'cuda:0')


if __name__=='__main__': unittest.main()
