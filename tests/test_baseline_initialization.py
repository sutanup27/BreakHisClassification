"""M2-T05--T09: initialization, immutable parents, compact and exact child resume."""
import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from breakhis import experiments as e, checkpoints as c, experiment_setup as s, cli, baseline_init as b
from breakhis import phase_artifacts as a, training, fit_boundary as boundary
from breakhis.data import set_seed
from breakhis.models.vgg import VGG
import test_m1_execution as fixtures

ACTUAL_CREATE=c.create_model


def baseline(mode='standard',**values):
    return dict(kind=mode,parent_selection=None,channels=None,topology_checkpoint=None,
        ranking=None,ranking_seed=None) | values


class InitializationTests(unittest.TestCase):
    save=fixtures.ExecutionTests.save
    model=staticmethod(fixtures.ExecutionTests.model)
    child=fixtures.ExecutionTests.child
    run_dev=fixtures.ExecutionTests.run_dev
    directory=fixtures.ExecutionTests.directory

    def setUp(self):
        fixtures.ExecutionTests.setUp(self)
        self.config['training']['epochs']=1; self.save()

    def v2(self,mode='standard',**values):
        self.config.update(schema_version=2,baseline=baseline(mode,**values)); self.save()

    def parent(self):
        run=self.run_dev('parent')
        return run,run/'development/selection.json'

    def snapshot(self,run):
        root=self.directory(run.name)
        return {str(p):e.digest(p) for anchor in (run,root) for p in anchor.rglob('*') if p.is_file()}

    def test_t05_config1_unchanged_and_all_config2_modes_validate(self):
        old=e.load_config(self.file); self.assertEqual(old['schema_version'],1)
        for mode in ('standard','prune_from_parent','dense_continue','narrow_scratch'):
            self.config['pruning']=dict(method='channel',sparsity=.5,finetune_epochs=1) if mode=='prune_from_parent' else None
            fields={} if mode=='standard' else dict(parent_selection='parent/development/selection.json')
            if mode=='prune_from_parent': fields.update(ranking='random',ranking_seed=202)
            if mode=='narrow_scratch': fields.update(channels=[1]*8,topology_checkpoint='topology.pth')
            self.v2(mode,**fields); value=e.load_config(self.file)
            self.assertEqual(value['baseline']['kind'],mode)
            if fields: self.assertTrue(Path(value['baseline']['parent_selection']).is_absolute())
        self.config.pop('baseline'); self.config['schema_version']=1; self.config['pruning']=None
        self.config['training']['model']='mobilenet_v3_small'; self.save()
        with self.assertRaisesRegex(ValueError,'model'): e.load_config(self.file)

    def test_t05_contradictions_refuse_before_model_or_output(self):
        cases=[baseline('bad'),baseline('standard',channels=[1]*8),baseline('dense_continue'),
            baseline('narrow_scratch',parent_selection='p',channels=[True]*8,topology_checkpoint='t'),
            baseline('dense_continue',parent_selection='p',ranking='l1')]
        for value in cases:
            self.config.update(schema_version=2,baseline=value); self.save()
            with self.subTest(value=value),self.assertRaises(ValueError): s.prepare_run(self.file,run_id='bad')
        self.assertFalse((self.root/'runs').exists())

    def test_t06_parent_acquisition_copies_no_weights_or_histories(self):
        run,selection=self.parent(); before=self.snapshot(run)
        self.v2('dense_continue',parent_selection=str(selection))
        prepared=s.prepare_run(self.file,run_id='child')
        record=prepared['inputs']['initialization']; b.verify_record(record)
        self.assertEqual(record['parent']['run_id'],'parent')
        self.assertEqual(record['parent']['executed_epochs'],1)
        expected=prepared['inputs']['expected_artifacts']
        self.assertIn('parent',expected); self.assertNotIn('train',expected)
        self.assertIn('finetune',expected)
        self.assertFalse(list(self.directory('child').glob('*.pth')))
        self.assertEqual(before,self.snapshot(run))

    def test_t06_seed_recipe_and_manifest_mismatch_refused_before_child_creation(self):
        run,selection=self.parent(); old=copy.deepcopy(self.config)
        for field,value in [('seed',99),('learning_rate',.03),('batch_size',3),('selection_metric','accuracy')]:
            self.config=copy.deepcopy(old); self.config['training'][field]=value
            self.v2('dense_continue',parent_selection=str(selection))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'recipe/seed'): s.prepare_run(self.file,run_id='bad')
        self.config=old; self.v2('dense_continue',parent_selection=str(selection))
        self.config['data']['magnification']='all'; self.save()
        with self.assertRaisesRegex(ValueError,'magnification'): s.prepare_run(self.file,run_id='bad')
        self.assertFalse((self.root/'runs/bad').exists())

    def test_t06_changed_parent_and_changed_initialization_refuse(self):
        run,selection=self.parent(); self.v2('dense_continue',parent_selection=str(selection))
        child=s.prepare_run(self.file,run_id='child')
        checkpoint=self.directory('parent')/'train.pth'; old=checkpoint.read_bytes(); checkpoint.write_bytes(old+b' ')
        with self.assertRaisesRegex(ValueError,'changed'): s.prepare_run(self.file,run_id='child',resume=True)
        checkpoint.write_bytes(old)
        init=child['run']/'initialization.json'; init.write_bytes(init.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'initialization'): s.prepare_run(self.file,run_id='child',resume=True)

    def test_t06_new_fit_can_use_older_origin_but_exact_resume_remains_source_bound(self):
        run,selection=self.parent(); origin=e.read_json(run/'inputs.json')['code']
        self.v2('dense_continue',parent_selection=str(selection))
        changed=e.revision(); changed['source_sha256']={'new-code':'f'*64}
        with patch.object(e,'revision',return_value=changed):
            with self.assertRaisesRegex(ValueError,'Implementation'): a.development(run)
            child=s.prepare_run(self.file,run_id='child')
        self.assertEqual(child['inputs']['initialization']['parent']['code'],origin)
        self.assertEqual(child['inputs']['code'],changed)
        with self.assertRaisesRegex(ValueError,'Implementation'): s.prepare_run(self.file,run_id='child',resume=True)

    def test_t06_undeclared_roots_and_reference_escape_refuse(self):
        run,selection=self.parent(); self.v2('dense_continue',parent_selection=str(selection))
        record=s.prepare_run(self.file,run_id='child')['inputs']['initialization']
        for change in ('root','escape'):
            value=copy.deepcopy(record)
            if change=='root': value['roots']['arbitrary']=str(self.root)
            else: value['parent']['checkpoint']['path']='../outside.pth'
            with self.subTest(change=change),self.assertRaises(ValueError): b.verify_record(value)

    def test_t07_real_compact_forward_backward_reload_without_download(self):
        with patch.object(c,'create_model',side_effect=ACTUAL_CREATE), \
             patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('download')):
            model=c.create_model('mobilenet_v3_small'); inputs=torch.randn(2,3,64,64)
            model.train(); logits=model(inputs); torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            model.eval(); expected=model(inputs).detach(); path=self.root/'compact.pth'
            c.save_checkpoint(path,model,'mobilenet_v3_small',{})
            loaded,payload=c.load_checkpoint(path)
            loaded.eval(); self.assertTrue(torch.allclose(expected,loaded(inputs),atol=1e-6,rtol=1e-5))
            self.assertEqual(payload['classes'],['benign','malignant'])
            self.assertEqual(loaded.classifier[-1].out_features,2)
            with self.assertRaises(ValueError): c.create_model('mobilenet_v3_small',pretrained=True)

    def test_t07_compact_high_level_standard_development_and_final(self):
        self.config['training']['model']='mobilenet_v3_small'; self.v2()
        with patch.object(c,'create_model',side_effect=ACTUAL_CREATE):
            run=self.run_dev('compact')
            a.development(run)
            from breakhis.final_evaluation import run_final
            with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): run_final(run_id='compact')
        self.assertEqual(e.read_json(run/'run.json')['phases']['final']['status'],'completed')
        self.assertEqual(e.read_json(run/'final/dense.metrics.json')['baseline_kind'],'standard')

    def topology(self,selection):
        parent=self.directory('parent')/'train.pth'; output=self.root/'checkpoint/binary/topology/prune.pth'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            cli.main(['prune','--dataset','breakhis','--manifest',str(self.root/'manifest.json'),
                '--checkpoint',str(parent),'--method','channel','--sparsity','.5','--device','cpu','--output',str(output)])
        return output

    def tiny_vgg(self,*args,**kwargs): return VGG(classes=2,channels=kwargs.get('channels') or [2]*8)

    def test_t08_scratch_has_matching_widths_and_fresh_numerical_bn_state(self):
        with patch.object(c,'create_model',side_effect=self.tiny_vgg):
            run,selection=self.parent(); topology=self.topology(selection)
            self.v2('narrow_scratch',parent_selection=str(selection),channels=[1]*8,topology_checkpoint=str(topology))
            prepared=s.prepare_run(self.file,run_id='scratch'); record=prepared['inputs']['initialization']
            set_seed(22); fresh=b.fresh_model(record,2,'cpu')
            set_seed(22); same=b.fresh_model(record,2,'cpu')
            self.assertTrue(boundary.same(fresh.state_dict(),same.state_dict()))
            set_seed(23); other=b.fresh_model(record,2,'cpu')
            self.assertFalse(boundary.same(fresh.state_dict(),other.state_dict()))
            inherited=c.read_checkpoint(topology)['state_dict']
            self.assertEqual(fresh.channels,[1]*8); self.assertFalse(hasattr(fresh,'channel_pruning'))
            self.assertFalse(torch.equal(fresh.backbone.conv0.weight,inherited['backbone.conv0.weight']))
            self.assertEqual(fresh.backbone.bn0.num_batches_tracked.item(),0)
            self.assertTrue(torch.all(fresh.backbone.bn0.running_mean==0))
            self.assertTrue(torch.all(fresh.backbone.bn0.running_var==1))
            self.assertTrue(torch.all(fresh.backbone.bn0.weight==1))

    def test_t08_wrong_topology_or_unbound_legacy_topology_refused(self):
        with patch.object(c,'create_model',side_effect=self.tiny_vgg):
            run,selection=self.parent(); topology=self.topology(selection)
            self.v2('narrow_scratch',parent_selection=str(selection),channels=[2]*8,topology_checkpoint=str(topology))
            with self.assertRaisesRegex(ValueError,'Topology'): s.prepare_run(self.file,run_id='bad')
            self.config['baseline']['channels']=[1]*8; self.save()
            payload=c.read_checkpoint(topology); payload['config'].pop('parent_checkpoint_sha256')
            c.save_payload(topology,payload)
            with self.assertRaisesRegex(ValueError,'regenerate'): s.prepare_run(self.file,run_id='bad')

    def test_t08_actual_scratch_fit_uses_train_not_finetune_and_preserves_parent(self):
        with patch.object(c,'create_model',side_effect=self.tiny_vgg):
            run,selection=self.parent(); topology=self.topology(selection); before=self.snapshot(run)
            self.v2('narrow_scratch',parent_selection=str(selection),channels=[1]*8,topology_checkpoint=str(topology))
            child=self.run_dev('scratch'); a.development(child)
            payload=c.read_checkpoint(self.directory('scratch')/'train-last.pth')
        self.assertEqual(payload['architecture_config']['channels'],[1]*8)
        self.assertEqual(payload['pruning']['type'],'dense')
        self.assertEqual(e.read_json(child/'parent.json')['child_parent_fit_steps'],0)
        self.assertFalse((child/'finetune.history.json').exists()); self.assertEqual(before,self.snapshot(run))

    def test_t09_continuation_starts_fresh_optimizer_and_has_no_synthetic_parent_history(self):
        run,selection=self.parent(); before=self.snapshot(run)
        self.v2('dense_continue',parent_selection=str(selection))
        original=training.train; optimizer_sizes=[]
        def observe(*args,**kwargs):
            optimizer_sizes.append(len(args[3].state)); return original(*args,**kwargs)
        with patch.object(training,'train',side_effect=observe): child=self.run_dev('continue')
        a.development(child)
        self.assertEqual(optimizer_sizes,[0])
        self.assertFalse((child/'train.history.json').exists()); self.assertFalse((self.directory('continue')/'train-last.pth').exists())
        self.assertEqual(e.read_json(child/'run.json')['stages']['parent']['optimizer_steps'],0)
        self.assertEqual(before,self.snapshot(run))
        selected=e.read_json(child/'development/selection.json')
        self.assertEqual(selected['designated_role'],'finetuned')
        self.assertEqual(selected['roles']['dense']['checkpoint']['root'],'parent_checkpoint')

    def test_t09_continuation_resume_matches_uninterrupted_fit(self):
        run,selection=self.parent(); self.config['training']['epochs']=3
        self.v2('dense_continue',parent_selection=str(selection))
        full=self.run_dev('full'); original=training.train
        def stop(*args,**kwargs):
            value=original(*args,**kwargs)
            if kwargs['epoch']==2: raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=stop),self.assertRaises(InterruptedError): self.run_dev('resumed')
        child=self.run_dev('resumed',resume=True)
        left=c.read_checkpoint(self.directory('full')/'finetune-last.pth')
        right=c.read_checkpoint(self.directory('resumed')/'finetune-last.pth')
        self.assertTrue(boundary.same(left['state_dict'],right['state_dict']))
        self.assertTrue(boundary.same(fixtures.ExecutionTests.numerical(left['training_state']),fixtures.ExecutionTests.numerical(right['training_state'])))
        self.assertEqual([r['epoch'] for r in e.read_json(child/'finetune.history.json')['epochs']],[1,2,3])

    def test_t09_epoch_zero_and_final_continuation_do_not_resume_parent(self):
        run,selection=self.parent(); self.v2('dense_continue',parent_selection=str(selection))
        original=training.train
        def stop(*args,**kwargs): original(*args,**kwargs); raise KeyboardInterrupt
        with patch.object(training,'train',side_effect=stop),self.assertRaises(InterruptedError): self.run_dev('continue')
        self.assertEqual(c.read_checkpoint(self.directory('continue')/'finetune-last.pth')['epoch'],0)
        child=self.run_dev('continue',resume=True)
        from breakhis.final_evaluation import run_final
        with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
             patch.object(training,'train',side_effect=AssertionError('fitting')), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): run_final(run_id='continue')
        self.assertEqual(e.read_json(child/'run.json')['phases']['final']['status'],'completed')

    def test_low_level_initialization_drift_refuses_before_loaders(self):
        run,selection=self.parent(); self.v2('dense_continue',parent_selection=str(selection))
        prepared=s.prepare_run(self.file,run_id='child')
        with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('pixels')), \
             contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            cli.main(['finetune','--dataset','breakhis','--manifest',str(self.root/'manifest.json'),
                '--checkpoint',str(self.directory('parent')/'train.pth'),'--initialization',str(prepared['run']/'initialization.json'),
                '--seed','999','--device','cpu','--experiment-id','wrong'])

    def test_final_does_not_require_original_fitting_pixels_or_optimizer(self):
        run,selection=self.parent(); self.v2('dense_continue',parent_selection=str(selection))
        child=self.run_dev('continue')
        for row in self.rows:
            if row['split']!='test': (self.images/row['path']).unlink()
        from breakhis.final_evaluation import run_final
        with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
             patch.object(torch.optim,'Adam',side_effect=AssertionError('optimizer')), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): run_final(run_id='continue')
        self.assertEqual(e.read_json(child/'run.json')['phases']['final']['status'],'completed')

    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA installation/device')
    def test_t07_compact_cuda_training_checkpoint_loads_on_cpu(self):
        with patch.object(c,'create_model',side_effect=ACTUAL_CREATE):
            model=c.create_model('mobilenet_v3_small').cuda(); inputs=torch.randn(2,3,64,64,device='cuda')
            optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
            loss=torch.nn.functional.cross_entropy(model(inputs),torch.tensor([0,1],device='cuda'))
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): expected=model(inputs).cpu()
            path=self.root/'compact-cuda.pth'; c.save_checkpoint(path,model,'mobilenet_v3_small',{})
            loaded,payload=c.load_checkpoint(path,'cpu'); loaded.eval()
            with torch.no_grad(): actual=loaded(inputs.cpu())
        self.assertTrue(torch.allclose(expected,actual,atol=1e-4,rtol=1e-3))
        self.assertTrue(torch.isfinite(actual).all()); self.assertEqual(next(loaded.parameters()).device.type,'cpu')

    def test_parent_pruning_prepares_and_executes_with_m23_controls(self):
        with patch.object(c,'create_model',side_effect=self.tiny_vgg):
            run,selection=self.parent(); self.config['pruning']=dict(method='channel',sparsity=.5,finetune_epochs=1)
            self.v2('prune_from_parent',parent_selection=str(selection),ranking='random',ranking_seed=202)
            s.prepare_run(self.file,run_id='prepared-prune')
            child=self.run_dev('random');a.development(child)
        self.assertFalse((self.directory('random')/'train-last.pth').exists())


if __name__=='__main__': unittest.main()
