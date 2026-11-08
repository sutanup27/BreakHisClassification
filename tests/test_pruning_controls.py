"""M2-T10--T13: shared parents and conventional matched-width controls."""
import contextlib
import copy
import io
import random
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from breakhis import pruning as p, checkpoints as c, experiments as e, training, phase_artifacts as a
from breakhis import baseline_init as b, experiment_setup as setup, fit_boundary as boundary
from breakhis.models.vgg import VGG
from breakhis.reporting import convolution_linear_macs
import test_baseline_initialization as fixtures
import test_m1_execution as execution


class PruningControlTests(unittest.TestCase):
    save=fixtures.InitializationTests.save
    model=staticmethod(fixtures.InitializationTests.model)
    child=fixtures.InitializationTests.child
    run_dev=fixtures.InitializationTests.run_dev
    directory=fixtures.InitializationTests.directory
    parent=fixtures.InitializationTests.parent
    v2=fixtures.InitializationTests.v2
    snapshot=fixtures.InitializationTests.snapshot

    def setUp(self):
        fixtures.InitializationTests.setUp(self)
        patch.object(c,'create_model',side_effect=self.tiny_vgg).start()

    @staticmethod
    def tiny_vgg(*args,**kwargs): return VGG(classes=2,channels=kwargs.get('channels') or [4]*8)

    def configure(self,selection,ranking='l1',method='channel',epochs=1):
        self.config['pruning']=dict(method=method,sparsity=.5,finetune_epochs=epochs)
        self.v2('prune_from_parent',parent_selection=str(selection),ranking=ranking,
            ranking_seed=202 if ranking=='random' else None)

    def test_t10_l1_ties_random_local_seed_and_source_rng_unchanged(self):
        model=VGG(channels=[8]*8)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,torch.nn.Conv2d): module.weight.fill_(1)
        original=copy.deepcopy(model.state_dict()); rng=c.capture_random_state()
        l1=p.prune_vgg_channels(model,.5)
        first=p.prune_vgg_channels(model,.5,ranking='random',ranking_seed=202)
        again=p.prune_vgg_channels(model,.5,ranking='random',ranking_seed=202)
        other=p.prune_vgg_channels(model,.5,ranking='random',ranking_seed=203)
        self.assertTrue(boundary.same(rng,c.capture_random_state()))
        self.assertTrue(boundary.same(original,model.state_dict()))
        self.assertEqual(l1.channels,first.channels)
        self.assertTrue(all(v['retained_indices']==[0,1,2,3] for v in l1.channel_pruning['layers'].values()))
        self.assertEqual(first.channel_pruning,again.channel_pruning)
        self.assertNotEqual(first.channel_pruning,other.channel_pruning)
        self.assertTrue(any(v['retained_indices']!=[0,1,2,3] for v in first.channel_pruning['layers'].values()))
        self.assertEqual(first.channel_pruning['method'],'random-output-channel')

    def test_t10_invalid_seeds_rankings_and_indices_fail_before_rebuild(self):
        model=self.tiny_vgg()
        for ranking,seed in [('random',None),('random',True),('random',2**32),('bad',1),('l1',1)]:
            with self.subTest(ranking=ranking,seed=seed),self.assertRaises(ValueError):
                p.prune_vgg_channels(model,.5,ranking=ranking,ranking_seed=seed)
        selected,ratios=p.select_vgg_channels(model,.5)
        selected[0]=torch.tensor([0,0])
        with self.assertRaisesRegex(ValueError,'indices'): p.rebuild_vgg_channels(model,selected,ratios)

    def test_t11_exact_slices_bn_classifier_backward_and_random_reload(self):
        model=self.tiny_vgg(); model.eval()
        with torch.no_grad():
            for i in range(8):
                bn=model.backbone.get_submodule(f'bn{i}')
                bn.running_mean.copy_(torch.arange(4.));bn.running_var.fill_(2);bn.num_batches_tracked.fill_(17)
        rebuilt=p.prune_vgg_channels(model,.5,ranking='random',ranking_seed=202)
        previous=torch.arange(3)
        for i,(name,cfg) in enumerate(rebuilt.channel_pruning['layers'].items()):
            retained=torch.tensor(cfg['retained_indices'])
            self.assertTrue(torch.equal(rebuilt.get_submodule(name).weight,model.get_submodule(name).weight[retained][:,previous]))
            self.assertTrue(torch.equal(rebuilt.backbone.get_submodule(f'bn{i}').running_mean,
                model.backbone.get_submodule(f'bn{i}').running_mean[retained]))
            self.assertEqual(rebuilt.backbone.get_submodule(f'bn{i}').num_batches_tracked.item(),17)
            previous=retained
        self.assertTrue(torch.equal(rebuilt.fc2.weight,model.fc2.weight[:,previous]))
        inputs=torch.randn(2,3,32,32); expected=rebuilt(inputs).detach()
        rebuilt.train();torch.nn.functional.cross_entropy(rebuilt(inputs),torch.tensor([0,1])).backward()
        rebuilt.eval(); expected=rebuilt(inputs).detach();path=self.root/'random.pth'
        c.save_checkpoint(path,rebuilt,'vgg',{})
        loaded,payload=c.load_checkpoint(path);loaded.eval()
        self.assertTrue(torch.allclose(expected,loaded(inputs),atol=1e-6,rtol=1e-5))
        self.assertEqual(payload['pruning']['channel_config']['ranking_seed'],202)
        dense_macs=convolution_linear_macs(model,inputs); small_macs=convolution_linear_macs(loaded,inputs)
        self.assertLess(small_macs,dense_macs)
        masked=p.prune_weights(copy.deepcopy(model),.5)
        self.assertEqual(convolution_linear_macs(masked,inputs),dense_macs)

    def test_t11_corrupt_random_metadata_and_mixed_residual_refuse(self):
        model=p.prune_vgg_channels(self.tiny_vgg(),.5,ranking='random',ranking_seed=202)
        path=self.root/'model.pth'; c.save_checkpoint(path,model,'vgg',{});original=c.read_checkpoint(path)
        for change in ('seed','different_seed','indices','method'):
            payload=copy.deepcopy(original);meta=payload['pruning']['channel_config']
            if change=='seed': meta['ranking_seed']=True
            if change=='different_seed': meta['ranking_seed']=203
            if change=='indices': meta['layers']['backbone.conv0']['retained_indices']=[0,0]
            if change=='method': meta['method']='invented'
            c.save_payload(path,payload)
            with self.subTest(change=change),self.assertRaises(ValueError): c.read_checkpoint(path)
        with self.assertRaises(ValueError): p.prune_vgg_channels(torch.nn.Linear(2,2),.5)
        with self.assertRaises(ValueError): p.prune_vgg_channels(p.prune_weights(self.tiny_vgg(),.5),.5)
        with self.assertRaises(ValueError): p.prune_weights(model,.5)

    def test_t12_shared_l1_random_mask_all_recover_from_same_parent_with_fresh_optimizer(self):
        run,selection=self.parent(); before=self.snapshot(run); original=training.train
        for name,rank,method in [('l1','l1','channel'),('random','random','channel'),('mask','l1','unstructured')]:
            self.configure(selection,rank,method); states=[]
            def observe(*args,**kwargs): states.append(len(args[3].state)); return original(*args,**kwargs)
            with patch.object(training,'train',side_effect=observe): child=self.run_dev(name)
            a.development(child); state=e.read_json(child/'run.json')
            self.assertEqual(states,[0]);self.assertNotIn('train',state['stages'])
            self.assertFalse((child/'train.history.json').exists())
            self.assertEqual(len(e.read_json(child/'finetune.history.json')['epochs']),1)
            selected=e.read_json(child/'development/selection.json')
            self.assertEqual(selected['roles']['dense']['checkpoint']['sha256'],e.digest(self.directory('parent')/'train.pth'))
            budget=e.read_json(child/'development/results.json')['baseline_budget']
            self.assertEqual(budget['child_completed_epochs'],1); self.assertEqual(budget['full_lifecycle_fit_epochs'],2)
            self.assertEqual(budget['shared_parent_fit_epochs'],1)
            if method=='unstructured':
                payload=c.read_checkpoint(self.directory(name)/'finetune-last.pth')
                for key,value in payload['state_dict'].items():
                    if key.endswith('weight_mask'):
                        original_key=key.removesuffix('_mask')+'_orig'
                        self.assertTrue(torch.all(payload['state_dict'][original_key][value==0]==0))
        self.assertEqual(before,self.snapshot(run))
        l1=c.read_checkpoint(self.directory('l1')/'prune.pth');randomized=c.read_checkpoint(self.directory('random')/'prune.pth')
        self.assertEqual(l1['architecture_config'],randomized['architecture_config'])

    def test_t12_parent_budget_batch_and_epoch_contract_cannot_be_hidden(self):
        run,selection=self.parent(); self.configure(selection)
        for field,value in [('batch_size',3),('epochs',5)]:
            original=self.config['training'][field];self.config['training'][field]=value;self.save()
            with self.subTest(field=field),self.assertRaises(ValueError): setup.prepare_run(self.file,run_id='bad')
            self.config['training'][field]=original;self.save()
        prepared=setup.prepare_run(self.file,run_id='prepared')
        self.assertEqual(prepared['inputs']['initialization']['contract']['epochs'],1)
        metadata=dict(manifest_sha256=prepared['inputs']['identity']['manifest_sha256'],source='cancer_binary',magnification='40X',preprocessing=e.PREPROCESSING)
        record=prepared['inputs']['initialization'];cfg=dict(record['contract'],command='finetune',epochs=99)
        with self.assertRaisesRegex(ValueError,'epochs'): b.verify_cli(record,cfg,metadata,resume=True)
        # Scratch receives its own B+F fit; discovery cost is listed separately.
        self.config['pruning']=None;self.config['training']['epochs']=3
        self.v2('narrow_scratch',parent_selection=str(selection),channels=[2]*8,topology_checkpoint='unused')
        fake_record=dict(parent=dict(executed_epochs=2,selected_epoch=1,optimizer_steps=4))
        result=b.budget_summary(self.config,fake_record,{'train':[dict(optimizer_steps=2)]*3})
        self.assertEqual(result['full_lifecycle_fit_epochs'],3);self.assertEqual(result['including_topology_discovery_epochs'],5)

    def test_t12_random_recovery_resume_matches_full_and_parent_is_unchanged(self):
        run,selection=self.parent(); before=self.snapshot(run);self.configure(selection,'random',epochs=3)
        full=self.run_dev('full'); original=training.train
        def stop(*args,**kwargs):
            value=original(*args,**kwargs)
            if kwargs['epoch']==2: raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=stop),self.assertRaises(InterruptedError): self.run_dev('resumed')
        child=self.run_dev('resumed',resume=True)
        left=c.read_checkpoint(self.directory('full')/'finetune-last.pth');right=c.read_checkpoint(self.directory('resumed')/'finetune-last.pth')
        self.assertTrue(boundary.same(left['state_dict'],right['state_dict']))
        self.assertTrue(boundary.same(execution.ExecutionTests.numerical(left['training_state']),execution.ExecutionTests.numerical(right['training_state'])))
        self.assertEqual(before,self.snapshot(run))

    def test_t13_development_without_test_then_explicit_test_only_final_for_all_controls(self):
        run,selection=self.parent(); original_data={}
        for row in self.rows:
            if row['split']=='test':
                path=self.images/row['path']; original_data[path]=path.read_bytes();path.unlink()
        children=[]
        for name,rank,method in [('l1','l1','channel'),('random','random','channel'),('mask','l1','unstructured')]:
            self.configure(selection,rank,method);children.append(self.run_dev(name))
        for path,data in original_data.items():path.write_bytes(data)
        for row in self.rows:
            if row['split']!='test': (self.images/row['path']).unlink()
        from breakhis.final_evaluation import run_final
        from breakhis import data
        original=data.get_split_loaders
        def loaders(*args,**kwargs):
            self.assertEqual(kwargs['splits'],('test',));return original(*args,**kwargs)
        for child in children:
            with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
                 patch.object(data,'get_split_loaders',side_effect=loaders), \
                 patch.object(training,'train',side_effect=AssertionError('fitting')), \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): run_final(run_id=child.name)
            state=e.read_json(child/'run.json');self.assertEqual(state['phases']['final']['status'],'completed')
            report=e.read_json(child/'final/finetuned.metrics.json')
            self.assertEqual(report['split'],'test');self.assertEqual(report['baseline_kind'],'prune_from_parent')
            self.assertEqual(report['ranking'],'random' if child.name=='random' else 'l1')

    def test_t13_changed_origin_or_completed_control_artifact_refuses(self):
        run,selection=self.parent(); self.configure(selection,'random');child=self.run_dev('random')
        path=self.directory('random')/'prune.pth';before=path.read_bytes();path.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'changed'): a.development(child)
        path.write_bytes(before);parent=self.directory('parent')/'train.pth';parent.write_bytes(parent.read_bytes()+b' ')
        from breakhis.final_evaluation import run_final
        with self.assertRaisesRegex(ValueError,'changed'): run_final(run_id='random')

    def test_t13_failed_prune_publication_retries_only_owned_uncommitted_outputs(self):
        run,selection=self.parent();before=self.snapshot(run);self.configure(selection,'random')
        from breakhis import cli
        original=cli.write_json
        def fail(path,*args,**kwargs):
            if Path(path).name=='prune.validation.json': raise OSError('injected pruning publication failure')
            return original(path,*args,**kwargs)
        with patch.object(cli,'write_json',side_effect=fail),self.assertRaises(OSError): self.run_dev('retry')
        child=self.root/'runs/retry';self.assertEqual(e.read_json(child/'run.json')['stages']['prune']['status'],'failed')
        self.assertTrue((self.directory('retry')/'prune.config.json').exists())
        self.run_dev('retry',resume=True);a.development(child)
        self.assertEqual(before,self.snapshot(run))

    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA installation/device')
    def test_t13_cuda_origin_random_surgery_preserves_rng_and_cpu_final(self):
        self.config['training']['device']='cuda';self.save();run,selection=self.parent()
        payload=c.read_checkpoint(self.directory('parent')/'train.pth');model,_=c.load_checkpoint('unused',device='cuda',payload=payload)
        state=c.capture_random_state(); candidate=p.prune_vgg_channels(model,.5,ranking='random',ranking_seed=202)
        self.assertTrue(boundary.same(state,c.capture_random_state()))
        self.configure(selection,'random');child=self.run_dev('random')
        from breakhis.final_evaluation import run_final
        with patch.object(torch.cuda,'is_available',return_value=False), \
             patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):run_final(run_id='random')
        self.assertEqual(e.read_json(child/'final/finetuned.metrics.json')['evaluation_runtime']['device'],'cpu')


if __name__=='__main__':unittest.main()
