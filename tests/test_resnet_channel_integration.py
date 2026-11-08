"""R4-R5: public ResNet controls, frozen references, recovery and study expansion."""
import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from breakhis import checkpoints as c,experiments as e,training,baseline_init as b,phase_artifacts as a
from breakhis import resnet_pruning as p,cli,experiment_setup as setup,fit_boundary as boundary
import test_baseline_initialization as fixtures
import test_m1_execution as execution


class ResNetIntegration(unittest.TestCase):
    save=fixtures.InitializationTests.save
    child=fixtures.InitializationTests.child
    run_dev=fixtures.InitializationTests.run_dev
    parent=fixtures.InitializationTests.parent
    v2=fixtures.InitializationTests.v2
    snapshot=fixtures.InitializationTests.snapshot
    model=staticmethod(fixtures.InitializationTests.model)

    def setUp(self):
        fixtures.InitializationTests.setUp(self)
        patch.object(c,'create_model',side_effect=fixtures.ACTUAL_CREATE).start()
        self.config['training']['model']='resnet18';self.save()

    def directory(self,identifier):return self.root/f'checkpoint/binary/40X/resnet18/{identifier}'

    def configure(self,selection,ranking='l1',method='channel',epochs=1):
        self.config['pruning']=dict(method=method,sparsity=.75,finetune_epochs=epochs)
        self.v2('prune_from_parent',parent_selection=str(selection),ranking=ranking,ranking_seed=202 if ranking=='random' else None)

    def test_t14_t15_t18_all_controls_develop_then_test_only_final_and_scratch(self):
        run,selection=self.parent();parent=self.directory('parent')/'train.pth';before=e.digest(parent)
        children=[]
        testfiles={self.images/r['path']:(self.images/r['path']).read_bytes() for r in self.rows if r['split']=='test'}
        for path in testfiles:path.unlink()
        original=training.train;states=[]
        def observe(*args,**kwargs):states.append(len(args[3].state));return original(*args,**kwargs)
        for name,rank,method in [('l1','l1','channel'),('random','random','channel'),('mask','l1','unstructured')]:
            self.configure(selection,rank,method)
            with patch.object(training,'train',side_effect=observe):child=self.run_dev(name)
            children.append(child);a.development(child)
            self.assertNotIn('train',e.read_json(child/'run.json')['stages']);self.assertFalse((child/'train.history.json').exists())
            budget=e.read_json(child/'development/results.json')['baseline_budget'];self.assertEqual(budget['full_lifecycle_fit_epochs'],2)
            self.assertEqual(budget['child_completed_epochs'],1)
        self.assertEqual(states,[0,0,0])
        self.config['pruning']=None;self.v2('dense_continue',parent_selection=str(selection));children.append(self.run_dev('continued'))
        topology=self.directory('l1')/'prune.pth';widths=c.read_checkpoint(topology)['architecture_config']['resnet_channels']
        self.config['training']['epochs']=2
        self.v2('narrow_scratch',parent_selection=str(selection),channels=widths,topology_checkpoint=str(topology))
        record=b.inspect_initialization(e.load_config(self.file),e.digest(self.root/'manifest.json'));fresh=b.fresh_model(record,2,'cpu')
        inherited,_=c.load_checkpoint(topology)
        self.assertFalse(torch.equal(fresh.conv1.weight,inherited.conv1.weight));self.assertTrue(torch.all(fresh.bn1.running_mean==0))
        scratch=self.run_dev('scratch');children.append(scratch)
        budget=e.read_json(scratch/'development/results.json')['baseline_budget'];self.assertEqual(budget['full_lifecycle_fit_epochs'],2);self.assertEqual(budget['including_topology_discovery_epochs'],3)
        for path,data in testfiles.items():path.write_bytes(data)
        for r in self.rows:
            if r['split']!='test':(self.images/r['path']).unlink()
        from breakhis.final_evaluation import run_final
        from breakhis import data
        original=data.get_split_loaders
        def loaders(*args,**kwargs):self.assertEqual(kwargs['splits'],('test',));return original(*args,**kwargs)
        for child in children:
            with patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child),patch.object(data,'get_split_loaders',side_effect=loaders),patch.object(training,'train',side_effect=AssertionError('fit in final')),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                run_final(run_id=child.name)
            self.assertEqual(e.read_json(child/'run.json')['phases']['final']['status'],'completed')
        self.assertEqual(before,e.digest(parent))
        l1=c.read_checkpoint(self.directory('l1')/'prune.pth');random=c.read_checkpoint(self.directory('random')/'prune.pth')
        self.assertEqual(l1['architecture_config'],random['architecture_config'])

    def test_t16_exact_recovery_resume_and_t17_failed_prune_retry(self):
        run,selection=self.parent();self.configure(selection,'random',epochs=2)
        full=self.run_dev('full');original=training.train
        def stop(*args,**kwargs):
            value=original(*args,**kwargs)
            if kwargs['epoch']==1:raise KeyboardInterrupt
            return value
        with patch.object(training,'train',side_effect=stop),self.assertRaises(InterruptedError):self.run_dev('resumed')
        self.run_dev('resumed',resume=True)
        left=c.read_checkpoint(self.directory('full')/'finetune-last.pth');right=c.read_checkpoint(self.directory('resumed')/'finetune-last.pth')
        self.assertTrue(boundary.same(left['state_dict'],right['state_dict']))
        self.assertTrue(boundary.same(execution.ExecutionTests.numerical(left['training_state']),execution.ExecutionTests.numerical(right['training_state'])))
        originalwrite=cli.write_json
        def fail(path,*args,**kwargs):
            if Path(path).name=='prune.validation.json':raise OSError('injected publication failure')
            return originalwrite(path,*args,**kwargs)
        with patch.object(cli,'write_json',side_effect=fail),self.assertRaises(OSError):self.run_dev('retry')
        resumed=self.run_dev('retry',resume=True);a.development(resumed)
        path=self.directory('retry')/'prune.pth';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):a.development(resumed)

    def test_t13_group_settings_and_recipe_errors_precede_pixels(self):
        run,selection=self.parent();self.configure(selection)
        original=self.config['training']['batch_size'];self.config['training']['batch_size']=3;self.save()
        with self.assertRaisesRegex(ValueError,'recipe'):setup.prepare_run(self.file,run_id='bad')
        self.config['training']['batch_size']=original;self.save()
        import json
        groups=self.root/'groups.json';groups.write_text(json.dumps({'stage1':.5}))
        for flag in ('--group-sparsities','--layer-sparsities'):
            with patch('breakhis.data.get_split_loaders',side_effect=AssertionError('pixels')),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                cli.main(['prune','--dataset','breakhis','--manifest',str(self.root/'manifest.json'),'--data-root',str(self.images),'--checkpoint',str(self.directory('parent')/'train.pth'),'--method','channel',flag,str(groups),'--experiment-id','bad'])

    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA device')
    def test_t19_cuda_fit_and_cpu_final(self):
        self.config['training']['device']='cuda';self.save();run,selection=self.parent();self.configure(selection,'random')
        child=self.run_dev('cuda-child')
        from breakhis.final_evaluation import run_final
        with patch.object(torch.cuda,'is_available',return_value=False),patch('breakhis.final_evaluation.subprocess.run',side_effect=self.child),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):run_final(run_id=child.name)
        self.assertEqual(e.read_json(child/'final/finetuned.metrics.json')['evaluation_runtime']['device'],'cpu')


if __name__=='__main__':unittest.main()
