"""R1-R3: exact dependency surgery, independent oracle and portable reconstruction."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torchvision.models import resnet18
from breakhis import checkpoints as c,fit_boundary as boundary
from breakhis import resnet_pruning as p
from breakhis.models.resnet import ChannelResNet18,validate_widths
from breakhis.reporting import convolution_linear_macs


class ResNetChannels(unittest.TestCase):
    def setUp(self):
        threads=torch.get_num_threads();torch.set_num_threads(2);self.addCleanup(torch.set_num_threads,threads)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.model=resnet18(weights=None,num_classes=2).eval()

    def test_t01_dependency_groups_cover_all_convs_and_bn(self):
        groups=p.producers();self.assertEqual(len(groups),12)
        convs=[conv for members in groups.values() for conv,bn in members]
        self.assertEqual(len(convs),20);self.assertEqual(len(set(convs)),20)
        self.assertEqual(set(convs),{k for k,m in self.model.named_modules() if isinstance(m,nn.Conv2d)})
        self.assertIn(('conv1','bn1'),groups['stage1'])
        for s in range(2,5):self.assertIn((f'layer{s}.0.downsample.0',f'layer{s}.0.downsample.1'),groups[f'stage{s}'])

    def test_t02_t04_zero_pruning_matches_torchvision_and_keeps_transitions(self):
        inputs=torch.randn(2,3,64,64);rng=c.capture_random_state()
        result=p.prune_resnet_channels(self.model,0)
        self.assertTrue(boundary.same(rng,c.capture_random_state()))
        self.assertEqual(set(self.model.state_dict()),set(result.state_dict()))
        self.assertTrue(torch.allclose(self.model(inputs),result(inputs),atol=1e-6,rtol=1e-5))
        cfg=p.select_channels(self.model,.5)
        # Different original stage widths can be reduced to the same retained width.
        for name in p.GROUPS:
            if name.startswith('stage'):
                cfg['groups'][name]['retained_indices']=[0]
                cfg['groups'][name]['requested_sparsity']=1-1/p.WIDTHS[name]
        cfg['resnet_channels']=p.topology(cfg['groups']);result=p.rebuild(self.model,cfg)
        for s in range(2,5):self.assertIsNotNone(getattr(result,f'layer{s}')[0].downsample)

    def test_t03_unsupported_layouts_masks_nonfinite_shared_and_reprune_fail(self):
        for mutate in [lambda m:setattr(m.layer1[0].conv1,'groups',2),
                       lambda m:setattr(m.layer2[0].conv1,'dilation',(2,2)),
                       lambda m:setattr(m.layer1[0].bn1,'track_running_stats',False),
                       lambda m:setattr(m.layer1[0],'downsample',nn.Identity()),
                       lambda m:setattr(m.layer1[1].conv2,'weight',m.layer1[0].conv2.weight),
                       lambda m:setattr(m.layer1[1].bn1,'weight',nn.Parameter(m.layer1[0].bn1.weight.view(-1)))]:
            model=copy.deepcopy(self.model);mutate(model)
            with self.assertRaises(ValueError):p.select_channels(model)
        model=copy.deepcopy(self.model);model.bn1.running_mean[0]=float('nan')
        with self.assertRaises(ValueError):p.select_channels(model)
        from breakhis.pruning import prune_weights
        with self.assertRaises(ValueError):p.select_channels(prune_weights(copy.deepcopy(self.model)))
        with self.assertRaises(ValueError):p.select_channels(p.prune_resnet_channels(self.model))
        with self.assertRaises(ValueError):prune_weights(p.prune_resnet_channels(self.model))
        for widths in [dict(version=1,stage_widths=[True]*4,hidden_widths=[[2,2]]*4),dict(version=2,stage_widths=[1]*4,hidden_widths=[[1,1]]*4)]:
            with self.assertRaises(ValueError):validate_widths(widths)

    def test_t05_hand_computed_l1_group_scoring_and_stable_ties(self):
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module,nn.Conv2d):module.weight.fill_(1)
            self.model.conv1.weight[7].fill_(4)
            self.model.layer1[0].conv2.weight[8].fill_(7)
        cfg=p.select_channels(self.model,.95)
        # stage1 means: ID8=3, ID7=2, all others=1; stable tie selects ID0.
        self.assertEqual(cfg['groups']['stage1']['retained_indices'],[0,7,8])
        self.assertEqual(cfg['groups']['layer1.0.hidden']['retained_indices'],[0,1,2])
        self.assertEqual(cfg['score_rule'],p.SCORE)

    def test_t06_random_seed_widths_source_and_all_rng_preserved(self):
        original=copy.deepcopy(self.model.state_dict());rng=c.capture_random_state()
        a=p.prune_resnet_channels(self.model,.5,ranking='random',ranking_seed=202)
        b=p.prune_resnet_channels(self.model,.5,ranking='random',ranking_seed=202)
        l1=p.prune_resnet_channels(self.model,.5)
        self.assertEqual(a.channel_pruning,b.channel_pruning);self.assertEqual(a.resnet_channels,l1.resnet_channels)
        self.assertTrue(boundary.same(original,self.model.state_dict()));self.assertTrue(boundary.same(rng,c.capture_random_state()))
        for rank,seed in [('l1',2),('random',True),('random',None),('random',2**32),('bad',None)]:
            with self.assertRaises(ValueError):p.select_channels(self.model,ranking=rank,ranking_seed=seed)
        other=p.select_channels(self.model,.5,ranking='random',ranking_seed=203)
        self.assertNotEqual(a.channel_pruning['groups'],other['groups'])

    def test_t07_exact_slices_bn_settings_freezing_and_classifier(self):
        with torch.no_grad():
            for name,m in self.model.named_modules():
                if isinstance(m,nn.BatchNorm2d):
                    m.running_mean.copy_(torch.arange(m.num_features));m.running_var.fill_(2);m.num_batches_tracked.fill_(19)
                    m.eps=.002;m.momentum=None
        self.model.conv1.weight.requires_grad_(False)
        cfg=p.select_channels(self.model,.6,ranking='random',ranking_seed=202);new=p.rebuild(self.model,cfg)
        ids={k:torch.tensor(v['retained_indices']) for k,v in cfg['groups'].items()}
        previous=ids['stage3'];out=ids['stage4'];hidden=ids['layer4.0.hidden']
        self.assertTrue(torch.equal(new.layer4[0].conv1.weight,self.model.layer4[0].conv1.weight[hidden][:,previous]))
        self.assertTrue(torch.equal(new.layer4[0].conv2.weight,self.model.layer4[0].conv2.weight[out][:,hidden]))
        self.assertTrue(torch.equal(new.layer4[0].downsample[0].weight,self.model.layer4[0].downsample[0].weight[out][:,previous]))
        for group,members in p.producers().items():
            for _,name in members:
                a,b=self.model.get_submodule(name),new.get_submodule(name)
                for field in ('weight','bias','running_mean','running_var'):self.assertTrue(torch.equal(getattr(b,field),getattr(a,field)[ids[group]]))
                self.assertEqual(b.num_batches_tracked.item(),19);self.assertEqual(b.eps,.002);self.assertIsNone(b.momentum)
        self.assertTrue(torch.equal(new.fc.weight,self.model.fc.weight[:,out]));self.assertTrue(torch.equal(new.fc.bias,self.model.fc.bias))
        self.assertFalse(new.conv1.weight.requires_grad);self.assertFalse(new.training)
        path=self.root/'model.pth';c.save_checkpoint(path,new,'resnet',{});loaded,_=c.load_checkpoint(path)
        self.assertEqual(loaded.bn1.eps,.002);self.assertIsNone(loaded.bn1.momentum)

    def test_t08_independent_gated_oracle_nonprefix_unequal_topology(self):
        settings={k:.75 if k.startswith('stage') else .875 for k in p.GROUPS}
        result=p.prune_resnet_channels(self.model,.5,settings,ranking='random',ranking_seed=202)
        # Build independent hooks from explicit semantic locations, not rebuild edges.
        cfg=result.channel_pruning;handles=[]
        def gate(group):
            mask=torch.zeros(p.WIDTHS[group]);mask[cfg['groups'][group]['retained_indices']]=1
            return lambda module,args,out:out*mask.to(out).view(1,-1,1,1)
        handles.append(self.model.bn1.register_forward_hook(gate('stage1')))
        for s in range(1,5):
            for b in range(2):
                block=getattr(self.model,f'layer{s}')[b]
                handles.append(block.bn1.register_forward_hook(gate(f'layer{s}.{b}.hidden')))
                handles.append(block.bn2.register_forward_hook(gate(f'stage{s}')))
                if block.downsample is not None:handles.append(block.downsample[1].register_forward_hook(gate(f'stage{s}')))
        try:
            with torch.no_grad():
                for size in (64,224):
                    for batch in (1,2):
                        x=torch.randn(batch,3,size,size)
                        self.assertTrue(torch.allclose(result(x),self.model(x),atol=1e-6,rtol=1e-5),(size,batch))
        finally:
            for handle in handles:handle.remove()

    def test_t09_t10_physical_shapes_gradients_macs_and_parameters(self):
        result=p.prune_resnet_channels(self.model,.75);x=torch.randn(2,3,64,64)
        self.assertLess(sum(v.numel() for v in result.parameters()),sum(v.numel() for v in self.model.parameters()))
        self.assertLess(convolution_linear_macs(result,x),convolution_linear_macs(self.model,x))
        self.assertFalse(any(hasattr(m,'weight_mask') for m in result.modules()))
        result.train();opt=torch.optim.Adam(result.parameters());nn.functional.cross_entropy(result(x),torch.tensor([0,1])).backward();opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in result.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in result.parameters()))

    def test_t11_t12_checkpoint_roundtrip_and_corrupt_metadata(self):
        result=p.prune_resnet_channels(self.model,.75,ranking='random',ranking_seed=202);path=self.root/'model.pth'
        c.save_checkpoint(path,result,'resnet18',{});loaded,payload=c.load_checkpoint(path);loaded.eval();x=torch.randn(2,3,64,64)
        self.assertTrue(torch.allclose(result(x),loaded(x),atol=1e-6,rtol=1e-5))
        for change in ('seed','width','ids','class','layout'):
            bad=copy.deepcopy(payload);cfg=bad['pruning']['channel_config']
            if change=='seed':cfg['ranking_seed']=203
            elif change=='width':bad['architecture_config']['resnet_channels']['stage_widths'][0]+=1
            elif change=='ids':cfg['groups']['stage1']['retained_indices'][0]=63
            elif change=='class':bad['state_dict']['fc.weight']=bad['state_dict']['fc.weight'][:1]
            else:cfg['layout']='other'
            c.save_payload(path,bad)
            with self.subTest(change=change),self.assertRaises(ValueError):c.read_checkpoint(path)

    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA device')
    def test_t19_cuda_surgery_rng_and_cpu_reconstruction(self):
        self.model.cuda();rng=c.capture_random_state();new=p.prune_resnet_channels(self.model,.75,ranking='random',ranking_seed=202)
        self.assertTrue(boundary.same(rng,c.capture_random_state()));path=self.root/'cuda.pth'
        x=torch.randn(2,3,64,64,device='cuda');expected=new(x).detach().cpu();c.save_checkpoint(path,new,'resnet',{})
        with patch.object(torch.cuda,'is_available',return_value=False):loaded,_=c.load_checkpoint(path,'cpu')
        loaded.eval();self.assertTrue(torch.allclose(expected,loaded(x.cpu()),atol=1e-4,rtol=1e-3))


if __name__=='__main__':unittest.main()
