"""Dependency-aware physical surgery for the canonical dense torchvision ResNet18."""
import copy
import torch
from torch import nn
from torchvision.models.resnet import ResNet,BasicBlock
from .models.resnet import ChannelResNet18,validate_widths

GROUPS=tuple([f'stage{s}' for s in range(1,5)]+[f'layer{s}.{b}.hidden' for s in range(1,5) for b in range(2)])
WIDTHS=dict(zip(GROUPS,[64,128,256,512,64,64,128,128,256,256,512,512]))
SCORE='mean-producer-filter-mean-abs-v1'


def producers():
    result={}
    for s in range(1,5):
        result[f'stage{s}']=([('conv1','bn1')] if s==1 else [(f'layer{s}.0.downsample.0',f'layer{s}.0.downsample.1')])+[(f'layer{s}.{b}.conv2',f'layer{s}.{b}.bn2') for b in range(2)]
    result.update({f'layer{s}.{b}.hidden':[(f'layer{s}.{b}.conv1',f'layer{s}.{b}.bn1')] for s in range(1,5) for b in range(2)})
    return result


def validate_source(model):
    if type(model) is not ResNet or hasattr(model,'channel_pruning') or hasattr(model,'resnet_channels'):
        raise ValueError('ResNet surgery requires an unpruned canonical dense ResNet18')
    from .pruning import validate_independent_weights
    # An independently built canonical model supplies the exact structural contract.
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        reference=ResNet(BasicBlock,[2,2,2,2],num_classes=model.fc.out_features)
    actual=dict(model.named_modules());expected=dict(reference.named_modules())
    if set(actual)!=set(expected):raise ValueError('Unsupported ResNet18 module graph')
    for name,old in actual.items():
        new=expected[name]
        if type(old) is not type(new):raise ValueError('Unsupported ResNet18 block/normalization')
        if isinstance(old,nn.Conv2d):
            fields=('in_channels','out_channels','kernel_size','stride','padding','dilation','groups','padding_mode')
            if any(getattr(old,k)!=getattr(new,k) for k in fields) or old.bias is not None or hasattr(old,'weight_mask'):
                raise ValueError('Unsupported ResNet18 convolution')
        elif isinstance(old,nn.BatchNorm2d):
            if old.num_features!=new.num_features or not old.affine or not old.track_running_stats:
                raise ValueError('Unsupported ResNet18 BatchNorm')
        elif isinstance(old,nn.MaxPool2d):
            if any(getattr(old,k)!=getattr(new,k) for k in ('kernel_size','stride','padding','dilation','ceil_mode')):raise ValueError('Unsupported ResNet18 pooling')
        elif isinstance(old,nn.AdaptiveAvgPool2d) and old.output_size!=new.output_size:raise ValueError('Unsupported ResNet18 pooling')
        elif isinstance(old,nn.ReLU) and old.inplace!=new.inplace:raise ValueError('Unsupported ResNet18 activation')
        if old._forward_hooks or old._forward_pre_hooks:raise ValueError('Hooked ResNet18 sources are unsupported')
    if set(dict(model.named_parameters()))!=set(dict(reference.named_parameters())) or any(p.shape!=dict(reference.named_parameters())[name].shape for name,p in model.named_parameters()):
        raise ValueError('Unsupported ResNet18 parameter shapes')
    storage=set();reference_parameter=next(model.parameters())
    for parameter in model.parameters():
        key=(str(parameter.device),parameter.untyped_storage().data_ptr())
        if key in storage:raise ValueError('Shared ResNet parameter storage is unsupported')
        storage.add(key)
        if parameter.device!=reference_parameter.device or parameter.dtype!=reference_parameter.dtype:
            raise ValueError('ResNet source parameters must share device and dtype')
    validate_independent_weights([(name,m) for name,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite ResNet18 state')
    return model


def topology(groups):
    return dict(version=1,stage_widths=[len(groups[f'stage{s}']['retained_indices']) for s in range(1,5)],
        hidden_widths=[[len(groups[f'layer{s}.{b}.hidden']['retained_indices']) for b in range(2)] for s in range(1,5)])


def validate_metadata(config,widths):
    validate_widths(widths)
    if not isinstance(config,dict):raise ValueError('Invalid ResNet channel metadata')
    method=config.get('method');random=method=='random-output-channel'
    keys={'layout','method','group_order','groups','resnet_channels','batch_norm', 'ranking_seed' if random else 'score_rule'}
    if set(config)!=keys or config.get('layout')!='resnet18-dependency-v1' or method not in ('l1-output-channel','random-output-channel') or config['group_order']!=list(GROUPS) or config['resnet_channels']!=widths:
        raise ValueError('Invalid ResNet dependency metadata schema')
    import math
    expected_bn={bn for members in producers().values() for _,bn in members}
    if not isinstance(config['batch_norm'],dict) or set(config['batch_norm'])!=expected_bn:raise ValueError('Invalid ResNet BN settings')
    for row in config['batch_norm'].values():
        if not isinstance(row,dict) or set(row)!={'eps','momentum'}:raise ValueError('Invalid BN settings')
        if type(row['eps']) not in (float,int) or not math.isfinite(row['eps']) or row['eps']<=0:raise ValueError('Invalid BN epsilon')
        if row['momentum'] is not None and (type(row['momentum']) not in (float,int) or not math.isfinite(row['momentum']) or not 0<=row['momentum']<=1):raise ValueError('Invalid BN momentum')
    if not random and config['score_rule']!=SCORE:raise ValueError('Unknown ResNet L1 scoring rule')
    generator=None
    if random:
        seed=config['ranking_seed']
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid random ranking seed')
        generator=torch.Generator(device='cpu').manual_seed(seed)
    if not isinstance(config['groups'],dict) or set(config['groups'])!=set(GROUPS):raise ValueError('Specify all ResNet dependency groups')
    from .pruning import validate_sparsity
    for name in GROUPS:
        row=config['groups'][name]
        if not isinstance(row,dict) or set(row)!={'original_channels','requested_sparsity','retained_indices'}:raise ValueError('Invalid dependency group')
        c=row['original_channels'];ids=row['retained_indices']
        if type(c) is not int or c!=WIDTHS[name]:raise ValueError('Invalid original ResNet width')
        ratio=validate_sparsity(row['requested_sparsity'],channel=True);keep=max(1,c-round(c*ratio))
        if not isinstance(ids,list) or len(ids)!=keep or any(type(i) is not int or not 0<=i<c for i in ids) or ids!=sorted(set(ids)):
            raise ValueError('Invalid retained dependency indices')
        if generator is not None and ids!=torch.randperm(c,generator=generator)[:keep].sort().values.tolist():raise ValueError('Random dependency IDs differ from seed')
    if topology(config['groups'])!=widths:raise ValueError('ResNet topology differs from retained IDs')


@torch.no_grad()
def select_channels(model,sparsity=.5,settings=None,*,ranking='l1',ranking_seed=None):
    validate_source(model)
    from .pruning import layer_settings
    ratios=layer_settings([(k,None) for k in GROUPS],sparsity,settings,channel=True)
    if ranking not in ('l1','random'):raise ValueError('Unknown channel ranking')
    if ranking=='random':
        if type(ranking_seed) is not int or not 0<=ranking_seed<2**32:raise ValueError('Random ranking requires uint32 seed')
        generator=torch.Generator(device='cpu').manual_seed(ranking_seed)
    elif ranking_seed is not None:raise ValueError('L1 ranking seed must be omitted')
    groups={}
    for name,members in producers().items():
        c=WIDTHS[name];keep=max(1,c-round(c*ratios[name]))
        if ranking=='l1':
            scores=torch.stack([model.get_submodule(conv).weight.abs().flatten(1).mean(1) for conv,_ in members]).mean(0)
            ids=torch.argsort(scores,descending=True,stable=True)[:keep].sort().values.cpu().tolist()
        else:ids=torch.randperm(c,generator=generator)[:keep].sort().values.tolist()
        groups[name]=dict(original_channels=c,requested_sparsity=ratios[name],retained_indices=ids)
    cfg=dict(batch_norm={name:dict(eps=m.eps,momentum=m.momentum) for name,m in model.named_modules() if isinstance(m,nn.BatchNorm2d)},layout='resnet18-dependency-v1',method=ranking+'-output-channel',group_order=list(GROUPS),groups=groups,resnet_channels=topology(groups))
    cfg.update({'ranking_seed':ranking_seed} if ranking=='random' else {'score_rule':SCORE})
    validate_metadata(cfg,cfg['resnet_channels'])
    return cfg


@torch.no_grad()
def rebuild(model,config):
    validate_source(model);validate_metadata(config,config['resnet_channels'])
    ref=next(model.parameters())
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        new=ChannelResNet18(config['resnet_channels'],model.fc.out_features)
    new=new.to(device=ref.device,dtype=ref.dtype)
    ids={k:torch.tensor(v['retained_indices'],device=ref.device,dtype=torch.long) for k,v in config['groups'].items()}
    def conv(name,out,inp):new.get_submodule(name).weight.copy_(model.get_submodule(name).weight[out][:,inp])
    def bn(name,index):
        a,b=model.get_submodule(name),new.get_submodule(name)
        for field in ('weight','bias','running_mean','running_var'):getattr(b,field).copy_(getattr(a,field)[index])
        b.num_batches_tracked.copy_(a.num_batches_tracked);b.eps=a.eps;b.momentum=a.momentum
    conv('conv1',ids['stage1'],torch.arange(3,device=ref.device));bn('bn1',ids['stage1'])
    for s in range(1,5):
        out=ids[f'stage{s}']
        for b in range(2):
            prefix=f'layer{s}.{b}';inp=ids[f'stage{s-1}'] if s>1 and b==0 else out;hidden=ids[prefix+'.hidden']
            conv(prefix+'.conv1',hidden,inp);bn(prefix+'.bn1',hidden)
            conv(prefix+'.conv2',out,hidden);bn(prefix+'.bn2',out)
            if s>1 and b==0:conv(prefix+'.downsample.0',out,inp);bn(prefix+'.downsample.1',out)
    new.fc.weight.copy_(model.fc.weight[:,ids['stage4']]);new.fc.bias.copy_(model.fc.bias)
    oldparams=dict(model.named_parameters())
    for name,param in new.named_parameters():param.requires_grad_(oldparams[name].requires_grad)
    new.train(model.training)
    for name,module in new.named_modules():module.training=model.get_submodule(name).training
    new.channel_pruning=copy.deepcopy(config)
    return new


def prune_resnet_channels(model,sparsity=.5,settings=None,*,ranking='l1',ranking_seed=None):
    return rebuild(model,select_channels(model,sparsity,settings,ranking=ranking,ranking_seed=ranking_seed))
