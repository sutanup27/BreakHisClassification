"""Explicit ResNet18 widths with canonical torchvision state-dictionary names."""
import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


def validate_widths(config):
    if not isinstance(config,dict) or set(config)!={'version','stage_widths','hidden_widths'} or type(config['version']) is not int or config['version']!=1:
        raise ValueError('Invalid ResNet channel topology schema')
    stages,hidden=config['stage_widths'],config['hidden_widths']
    if (not isinstance(stages,list) or len(stages)!=4 or not isinstance(hidden,list) or len(hidden)!=4 or
        any(not isinstance(row,list) or len(row)!=2 for row in hidden) or
        any(type(n) is not int or n<1 for n in [*stages,*(n for row in hidden for n in row)])):
        raise ValueError('ResNet requires four positive stage and eight hidden widths')
    return config


class ChannelBlock(nn.Module):
    expansion=1
    def __init__(self,inputs,hidden,outputs,stride=1,projection=False):
        super().__init__()
        self.conv1=nn.Conv2d(inputs,hidden,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(hidden);self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(hidden,outputs,3,1,1,bias=False);self.bn2=nn.BatchNorm2d(outputs)
        self.downsample=nn.Sequential(nn.Conv2d(inputs,outputs,1,stride,bias=False),nn.BatchNorm2d(outputs)) if projection else None
        self.stride=stride
    def forward(self,x):
        identity=x if self.downsample is None else self.downsample(x)
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out));out+=identity
        return self.relu(out)


class ChannelResNet18(ResNet):
    def __init__(self,channels,classes=2):
        # Build only the requested topology, retaining torchvision's forward path.
        nn.Module.__init__(self)
        import copy
        self.resnet_channels=copy.deepcopy(validate_widths(channels))
        stages,hidden=channels['stage_widths'],channels['hidden_widths']
        self.conv1=nn.Conv2d(3,stages[0],7,2,3,bias=False);self.bn1=nn.BatchNorm2d(stages[0])
        self.relu=nn.ReLU(inplace=True);self.maxpool=nn.MaxPool2d(3,2,1)
        for s in range(4):
            inputs=stages[s-1] if s else stages[0]
            layer=nn.Sequential(ChannelBlock(inputs,hidden[s][0],stages[s],2 if s else 1,s>0),
                                ChannelBlock(stages[s],hidden[s][1],stages[s]))
            setattr(self,f'layer{s+1}',layer)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.fc=nn.Linear(stages[-1],classes)
        for module in self.modules():
            if isinstance(module,nn.Conv2d):nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(module,nn.BatchNorm2d):nn.init.ones_(module.weight);nn.init.zeros_(module.bias)
