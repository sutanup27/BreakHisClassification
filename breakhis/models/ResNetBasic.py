import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, in_channels, out_channels, stride=1,shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # The shortcut/skip connection
        self.shortcut = shortcut

    def forward(self, x):
        identity=x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        ################# This part is adjusted for channel pruning #############
        if out.shape!=identity.shape:
            if out.size(1)<identity.size(1):
                identity=identity[:,:out.size(1),:,:]
            else:
                padded = torch.zeros_like(out)
                padded[:,:identity.size(1),:,:]=identity
                identity=padded
        #########################################################################
        out += identity
        out = self.relu(out)
        return out
    
    
class ResNetBasic(nn.Module):
    def __init__(self, block, layers, num_classes=10):  # CIFAR-10 has 10 classes
        super(ResNetBasic, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AvgPool2d(kernel_size=4)  # or F.adaptive_avg_pool2d(x, 1)

        # Fully connected layer
        self.fc = nn.Linear(512* block.expansion, num_classes)

    def _make_layer(self,block, out_channels, num_blocks, stride):
        shortcut = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride,shortcut))
        self.in_channels = out_channels* block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)  # Average Pooling layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer
        return x

def ResNet18(classes=1000):  return ResNetBasic(BasicBlock, [2, 2, 2, 2],classes)
def ResNet34(classes=1000):  return ResNetBasic(BasicBlock, [3, 4, 6, 3],classes)
