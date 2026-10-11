import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels
        )

        layers = []

        # 1️ Expansion (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # 2️ Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # 3️ Projection (Linear Bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # print(f"Block structure: {self.block(x).shape}")
        y= self.block(x)
        if self.use_res_connect:
        ################# This part is adjusted for channel pruning #############
            if x.shape!=y.shape:
                if y.size(1)<x.size(1):
                    x=x[:,:y.size(1),:,:]
                else:
                    padded = torch.zeros_like(y)
                    padded[:,:x.size(1),:,:]=x
                    x=padded
        #########################################################################
            return x + y
        else:
            return y
        

class MobileNetV2(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()

        # Configuration: (expand_ratio, out_channels, num_blocks, stride)
        # Adapted for CIFAR-10 (32x32) - reduced stride=2 operations
        self.cfgs = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),  # Changed stride 2→1 for CIFAR-10
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 1),  # Changed stride 2→1 for CIFAR-10
            (6, 320, 1, 1),
        ]

        layers = [nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,
                      stride=1, padding=1, bias=False),  # Changed stride=2 to 1 for CIFAR-10
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )]

        in_channels = 32
        for expand_ratio, out_channels, num_blocks, stride in self.cfgs:
            for i in range(num_blocks):
                layers.append(InvertedResidual(
                    in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio
                ))
                in_channels = out_channels

        layers.extend([
            nn.Conv2d(in_channels, 1280,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        ])

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)