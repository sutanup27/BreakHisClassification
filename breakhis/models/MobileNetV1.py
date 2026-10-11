import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Depthwise Separable Conv Block
# -----------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


# -----------------------------
# MobileNet V1
# -----------------------------
class MobileNetV1(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()

        self.model = nn.Sequential(
            # Changed stride=2 to stride=1 for CIFAR-10's small 32x32 images
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 1),  # Changed stride=2 to 1
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),  # Keep one stride=2
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),  # Keep one stride=2

            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],

            DepthwiseSeparableConv(512, 1024, 1),  # Changed stride=2 to 1
            DepthwiseSeparableConv(1024, 1024, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x