import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.growth_rate = growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        # concatenate input with new features
        return torch.cat([x, new_features], 1)


# ---------------------------
# Dense Block
# ---------------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.growth_rate = growth_rate
        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate)
            layers.append(layer)
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


# ---------------------------
# Transition Layer
# ---------------------------
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


# ---------------------------
# DenseNet
# ---------------------------
class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),  # DenseNet-121
                 num_init_features=64,
                 classes=10):

        super().__init__()
        self.growth_rate = growth_rate
        self.block_config = block_config
        # initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        channels = num_init_features
        self.blocks = nn.ModuleList()

        # Dense blocks + transition layers
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(num_layers, channels, growth_rate)
            self.blocks.append(dense_block)
            channels = dense_block.out_channels

            if i != len(block_config) - 1:
                trans_out = channels // 2
                trans = TransitionLayer(channels, trans_out)
                self.blocks.append(trans)
                channels = trans_out

        self.final_bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, classes)

    def forward(self, x):
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = F.relu(self.final_bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ---------------------------
# Factory functions
# ---------------------------
def DenseNet121(classes=10):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), classes=classes)


def DenseNet169(classes=10):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), classes=classes)


def DenseNet201(classes=10):
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), classes=classes)

















# # ---------------------------
# # Dense Layer
# # ---------------------------
# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
#         super().__init__()

#         self.norm1 = nn.BatchNorm2d(in_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
#                                kernel_size=1, stride=1, bias=False)

#         self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
#                                kernel_size=3, stride=1, padding=1, bias=False)

#         self.drop_rate = drop_rate

#     def forward(self, x):
#         new_features = self.conv1(self.relu1(self.norm1(x)))
#         new_features = self.conv2(self.relu2(self.norm2(new_features)))

#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

#         # concatenate input with new features
#         return torch.cat([x, new_features], 1)


# # ---------------------------
# # Dense Block
# # ---------------------------
# class DenseBlock(nn.Module):
#     def __init__(self, num_layers, in_channels, growth_rate):
#         super().__init__()
#         layers = []
#         channels = in_channels

#         for _ in range(num_layers):
#             layer = DenseLayer(channels, growth_rate)
#             layers.append(layer)
#             channels += growth_rate

#         self.block = nn.Sequential(*layers)
#         self.out_channels = channels

#     def forward(self, x):
#         return self.block(x)


# # ---------------------------
# # Transition Layer
# # ---------------------------
# class TransitionLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.norm = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=1, stride=1, bias=False)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = self.conv(self.relu(self.norm(x)))
#         x = self.pool(x)
#         return x


# # ---------------------------
# # DenseNet
# # ---------------------------
# class DenseNet(nn.Module):
#     def __init__(self,
#                  growth_rate=32,
#                  block_config=(6, 12, 24, 16),  # DenseNet-121
#                  num_init_features=64,
#                  classes=10):

#         super().__init__()

#         self.block_config = block_config
#         # initial convolution
#         self.init_conv = nn.Sequential(
#             nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(num_init_features),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )

#         channels = num_init_features
#         self.blocks = nn.ModuleList()

#         # Dense blocks + transition layers
#         for i, num_layers in enumerate(block_config):
#             dense_block = DenseBlock(num_layers, channels, growth_rate)
#             self.blocks.append(dense_block)
#             channels = dense_block.out_channels

#             if i != len(block_config) - 1:
#                 trans_out = channels // 2
#                 trans = TransitionLayer(channels, trans_out)
#                 self.blocks.append(trans)
#                 channels = trans_out

#         self.final_bn = nn.BatchNorm2d(channels)
#         self.fc = nn.Linear(channels, classes)

#     def forward(self, x):
#         x = self.init_conv(x)

#         for block in self.blocks:
#             x = block(x)

#         x = F.relu(self.final_bn(x))
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x


# # ---------------------------
# # Factory functions
# # ---------------------------
# def DenseNet121(classes=10):
#     return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), classes=classes)


# def DenseNet169(classes=10):
#     return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), classes=classes)


# def DenseNet201(classes=10):
#     return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), classes=classes)
