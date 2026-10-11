from collections import defaultdict, OrderedDict
import torch
from torch import nn

class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
  #ARCH = [32 , 32, 'M', 64, 64,'M',128, 128, 'M', 256, 256, 'M', 512, 512]

  def __init__(self, classes=2, channels=None) -> None:
    super().__init__()
    expected = sum(value != 'M' for value in self.ARCH)
    self.channels = list(channels) if channels is not None else [value for value in self.ARCH if value != 'M']
    if len(self.channels) != expected or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.channels):
      raise ValueError(f'VGG channels must contain {expected} positive integer widths')

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    channel_index = 0
    for x in self.ARCH:
      if x != 'M':
        x = self.channels[channel_index]
        channel_index += 1
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))

    self.backbone = nn.Sequential(OrderedDict(layers))
#    self.fc1 = nn.Linear(2024, 500)
    self.fc2 = nn.Linear(in_channels, classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    x = x.mean([2, 3])

    # classifier: [N, 512] => [N, 10]
#    x = self.fc1(x)
    x = self.fc2(x)
    return x


def recover_model(PATH,model):
  from ..pruning_legacy import download_url
  cp = torch.load(download_url(PATH), map_location="cpu")
  return model.load_state_dict(cp)


