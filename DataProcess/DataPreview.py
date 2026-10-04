from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from DataPreprocessing import  get_datasets

path='./dataset/cifar10'
out_path='./output'
train_dataloader,test_datasets=get_datasets(path)

samples = [[] for _ in range(10)]
for image, label in test_datasets:
  print(label)
  if len(samples[label]) < 4:
    samples[label].append(image)

plt.figure(figsize=(20, 9))
for index in range(40):
  label = index % 10
  image = samples[label][index // 10]

  # Convert from CHW to HWC for visualization
  image = image.permute(1, 2, 0)

  # Convert from class index to class name
  label = test_datasets.classes[label]

  # Visualize the image
  plt.subplot(4, 10, index + 1)
  plt.imshow(image)
  plt.title(label)
  plt.axis("off")
plt.show()
plt.savefig(out_path+'/sample.png')

print("End")
