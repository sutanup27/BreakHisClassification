import os
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from DataPreprocessing import get_datasets,test_transform,train_transform

path='H:\sutanu\BreaKHis\BreaKHis_v2'
sub_dirs=['400X','200X','100X','40X']
types=['train','test']

dataset={}

root_dir,image_dataset={},{}
for d in sub_dirs:
    root_dir[d]=os.path.join(path, d)
    dataset[d]={}
    dataset[d]['train'],dataset[d]['test']=get_datasets(root_dir[d],train_transform, test_transform)

print(dataset)
width=2
fig, ax = plt.subplots(4, width, figsize=(15, 20))
fig.set_facecolor('lightgrey')
i,j=0,0

for d in sub_dirs:
    j=0
    for t in types:
        image=dataset[d][t][0][0]
        print(f"Magnification:{d}, Type:{t}, size: {image.shape}")
        ax[i,j].imshow(image.permute(1, 2, 0),cmap='gray')
        ax[i,j].set_title(f"Mag:{d}, Type:{t}")
        ax[i,j].axis("off")
        j=j+1
    i=i+1
plt.show()
