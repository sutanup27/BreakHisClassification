import os
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


def main():
    path=os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'BreaKHis_v2', '40X')

    # Load images using ImageFolder
    image_dataset = ImageFolder(root=path, transform=ToTensor())

    samples={}
    # Before transformation
    samples['Before']=[[],[]]
    for d , l in image_dataset:
        if len(samples['Before'][l])< 2 :
            samples['Before'][l].append(d)
    #    print(len(samples[0]),len(samples[1]))
        if len(samples['Before'][0])==4 and len(samples['Before'][1])==4:
            break

    # Plot the images
    width=2
    fig, ax = plt.subplots(2, width, figsize=(40, 40))
    fig.set_facecolor('lightgrey')
    e_status=image_dataset.classes
    for l in [0,1]:
        for i in range(2):
            image=samples['Before'][l][i]
            print(image.shape)
            ax[i,l].imshow(image.permute(1, 2, 0),cmap='gray')
            ax[i,l].set_title(f"Label:{e_status[l]}", fontsize=50)
            ax[i,l].axis("off")
    plt.show()

    print("End")

if __name__ == "__main__":
    main()
