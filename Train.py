from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import copy
from TrainingModules import train,evaluate


def Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=10,scheduler=None):
    losses,test_losses,accs=[],[],[]
    best_acc=0
    best_model=None
    for epoch_num in tqdm(range(1, num_epochs + 1)):
        loss=train(model, train_dataloader, criterion, optimizer, epoch=epoch_num)
        acc, val_loss = evaluate(model, test_dataloader, criterion)
        print(f"Training Loss: {loss:.6f} ,Test Loss: {val_loss:.4f}, Test Accuracy {acc:.4f}")
        if scheduler:
            print(f"LR:{scheduler.get_last_lr()} ")
        if acc>best_acc:
            best_acc=acc
            best_model=copy.deepcopy(model)
        losses.append(loss)
        test_losses.append(val_loss)
        accs.append(acc)
        if scheduler is not None:
            scheduler.step()
    return best_model, losses, test_losses, accs
    
