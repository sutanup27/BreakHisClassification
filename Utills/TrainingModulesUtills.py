import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  epoch=0,
  callbacks = None
) -> float:
  model.train()
  total_loss = 0
  for inputs, targets in tqdm(dataloader, desc=f'train epoch:{epoch}', leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad()

    # Forward inference
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    total_loss += loss.item()


    # Backward propagation
    loss.backward()
    # Update optimizer and LR scheduler
    optimizer.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()

  return total_loss/len(dataloader)



def predict(model , input):
    # model.to(device)
    # input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        print(input.to(device).unsqueeze(0).shape)
        output = model(input.unsqueeze(0))

    # Get predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class


@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  criterion =None,
  verbose=True,
) :
  model.eval()
  total_loss = float(0)
  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False,
                              disable=not verbose):
    # Move the data from CPU to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Inference
    outputs1 = model(inputs)

    # Convert logits to class indices
    outputs = outputs1.argmax(dim=1)

    # Calculate loss
    if criterion is not None:
      loss = criterion(outputs1, targets)
      total_loss += loss.item()


    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item(), total_loss/len(dataloader)

  

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


def TrainingPrunned(pruned_model,train_dataloader,test_dataloader,criterion, optimizer, pruner,scheduler=None,num_finetune_epochs=5,isCallback=True):
    accuracies=[]
    train_losses=[]
    test_losses=[]
    if scheduler:
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    best_accuracy = 0
    best_model=None
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask
        #    to keep the model sparse during the training
        if isCallback:
           callbacks=[lambda: pruner.apply(pruned_model)]
        else:
           callbacks=None
        train_loss=train(pruned_model, train_dataloader, criterion, optimizer,epoch+1,callbacks=callbacks )
        test_acc ,test_loss = evaluate(pruned_model, test_dataloader,criterion)
        accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        is_best = test_acc > best_accuracy
        if is_best:
            best_accuracy = test_acc
            best_model=copy.deepcopy(pruned_model)
        print(f'    Epoch {epoch+1} Test accuracy:{test_acc:.2f}% / Best Accuracy: {best_accuracy:.2f}%, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
        if scheduler is not None:
          scheduler.step()

    return best_accuracy, best_model, accuracies, train_losses,test_losses