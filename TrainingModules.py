import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm


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
    inputs = inputs.cuda()
    targets = targets.cuda()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        print(input.unsqueeze(0).shape)
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
    inputs = inputs.cuda()
    targets = targets.cuda()

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