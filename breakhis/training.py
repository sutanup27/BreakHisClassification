import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_device(model):
    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None)
    return parameter.device if parameter is not None else buffer.device if buffer is not None else torch.device('cpu')


def batch_loss(criterion, outputs, targets, phase):
    if outputs.ndim != 2 or outputs.size(0) != targets.size(0):
        raise ValueError(f'{phase}: expected batch-by-class model outputs')
    if not torch.isfinite(outputs).all():
        raise ValueError(f'{phase}: nonfinite model outputs')
    loss = criterion(outputs, targets)
    if loss.ndim != 0:
        raise ValueError(f'{phase}: criterion must return a scalar loss (mean or sum reduction)')
    if not torch.isfinite(loss):
        raise ValueError(f'{phase}: nonfinite loss')
    reduction = getattr(criterion, 'reduction', 'mean')
    if reduction not in ('mean', 'sum'):
        raise ValueError(f'{phase}: unsupported loss reduction {reduction}')
    return loss, loss.detach() * targets.size(0) if reduction == 'mean' else loss.detach()


def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  epoch=0,
  callbacks = None
) -> float:
  model.train()
  from .pruning import enforce_masks
  enforce_masks(model, optimizer)
  device = model_device(model)
  criterion = criterion.to(device)
  total_loss = torch.zeros((), device=device)
  num_samples = 0
  for inputs, targets in tqdm(dataloader, desc=f'train epoch:{epoch}', leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if targets.size(0) == 0:
        continue
    if not torch.isfinite(inputs).all():
        raise ValueError(f'Training epoch {epoch}: nonfinite inputs')

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad(set_to_none=True)

    # Forward inference
    outputs = model(inputs)

    loss, loss_sum = batch_loss(criterion, outputs, targets, f'Training epoch {epoch}')
    total_loss += loss_sum
    num_samples += targets.size(0)


    # Backward propagation
    loss.backward()
    # Validate gradients before updating weights; infinity leaves their scale unchanged.
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    except RuntimeError as error:
        raise ValueError(f'Training epoch {epoch}: nonfinite gradient norm; optimizer step skipped') from error
    # Update optimizer and LR scheduler
    optimizer.step()
    enforce_masks(model, optimizer)

    if callbacks is not None:
        for callback in callbacks:
            callback()

  if num_samples == 0:
      raise ValueError('Cannot train on an empty dataset or loader')
  if not torch.isfinite(total_loss):
      raise ValueError(f'Training epoch {epoch}: nonfinite accumulated loss')
  return (total_loss / num_samples).item()

def predict(model , input):
    # model.to(device)
    # input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input.to(model_device(model)).unsqueeze(0))

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
  device = model_device(model)
  if criterion is not None:
      criterion = criterion.to(device)
  total_loss = torch.zeros((), device=device)
  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False,
                              disable=not verbose):
    # Move the data from CPU to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if targets.size(0) == 0:
        continue
    if not torch.isfinite(inputs).all():
        raise ValueError('Evaluation: nonfinite inputs')

    # Inference
    outputs1 = model(inputs)
    if not torch.isfinite(outputs1).all():
        raise ValueError('Evaluation: nonfinite model outputs')

    # Convert logits to class indices
    outputs = outputs1.argmax(dim=1)

    # Calculate loss
    if criterion is not None:
      _, loss_sum = batch_loss(criterion, outputs1, targets, 'Evaluation')
      total_loss += loss_sum


    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  if num_samples == 0:
    raise ValueError("Cannot evaluate an empty dataset")
  if not torch.isfinite(total_loss):
    raise ValueError('Evaluation: nonfinite accumulated loss')
  return (num_correct / num_samples * 100).item(), (total_loss / num_samples).item()

  

def clone_model(model):
    """Deep-copy a mask-hook model without copying cached nonleaf weight tensors."""
    cached = []
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            cached.append((module, module.weight))
            delattr(module, 'weight')
    try:
        cloned = copy.deepcopy(model)
    finally:
        for module, weight in cached:
            module.weight = weight
    from .pruning import enforce_masks
    enforce_masks(cloned)
    return cloned


def Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=10,scheduler=None):
    if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs < 1:
        raise ValueError('Number of training epochs must be a positive integer')
    losses,test_losses,accs=[],[],[]
    best_acc=float('-inf')
    best_model=None
    for epoch_num in tqdm(range(1, num_epochs + 1)):
        started = time.perf_counter()
        loss=train(model, train_dataloader, criterion, optimizer, epoch=epoch_num)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        acc, val_loss = evaluate(model, test_dataloader, criterion)
        eval_seconds = time.perf_counter() - started
        print(f"Epoch {epoch_num}: train {train_seconds:.1f}s, evaluation {eval_seconds:.1f}s")
        print(f"Training Loss: {loss:.6f}, Validation Loss: {val_loss:.4f}, Validation Accuracy (%): {acc:.4f}")
        if scheduler:
            print(f"LR:{scheduler.get_last_lr()} ")
        if acc>best_acc:
            best_acc=acc
            best_model=clone_model(model)
        losses.append(loss)
        test_losses.append(val_loss)
        accs.append(acc)
        if scheduler is not None:
            scheduler.step()
    return best_model, losses, test_losses, accs


def TrainingPrunned(pruned_model,train_dataloader,test_dataloader,criterion, optimizer, pruner=None,scheduler=None,num_finetune_epochs=5,isCallback=True):
    """Compatibility tuple (best accuracy, model, accuracies); second loader is validation."""
    if not isinstance(num_finetune_epochs, int) or isinstance(num_finetune_epochs, bool) or num_finetune_epochs < 1:
        raise ValueError('Fine-tuning epochs must be a positive integer')
    if pruner is not None and not callable(getattr(pruner, 'apply', None)):
        raise ValueError('Pruner must expose a callable apply(model) method')
    if isCallback and pruner is None and not any(hasattr(module, 'weight_mask') for module in pruned_model.modules()):
        raise ValueError('A mask pruner is required when callbacks are enabled on an unmasked model')
    accuracies=[]
    best_accuracy = float('-inf')
    best_model=None
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask
        #    to keep the model sparse during the training
        if isCallback and pruner is not None:
           callbacks=[lambda: pruner.apply(pruned_model)]
        else:
           callbacks=None
        train_loss=train(pruned_model, train_dataloader, criterion, optimizer, epoch=epoch+1, callbacks=callbacks)
        test_acc ,test_loss = evaluate(pruned_model, test_dataloader,criterion)
        accuracies.append(test_acc)
        is_best = test_acc > best_accuracy
        if is_best:
            best_accuracy = test_acc
            best_model=clone_model(pruned_model)
        if scheduler is not None:
            scheduler.step()
        print(f'    Epoch {epoch+1} Validation accuracy:{test_acc:.2f}% / Best Accuracy: {best_accuracy:.2f}%, train loss: {train_loss:.4f}, validation loss {test_loss:.4f}')
    return best_accuracy, best_model, accuracies
