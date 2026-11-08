"""One-writer epoch publication: -last is authoritative; a journal permits repair.

A journal is written before committing -last, then best/history are published.
Without a journal, missing or inconsistent artifacts are errors, not repairs.
"""
from pathlib import Path
import torch
from .checkpoints import cpu_state, read_checkpoint, save_payload
from .experiment_io import read_json, write_json


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(v, right[k]) for k,v in left.items())
    if isinstance(left, (list,tuple)):
        return isinstance(right, type(left)) and len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left == right


def paths(output):
    output=Path(output)
    return output.with_suffix('.history.json'), output.with_suffix('.epoch-pending.json')


def best_payload(boundary):
    best=boundary['training_state']['best']
    return dict(boundary, state_dict=best['state_dict'], epoch=best['epoch'], training_state=None)


def reconcile(output, boundary):
    """Verify normal artifacts, or finish an explicitly journaled publication."""
    output=Path(output); history_path,journal=paths(output)
    state=boundary['training_state']; epoch=boundary['epoch']
    if 'history' not in state or 'best' not in state:
        raise ValueError('Exact resume needs the current epoch publication contract; start a new run')
    history=state['history']
    if not isinstance(history,list) or [r.get('epoch') for r in history] != list(range(1,epoch+1)):
        raise ValueError('Committed history has duplicate, stale or missing epochs')
    best=state['best']
    if epoch and (not isinstance(best,dict) or not 1 <= best.get('epoch',0) <= epoch):
        raise ValueError('Invalid committed best epoch')
    document=dict(schema_version=1,epochs=history,selection_split='validation')
    if journal.exists():
        pending=read_json(journal)
        if (set(pending)!={'previous_epoch','target_epoch'} or
            type(pending['previous_epoch']) is not int or
            pending['target_epoch'] != pending['previous_epoch']+1 or
            epoch not in (pending['previous_epoch'],pending['target_epoch'])):
            raise ValueError('Epoch publication journal does not match completed boundary')
        if best is not None:
            save_payload(output,best_payload(boundary))
        else:
            output.unlink(missing_ok=True)
        write_json(history_path,document)
        journal.unlink()
        return
    if epoch == 0:
        if output.exists(): raise ValueError('Unexpected best checkpoint at epoch zero')
        if history_path.exists() and read_json(history_path)!=document:
            raise ValueError('History differs from committed epoch zero')
        return
    if not output.is_file(): raise ValueError('Best checkpoint is missing')
    actual=read_checkpoint(output); expected=best_payload(boundary)
    if not same(actual,expected): raise ValueError('Best checkpoint differs from committed epoch boundary')
    if not history_path.is_file() or read_json(history_path)!=document:
        raise ValueError('History differs from committed epoch boundary')


def commit(output, latest, *, previous_epoch, save_latest):
    """Persist intent, atomically commit last, then materialize best and history."""
    _,journal=paths(output)
    write_json(journal,dict(previous_epoch=previous_epoch,target_epoch=previous_epoch+1))
    save_latest()
    reconcile(output,read_checkpoint(latest))
