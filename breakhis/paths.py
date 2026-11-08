"""Checkpoint locations shared by command entry points."""
from pathlib import Path
import os
import re

CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / 'checkpoint'


def checkpoint_directory(source):
    tasks = {'breakhis_v2': 'binary', 'cancer_binary': 'binary',
             'cancer_multiclass': 'multiclass'}
    if source not in tasks:
        raise ValueError(f'Unknown dataset source: {source}')
    return CHECKPOINT_ROOT / tasks[source]


def validate_experiment_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Experiment ID must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit')
    return value


def output_path(path, *, root, protected=()):
    """Confine a publication path before resolution can hide a linked ancestor.

    Static path checks prevent accidental writes, not hostile concurrent path
    replacement. Protected entries can be files or directories, including missing
    inputs; descendants are forbidden as well.
    """
    path = Path(os.path.abspath(Path(path).expanduser()))
    anchor = Path(os.path.abspath(Path(root).expanduser()))
    if path == anchor or not path.is_relative_to(anchor):
        raise ValueError(f'Output must be a child of {anchor}')
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError(f'Output traverses a linked path: {candidate}')
    resolved = path.resolve()
    if not resolved.is_relative_to(anchor.resolve()):
        raise ValueError('Output escapes its root')
    for value in protected:
        blocked = Path(value).expanduser().resolve()
        if resolved == blocked or resolved.is_relative_to(blocked):
            raise ValueError(f'Output overlaps protected input: {blocked}')
    return resolved


def checkpoint_output(source, magnification, architecture, command, output=None, experiment_id=None):
    """Keep every command checkpoint inside its classification task directory."""
    directory = checkpoint_directory(source)
    if output is None:
        run = directory / magnification / architecture
        if experiment_id is not None:
            run = run / validate_experiment_id(experiment_id)
        return output_path(run / f'{command}.pth', root=directory)
    output = Path(output)
    # A filename is relative to the task directory; explicit paths use the cwd.
    path = output_path(directory / output if output.parent == Path('.') else output, root=directory)
    if path.suffix.lower() != '.pth':
        raise ValueError('Checkpoint output must have a .pth extension')
    return path
