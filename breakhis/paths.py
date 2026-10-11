"""Checkpoint locations shared by command entry points."""
from pathlib import Path

CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / 'checkpoint'


def checkpoint_directory(source):
    tasks = {'breakhis_v2': 'binary', 'cancer_binary': 'binary',
             'cancer_multiclass': 'multiclass'}
    if source not in tasks:
        raise ValueError(f'Unknown dataset source: {source}')
    return CHECKPOINT_ROOT / tasks[source]


def checkpoint_output(source, magnification, architecture, command, output=None):
    """Keep every command checkpoint inside its classification task directory."""
    directory = checkpoint_directory(source).resolve()
    if output is None:
        return directory / magnification / architecture / f'{command}.pth'
    output = Path(output)
    # A filename is relative to the task directory; explicit paths use the cwd.
    path = (directory / output if output.parent == Path('.') else output).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f'Checkpoint output must be inside {directory}')
    if path.suffix.lower() != '.pth':
        raise ValueError('Checkpoint output must have a .pth extension')
    return path
