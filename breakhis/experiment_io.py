"""Small shared atomic writers for experiment configuration and reports."""
import json
import os
from pathlib import Path
import tempfile

from .paths import output_path


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate JSON field: '+key)
            result[key]=value
        return result
    result=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))
    json.dumps(result,allow_nan=False)
    return result


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8')


def publish(path, content, *, root=None, protected=(), overwrite=True):
    if not isinstance(content, bytes):
        raise ValueError('Publication requires bytes')
    path = Path(path).expanduser().absolute()
    anchor = Path(root) if root is not None else path.parent
    path = output_path(path, root=anchor, protected=protected)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        output_path(path, root=anchor, protected=protected)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic no-clobber publication on the same volume.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def write_json(path, value, **options):
    # Encode first: malformed/nonfinite content must not create a directory.
    return publish(path, json_bytes(value), **options)
