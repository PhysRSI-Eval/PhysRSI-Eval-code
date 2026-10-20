import hashlib
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree(root):
    """{relative path: sha256 | 'DIRECTORY'} of a workspace."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): ('DIRECTORY' if p.is_dir() else sha256(p))
            for p in sorted(root.rglob('*')) if not p.is_symlink()}
