"""Small artifact utilities shared by the self-contained SLp-1.2 module."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
