"""Verify and materialize already-trained weights; perform zero optimization."""
import argparse
import json
from pathlib import Path
import shutil
from io_utils import sha


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.bundle.resolve()
    files = json.loads((source / 'manifest.json').read_text())['files']
    for name, digest in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != digest:
            raise ValueError('Bundle checksum/path mismatch: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in [*files, 'manifest.json']:
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    print(json.dumps({'operation': 'materialize_already_trained_world', 'optimization_steps': 0}))
