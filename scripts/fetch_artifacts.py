"""Fetch checksum-pinned public SLp artifacts without replacing local changes."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from publish_hf import digest, safe_path


def url(spec, name):
    from urllib.parse import quote
    prefix = 'datasets/' if spec['repo_type'] == 'dataset' else ''
    return f'https://huggingface.co/{prefix}{spec["repo_id"]}/resolve/{spec["revision"]}/{quote(name, safe="/")}'


def remote_json(spec, name, expected):
    with urllib.request.urlopen(url(spec, name), timeout=60) as stream:
        payload = stream.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Remote inventory checksum mismatch')
    return json.loads(payload), payload


def fetch_file(spec, remote, target, expected):
    if target.exists():
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f'Refusing to overwrite changed local file: {target}')
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.slp-download-', delete=False) as out:
            temporary = Path(out.name)
            with urllib.request.urlopen(url(spec, remote), timeout=60) as stream:
                shutil.copyfileobj(stream, out, length=1024 * 1024)
        if digest(temporary) != expected:
            raise ValueError(f'Download checksum mismatch: {remote}')
        # Recheck because another process may have produced a file during download.
        if target.exists():
            raise FileExistsError(target)
        # Atomic no-clobber install on both POSIX and Windows, on the same volume.
        os.link(temporary, target)
        temporary.unlink()
        temporary = None
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(kind, root, lock_path, verify_only=False):
    spec = json.loads(lock_path.read_text(encoding='utf-8'))[kind]
    if not spec.get('revision'):
        raise ValueError(f'{kind} publication is not yet pinned in this checkout')
    if kind == 'model':
        base = safe_path(root, spec['local_path'])
        manifest, _ = remote_json(spec, spec['subfolder'] + '/manifest.json', spec['manifest_sha256'])
        rows = [(spec['subfolder'] + '/' + n, n, sha) for n, sha in manifest['files'].items()]
        rows.append((spec['subfolder'] + '/manifest.json', 'manifest.json', spec['manifest_sha256']))
        rows.extend((spec['subfolder'] + '/' + n, n, sha) for n, sha in spec.get('notices', {}).items())
    else:
        base = root
        inventory, _ = remote_json(spec, 'inventory.json', spec['inventory_sha256'])
        rows = [(r['remote'], r['remote'], r['sha256']) for r in inventory['files']]
        rows.extend([('inventory.json', 'results/hf-slp11-r1/inventory.json', spec['inventory_sha256']),
            ('THIRD_PARTY_NOTICES.md', 'results/hf-slp11-r1/THIRD_PARTY_NOTICES.md', spec['notices_sha256'])])
    paths = [(remote, safe_path(base, local), sha) for remote, local, sha in rows]
    # Check all conflicts before starting downloads.
    for _, path, sha in paths:
        if path.exists() and (not path.is_file() or digest(path) != sha):
            raise ValueError(f'Refusing to overwrite changed local file: {path}')
        if verify_only and not path.is_file():
            raise FileNotFoundError(path)
    if not verify_only:
        for index, (remote, path, sha) in enumerate(paths, 1):
            if fetch_file(spec, remote, path, sha):
                print(f'{index}/{len(paths)} {path.relative_to(root)}', flush=True)
    print(f'Verified {len(paths)} files at {base} ({spec["revision"]})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['model', 'data'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    fetch(args.kind, root, args.lock or root / 'artifacts.lock.json', args.verify_only)
