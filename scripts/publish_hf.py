"""Publish an explicit, hashed file plan without staging copies of large files.

Plans are local release artifacts, not credentials. Authentication uses the
Hugging Face credential store. Existing payloads may only be resumed unchanged;
only explicitly named mutable cards can be replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe path: {name}')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes root: {name}')
    return path


def publish(plan_path, root, receipt, execute=False):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    rows = plan['files']
    names = [row['remote'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate destination paths')
    for row in rows:
        safe_path(root, row['remote'])
        path = safe_path(root, row['local'])
        if path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            raise ValueError(f'Changed publication input: {path}')
    print(json.dumps({'repo': plan['repo_id'], 'type': plan['repo_type'],
                      'files': len(rows), 'bytes': sum(r['size'] for r in rows),
                      'execute': execute}), flush=True)
    if not execute:
        return
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi()
    api.create_repo(plan['repo_id'], repo_type=plan['repo_type'], private=False, exist_ok=True)
    info = api.repo_info(plan['repo_id'], repo_type=plan['repo_type'], files_metadata=True)
    existing = {f.rfilename: f for f in info.siblings}
    pending = []
    for row in rows:
        old = existing.get(row['remote'])
        if old is not None:
            if old.lfs:
                old_hash = old.lfs.sha256
            else:
                old_hash = digest(hf_hub_download(plan['repo_id'], row['remote'],
                    repo_type=plan['repo_type'], revision=info.sha))
            if old_hash == row['sha256']:
                continue
            if row['remote'] not in plan.get('mutable', []):
                raise ValueError(f'Refusing to overwrite release payload: {row["remote"]}')
        pending.append(row)
    head = info.sha
    for start in range(0, len(pending), 25):
        batch = pending[start:start + 25]
        result = api.create_commit(plan['repo_id'], repo_type=plan['repo_type'],
            parent_commit=head, commit_message=plan['message'], num_threads=2,
            operations=[CommitOperationAdd(path_in_repo=r['remote'],
                        path_or_fileobj=str(safe_path(root, r['local']))) for r in batch])
        head = result.oid
        print(f'Committed {min(start + 25, len(pending))}/{len(pending)} files: {head}', flush=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({'repo_id': plan['repo_id'], 'repo_type': plan['repo_type'],
        'revision': head, 'files': len(rows), 'bytes': sum(r['size'] for r in rows),
        'plan_sha256': digest(plan_path)}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    publish(args.plan, args.root.resolve(), args.receipt, args.execute)
