"""Collect SLp-1.2 checkpoints and release this campaign's paid resources.

Runs locally. Never forwards the account key over SSH. Resource deletion is
restricted to the IDs in the campaign receipt. A network volume is deleted only
after complete local artifact verification and successful CPU inference replay.
"""
from __future__ import annotations
import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
from slp12_runpod import ROOT, STATE, RECORD, run

NAME = 'slp12-joint-142m-r1'
REMOTE = '/workspace/SLp/results/' + NAME
LOCAL = ROOT / 'results' / NAME


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest.get('files', manifest).items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Unexpected manifest path')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Incomplete local artifact: ' + str(path))


def prune_checkpoints(folder, keep):
    """Bound backups for an explicitly configured run after a verified copy.

    Zero preserves the historical collector's retention. Never follow a symlink
    or prune a partial/unverified checkpoint. Training retains best weights
    separately from these resumable optimizer snapshots.
    """
    if keep == 0:
        return
    if keep < 2:
        raise ValueError('Retain at least two verified checkpoints')
    checkpoints = sorted(p for p in folder.iterdir() if re.fullmatch(r'checkpoint-\d{7}', p.name))
    if len(checkpoints) <= keep:
        return
    for path in checkpoints:
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Unexpected checkpoint path')
        verify(path)
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def main(ops='/workspace/slp12-ops', backup_seconds=3600, keep_checkpoints=0):
    record = json.loads(RECORD.read_text())
    connection = json.loads(run('ssh', 'info', record['pod_id']))
    ssh = ['ssh', '-i', connection['ssh_key']['path'], '-p', str(connection['port']),
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
           '-o', 'ServerAliveCountMax=2']
    host = 'root@' + connection['ip']
    LOCAL.mkdir(parents=True, exist_ok=True)
    last_backup = 0
    def delete(resource, identifier):
        try:
            print(run(resource, 'delete', identifier), flush=True)
        except RuntimeError as error:
            if 'not_found' not in str(error) and 'not found' not in str(error).lower():
                raise
    def sync(remote, local):
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-az', '--partial', '--timeout=120', '-e', shlex.join(ssh),
                        host + ':' + remote.rstrip('/') + '/', str(local) + '/'], check=True, timeout=1200,
                       stdout=subprocess.DEVNULL)
    while True:
        try:
            code = '''import json,pathlib
p=pathlib.Path(REMOTE)
result={}
for name in ['latest.json','completion.json','failure.json','export-complete.json']:
    f=p/name
    if f.exists(): result[name]=json.loads(f.read_text())
cleanup=pathlib.Path(OPS)/'cleanup-request.json'
if cleanup.exists(): result['cleanup']=json.loads(cleanup.read_text())
log=p/'training.jsonl'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-65536))
        for line in reversed(stream.read().splitlines()):
            try: result['last_record']=json.loads(line); break
            except ValueError: pass
print(json.dumps(result))
'''.replace('REMOTE', repr(REMOTE)).replace('OPS', repr(ops))
            response = subprocess.run(ssh + [host, 'python3 -'], input=code, text=True,
                                      capture_output=True, check=True, timeout=50)
            status = json.loads(response.stdout)
            status['collected_at'] = time.time()
            write(STATE / 'latest-status.json', status)
            latest = status.get('latest.json')
            final = 'export-complete.json' in status
            if latest and (time.time() - last_backup > backup_seconds or final or 'cleanup' in status):
                folder = latest['path']
                if '/' in folder or not folder.startswith('checkpoint-'):
                    raise ValueError('Unexpected checkpoint path')
                sync(REMOTE + '/' + folder, LOCAL / folder)
                verify(LOCAL / folder)
                prune_checkpoints(LOCAL, keep_checkpoints)
                last_backup = time.time()
                print(json.dumps({'event': 'checkpoint_backed_up', 'path': folder}), flush=True)
            if final:
                sync(REMOTE, LOCAL)
                bundle = LOCAL.with_name(NAME + '-bundle')
                sync(REMOTE + '-bundle', bundle)
                with (bundle / 'manifest.json').open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != status['export-complete.json']['manifest_sha256']:
                        raise ValueError('Collected bundle differs from the completed remote export')
                verify(bundle)
                for checkpoint in LOCAL.glob('checkpoint-*'):
                    verify(checkpoint)
                prune_checkpoints(LOCAL, keep_checkpoints)
                sync(ops, STATE / 'remote-ops')
                with (STATE / 'cpu-replay.log').open('w') as log:
                    subprocess.run([str(ROOT / '.venv/bin/python'), str(bundle / 'replay.py'),
                                    '--bundle', str(bundle)], stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=300)
                delete('pod', record['pod_id'])
                delete('network-volume', record['volume_id'])
                receipt = {'status': 'complete', 'completed_at': time.time(), 'bundle': str(bundle),
                           'pod_deleted': record['pod_id'], 'volume_deleted': record['volume_id'],
                           'cpu_replay': 'passed', 'allocation_estimated_usd': record['allocation_estimate_usd'],
                           'elapsed_gpu_cost_estimate_usd': (time.time() - record['created_at']) / 3600 * record['gpu_hourly_usd']}
                write(STATE / 'collection.json', receipt)
                current = json.loads(RECORD.read_text())
                current['termination_confirmed_at'] = time.time()
                current['volume_deleted_at'] = time.time()
                write(RECORD, current)
                try:
                    os.killpg(current['local_guard_pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                print(json.dumps(receipt), flush=True)
                return
            if 'cleanup' in status and not final:
                sync(REMOTE, LOCAL)
                sync(ops, STATE / 'remote-ops')
                delete('pod', record['pod_id'])
                write(STATE / 'collection.json', {'status': 'needs_attention', 'reason': status['cleanup'],
                      'pod_deleted': record['pod_id'], 'volume_preserved': record['volume_id']})
                return
        except Exception as error:
            print(json.dumps({'event': 'collector_retry', 'error_type': type(error).__name__, 'message': str(error)}), flush=True)
            if time.time() > record['terminate_at'] + 600:
                write(STATE / 'collection.json', {'status': 'needs_attention',
                      'reason': 'Deadline passed before complete artifact collection; inspect the persistent volume.'})
                return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--run-name', default=NAME)
    parser.add_argument('--ops', default='/workspace/slp12-ops')
    parser.add_argument('--backup-seconds', type=int, default=3600)
    parser.add_argument('--keep-checkpoints', type=int, default=0,
                        help='0 keeps all historical backups; 2 bounds a new run to two verified snapshots')
    args = parser.parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('Invalid run directory name')
    if args.backup_seconds < 60:
        raise ValueError('Backup interval must be at least one minute')
    if args.keep_checkpoints != 0 and args.keep_checkpoints < 2:
        raise ValueError('Retain at least two checkpoints, or zero for unlimited retention')
    STATE = args.state.resolve()
    if not STATE.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Campaign state must stay under this repository data directory')
    RECORD = STATE / 'campaign.json'
    NAME = args.run_name
    REMOTE = '/workspace/SLp/results/' + NAME
    LOCAL = ROOT / 'results' / NAME
    main(args.ops, args.backup_seconds, args.keep_checkpoints)
