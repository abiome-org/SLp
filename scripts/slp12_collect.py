"""Collect SLp-1.2 checkpoints and release this campaign's paid resources.

Runs locally. Never forwards the account key over SSH. Resource deletion is
restricted to the IDs in the campaign receipt. A network volume is deleted only
after complete local artifact verification and successful CPU inference replay.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shlex
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


def main():
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
cleanup=pathlib.Path('/workspace/slp12-ops/cleanup-request.json')
if cleanup.exists(): result['cleanup']=json.loads(cleanup.read_text())
log=p/'training.jsonl'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-65536))
        for line in reversed(stream.read().splitlines()):
            try: result['last_record']=json.loads(line); break
            except ValueError: pass
print(json.dumps(result))
'''.replace('REMOTE', repr(REMOTE))
            response = subprocess.run(ssh + [host, 'python3 -'], input=code, text=True,
                                      capture_output=True, check=True, timeout=50)
            status = json.loads(response.stdout)
            status['collected_at'] = time.time()
            write(STATE / 'latest-status.json', status)
            latest = status.get('latest.json')
            final = 'export-complete.json' in status
            if latest and (time.time() - last_backup > 3600 or final or 'cleanup' in status):
                folder = latest['path']
                if '/' in folder or not folder.startswith('checkpoint-'):
                    raise ValueError('Unexpected checkpoint path')
                sync(REMOTE + '/' + folder, LOCAL / folder)
                verify(LOCAL / folder)
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
                sync('/workspace/slp12-ops', STATE / 'remote-ops')
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
                sync('/workspace/slp12-ops', STATE / 'remote-ops')
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
    main()
