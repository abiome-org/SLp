"""Bounded SLp-1.2 RunPod campaign. Never print or forward the account key.

Provisioning creates only this campaign's volume and one GPU. A separate local
watchdog deletes that exact pod at its recorded deadline. The pod also runs
modules/slp-1-2/pod_guard.py using its built-in scoped credential.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'data/slp12-campaign'
RECORD = STATE / 'campaign.json'


def environment():
    env = os.environ.copy()
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, raw = line.removeprefix('export ').split('=', 1)
            if name.strip() not in {'RUNPOD_API_KEY'}:
                continue
            values = shlex.split(raw, comments=True)
            if len(values) > 1:
                raise ValueError('Invalid .env value')
            env[name.strip()] = values[0] if values else ''
    if not env.get('RUNPOD_API_KEY'):
        raise RuntimeError('Add RUNPOD_API_KEY to the local .env file')
    return env


def run(*args):
    env = environment()
    proc = subprocess.run(['runpodctl', *args], env=env, capture_output=True, text=True, timeout=90)
    if proc.returncode:
        safe = proc.stderr.replace(env['RUNPOD_API_KEY'], '[REDACTED]')
        raise RuntimeError(safe[:3000])
    return proc.stdout


def inspect():
    print('CLI:', run('version').strip())
    print('GPU CATALOG:', run('gpu', 'list', '--include-unavailable'))
    print('PYTORCH TEMPLATES:', run('template', 'search', 'pytorch'))


def save(record):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = RECORD.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECORD)


def guard():
    record = json.loads(RECORD.read_text())
    pod_id = record['pod_id']
    print(json.dumps({'event': 'guard_started', 'pod_id': pod_id, 'deadline': record['terminate_at']}), flush=True)
    while time.time() < record['terminate_at']:
        time.sleep(max(0, min(30, record['terminate_at'] - time.time())))
    # Persistent retries: a transient control-plane outage must not turn a
    # one-shot timer into an indefinitely running GPU.
    while True:
        try:
            print(run('pod', 'delete', pod_id), flush=True)
            record['termination_confirmed_at'] = time.time()
            save(record)
            return
        except Exception as error:
            if 'not_found' in str(error) or 'not found' in str(error).lower():
                record['termination_confirmed_at'] = time.time()
                save(record)
                return
            print(json.dumps({'event': 'termination_retry', 'error': str(error)}), flush=True)
            time.sleep(30)


def provision():
    existing = json.loads(RECORD.read_text()) if RECORD.exists() else None
    if existing and existing.get('pod_id'):
        raise RuntimeError('Campaign already recorded; inspect it before provisioning again')
    account = json.loads(run('user'))
    gpus = json.loads(run('gpu', 'list', '--include-unavailable'))
    gpu = next(g for g in gpus if g['gpuId'] == 'NVIDIA GeForce RTX 4090')
    price = gpu['securePricePerHr']
    # 26 hours includes setup, a <=24-hour trainer, and export time. Reserve
    # the upper end of storage pricing, even if the default tier costs less.
    hours, storage_gb, disk_gb = 26, 50, 30
    estimate = hours * (price + (storage_gb * .14 + disk_gb * .10) / 720)
    if price > .80 or estimate > min(45, account['clientBalance'] - 5):
        raise RuntimeError(f'Current price/balance cannot support this allocation: ${estimate:.2f}')
    record = {'schema': 'slp.campaign/v1', 'created_at': time.time(), 'max_spend_usd': 50,
              'reserve_usd': 5, 'gpu_hourly_usd': price, 'allocation_hours': hours,
              'allocation_estimate_usd': estimate, 'network_volume_gb': storage_gb,
              'container_disk_gb': disk_gb, 'data_center': 'EU-RO-1',
              'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    volume = {'id': existing['volume_id']} if existing else json.loads(run('network-volume', 'create',
                            '--name', 'slp12-checkpoints-20260907', '--size', str(storage_gb),
                            '--data-center-id', record['data_center']))
    record['volume_id'] = volume['id']
    save(record)
    # Record the clock before creation so image-pull time is covered.
    record['terminate_at'] = time.time() + hours * 3600
    save(record)
    pod = json.loads(run('pod', 'create', '--name', 'slp12-142m-20260907',
                         '--cloud-type', 'SECURE', '--gpu-id', gpu['gpuId'], '--gpu-count', '1',
                         '--data-center-ids', record['data_center'],
                         '--template-id', 'runpod-torch-v280', '--network-volume-id', volume['id'],
                         '--container-disk-in-gb', str(disk_gb), '--volume-mount-path', '/workspace',
                         '--ports', '22/tcp', '--min-cuda-version', '12.8'))
    record['pod_id'] = pod['id']
    record['actual_pod_cost_per_hour'] = pod.get('costPerHr')
    save(record)
    with (STATE / 'local-guard.log').open('a') as log:
        proc = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, str(Path(__file__).resolve()), 'guard'],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['local_guard_pid'] = proc.pid
    save(record)
    print(json.dumps(record, indent=2))


def status():
    record = json.loads(RECORD.read_text())
    print(json.dumps(record, indent=2))
    pod = json.loads(run('pod', 'get', record['pod_id']))
    # Provider pod objects contain injected environment variables; print only
    # the operational fields needed here.
    print(json.dumps({k: pod.get(k) for k in ('id', 'name', 'desiredStatus', 'runtimeStatus',
                     'runtimeReason', 'costPerHr', 'publicIp', 'portMappings', 'machineId')}))
    print(run('ssh', 'info', record['pod_id']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inspect', 'pod-help', 'ssh-help', 'volumes', 'provision', 'guard', 'status'])
    args = parser.parse_args()
    if args.operation == 'inspect':
        inspect()
    elif args.operation == 'pod-help':
        print(run('pod', 'create', '--help'))
    elif args.operation == 'ssh-help':
        print(run('ssh', '--help'))
    elif args.operation == 'volumes':
        print(run('network-volume', 'list'))
    elif args.operation == 'provision':
        provision()
    elif args.operation == 'guard':
        guard()
    else:
        status()
