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
    next_credit_check = 0.
    reason = 'allocation_deadline'
    while time.time() < record['terminate_at']:
        if record.get('credit_stop_usd', 0) and time.time() >= next_credit_check:
            next_credit_check = time.time() + 300
            try:
                balance = float(json.loads(run('user'))['clientBalance'])
                if balance < record['credit_stop_usd']:
                    reason = 'low_account_credit'
                    print(json.dumps({'event': reason, 'balance_usd': balance}), flush=True)
                    break
            except Exception as error:
                print(json.dumps({'event': 'credit_check_retry', 'error': str(error)}), flush=True)
        time.sleep(max(0, min(30, record['terminate_at'] - time.time())))
    # Persistent retries: a transient control-plane outage must not turn a
    # one-shot timer into an indefinitely running GPU.
    while True:
        try:
            print(run('pod', 'delete', pod_id), flush=True)
            record['termination_reason'] = reason
            record['termination_confirmed_at'] = time.time()
            save(record)
            return
        except Exception as error:
            if 'not_found' in str(error) or 'not found' in str(error).lower():
                record['termination_confirmed_at'] = time.time()
                record['termination_reason'] = reason
                save(record)
                return
            print(json.dumps({'event': 'termination_retry', 'error': str(error)}), flush=True)
            time.sleep(30)


def provision(plan_path=None):
    existing = json.loads(RECORD.read_text()) if RECORD.exists() else None
    if existing and existing.get('pod_id'):
        raise RuntimeError('Campaign already recorded; inspect it before provisioning again')
    if existing and existing.get('creation_needs_inspection'):
        raise RuntimeError('Previous creation outcome is ambiguous; inspect provider resources before retrying')
    account = json.loads(run('user'))
    gpus = json.loads(run('gpu', 'list', '--include-unavailable'))
    plan = json.loads(Path(plan_path).read_text()) if plan_path else {}
    gpu_id = plan.get('gpu_id', 'NVIDIA GeForce RTX 4090')
    gpu = next(g for g in gpus if g['gpuId'] == gpu_id)
    price = gpu['securePricePerHr']
    # The allocation includes setup and export. Reserve the upper end of
    # storage pricing, even if the default tier costs less.
    hours, storage_gb, disk_gb = (plan.get(k, default) for k, default in
                                  (('allocation_hours', 26), ('network_volume_gb', 50), ('container_disk_gb', 30)))
    prior = plan.get('prior_spend_usd', 0.)
    if price is None or price > plan.get('max_gpu_hourly_usd', .80):
        raise RuntimeError('GPU is outside the recorded price ceiling')
    # Account credit is shared. Reserve for other running pods without changing
    # their state or charging their costs to this campaign's own authorization.
    other_pods = json.loads(run('pod', 'list'))
    if isinstance(other_pods, dict):
        other_pods = other_pods.get('pods', [])
    running = [p for p in other_pods if p.get('desiredStatus') == 'RUNNING']
    if any(p.get('costPerHr') is None for p in running):
        raise RuntimeError('Cannot reserve shared credit without other running pod prices')
    other_rate = sum(float(p['costPerHr']) for p in running)
    other_hours = plan.get('other_pods_reserve_hours', hours)
    budget = allocation_budget(hours, price, storage_gb, disk_gb, prior,
                               float(account['clientBalance']), other_rate, other_hours)
    estimate = budget['allocation_estimate_usd']
    record = {'schema': 'slp.campaign/v1', 'created_at': time.time(), 'max_spend_usd': 50,
              'reserve_usd': 5, 'gpu_hourly_usd': price, 'gpu_id': gpu_id, 'allocation_hours': hours,
              'allocation_estimate_usd': estimate, 'network_volume_gb': storage_gb,
              'container_disk_gb': disk_gb, 'data_center': plan.get('data_center', 'EU-RO-1'),
              'prior_spend_usd': prior, 'budget': budget, 'plan': plan, 'credit_stop_usd': 5,
              'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    volume = {'id': existing['volume_id']} if existing else json.loads(run('network-volume', 'create',
                            '--name', plan.get('volume_name', 'slp12-checkpoints-20260907'), '--size', str(storage_gb),
                            '--data-center-id', record['data_center']))
    record['volume_id'] = volume['id']
    save(record)
    # Record the clock before creation so image-pull time is covered.
    record['terminate_at'] = time.time() + hours * 3600
    save(record)
    try:
        pod = json.loads(run('pod', 'create', '--name', plan.get('pod_name', 'slp12-142m-20260907'),
                             '--cloud-type', 'SECURE', '--gpu-id', gpu['gpuId'], '--gpu-count', '1',
                             '--data-center-ids', record['data_center'],
                             '--template-id', 'runpod-torch-v280', '--network-volume-id', volume['id'],
                             '--container-disk-in-gb', str(disk_gb), '--volume-mount-path', '/workspace',
                             '--ports', '22/tcp', '--min-cuda-version', '12.8'))
    except Exception:
        # A timeout can be ambiguous; leave the recorded volume for inspection
        # rather than retrying creation and accidentally allocating twice.
        record['creation_needs_inspection'] = True
        save(record)
        raise
    record['pod_id'] = pod['id']
    record['actual_pod_cost_per_hour'] = pod.get('costPerHr')
    save(record)
    with (STATE / 'local-guard.log').open('a') as log:
        proc = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, str(Path(__file__).resolve()),
                                 '--state', str(STATE), 'guard'],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['local_guard_pid'] = proc.pid
    save(record)
    print(json.dumps(record, indent=2))


def allocation_budget(hours, gpu_rate, storage_gb, disk_gb, prior, balance, other_rate=0., other_hours=None):
    """Enforce the original total cap and available shared credit before creating resources."""
    import math
    other_hours = hours if other_hours is None else other_hours
    if any(not math.isfinite(v) or v < 0 for v in (hours, gpu_rate, storage_gb, disk_gb, prior, balance, other_rate, other_hours)) or hours == 0:
        raise ValueError('Invalid allocation inputs')
    estimate = hours * (gpu_rate + (storage_gb * .14 + disk_gb * .10) / 720)
    campaign_total = prior + estimate + 5
    account_required = estimate + other_hours * other_rate + 5
    if campaign_total > 50:
        raise RuntimeError(f'Allocation plus prior spend and reserve exceeds the $50 campaign cap: ${campaign_total:.2f}')
    if account_required > balance:
        raise RuntimeError(f'Insufficient shared credit: ${account_required:.2f} needed, ${balance:.2f} available; includes other running pods')
    return dict(allocation_estimate_usd=estimate, prior_spend_usd=prior, reserve_usd=5,
                campaign_total_with_reserve_usd=campaign_total, other_pods_hourly_usd=other_rate,
                other_pods_reserve_hours=other_hours,
                account_required_usd=account_required, account_balance_at_check_usd=balance)


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
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    STATE = args.state.resolve()
    if not STATE.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Campaign state must stay under this repository data directory')
    RECORD = STATE / 'campaign.json'
    if args.operation == 'inspect':
        inspect()
    elif args.operation == 'pod-help':
        print(run('pod', 'create', '--help'))
    elif args.operation == 'ssh-help':
        print(run('ssh', '--help'))
    elif args.operation == 'volumes':
        print(run('network-volume', 'list'))
    elif args.operation == 'provision':
        provision(args.plan)
    elif args.operation == 'guard':
        guard()
    else:
        status()
