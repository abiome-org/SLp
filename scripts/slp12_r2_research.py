"""Provision and supervise the user-authorized r2 research allocation.

Only metadata crosses back to this host. Reusable credentials remain local;
the pod receives expiring exact-file tickets from the separate broker.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

from slp12_runpod import run
from slp12_r2_runpod import IMAGE
from cloud_data import settings, call

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def detached(command, log):
    with log.open("a") as file:
        return subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def create(args):
    args.state.mkdir(parents=True, exist_ok=True)
    record_path = args.state / "campaign.json"
    if record_path.exists():
        raise ValueError("Existing allocation must be reconciled before creation")
    account = json.loads(run("user"))
    pods = json.loads(run("pod", "list"))
    pods = pods.get("pods", []) if isinstance(pods, dict) else pods
    name = "slp-r2-" + args.run_id
    if any(p.get("name") == name for p in pods):
        raise ValueError("Matching pod already exists")
    gpu = next(
        g
        for g in json.loads(run("gpu", "list", "--include-unavailable"))
        if g["gpuId"] == "NVIDIA GeForce RTX 4090"
    )
    rate = float(gpu["securePricePerHr"])
    balance = float(account["clientBalance"])
    if rate > 0.80 or balance < 55:
        raise ValueError("Fresh price/credit does not cover the approved campaign")
    hourly = rate + 80 * 0.20 / 730
    # $9.25 first-month archive allowance, $0.25 shutdown/control-plane buffer.
    allowance = 50 - 9.25 - 0.25
    now = time.time()
    record = {
        "schema": "slp.r2-research-allocation/v1",
        "run_id": args.run_id,
        "pod_name": name,
        "created_at": now,
        "terminate_at": now + int(allowance / hourly * 3600),
        "total_campaign_ceiling_usd": 50,
        "r2_first_month_reserve_usd": 9.25,
        "shutdown_reserve_usd": 0.25,
        "gpu_disk_allowance_usd": allowance,
        "gpu_hourly_usd": rate,
        "gpu_and_disk_hourly_estimate_usd": hourly,
        "container_disk_gb": 80,
        "image": IMAGE,
        "credit_stop_usd": 5,
        "initial_balance_usd": balance,
        "research_training_authorized": True,
        "outer_testing_authorized": True,
        "authorization": "User: go ahead and train, then test; monitor and intervene",
        "plan": str(args.plan),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "creation_needs_inspection": True,
    }
    save(record_path, record)
    pod = json.loads(
        run(
            "pod",
            "create",
            "--name",
            name,
            "--image",
            IMAGE,
            "--gpu-id",
            gpu["gpuId"],
            "--gpu-count",
            "1",
            "--cloud-type",
            "SECURE",
            "--data-center-ids",
            "EU-RO-1",
            "--container-disk-in-gb",
            "80",
            "--volume-in-gb",
            "0",
            "--ports",
            "22/tcp",
            "--ssh",
            "--min-cuda-version",
            "12.8",
        )
    )
    record.update(pod_id=pod["id"], creation_needs_inspection=False)
    save(record_path, record)
    guard = detached(
        [
            sys.executable,
            str(ROOT / "scripts/slp12_runpod.py"),
            "--state",
            str(args.state),
            "guard",
        ],
        args.state / "local-guard.log",
    )
    record["local_guard_pid"] = guard.pid
    save(record_path, record)
    print(json.dumps(record, indent=2))


def ssh(record):
    info = json.loads(run("ssh", "info", record["pod_id"]))
    return [
        "ssh",
        "-i",
        info["ssh_key"]["path"],
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-p",
        str(info["port"]),
        "root@" + info["ip"],
    ]


def remote(connection, code):
    p = subprocess.run(
        connection + ["python3 -c " + shlex.quote(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(p.stdout)


def bootstrap(args):
    record_path = args.state / "campaign.json"
    record = json.loads(record_path.read_text())
    if record.get("bootstrap_pid"):
        raise ValueError("Inspect existing bootstrap before retrying")
    connection = ssh(record)

    def send(path, body):
        code = (
            "import sys; from pathlib import Path; p=Path("
            + repr(path)
            + "); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read()); p.chmod(0o600)"
        )
        subprocess.run(
            connection + ["python3 -c " + shlex.quote(code)],
            input=body,
            check=True,
            timeout=90,
            capture_output=True,
        )

    send(
        "/workspace/slp-r2-pod-guard.py",
        (ROOT / "modules/slp-1-2-r2/pod_guard.py").read_bytes(),
    )
    guard = remote(
        connection,
        "import subprocess,json; f=open('/workspace/slp-r2-pod-guard.log','a'); "
        "p=subprocess.Popen(['python3','/workspace/slp-r2-pod-guard.py','--deadline',"
        + repr(str(record["terminate_at"]))
        + "],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({'pid':p.pid}))",
    )
    record["pod_guard_pid"] = guard["pid"]
    save(record_path, record)
    time.sleep(3)
    armed = remote(
        connection,
        "import json; from pathlib import Path; print(json.dumps(Path('/workspace/slp-r2-pod-guard.log').read_text()))",
    )
    if '"scoped_credential_verified": true' not in armed:
        raise RuntimeError("Independent in-pod guard did not arm")
    print(armed, flush=True)
    plan = json.loads(args.plan.read_text())
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for relative, sha in plan["source"].items():
            path = ROOT / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("Frozen source changed")
            archive.add(path, arcname=path.name)
        archive.add(args.plan, arcname="campaign/plan.json")
        for spec in plan["variants"].values():
            archive.add(
                args.plan.parent / spec["recipe"], arcname="campaign/" + spec["recipe"]
            )
        for fold in plan["protocols"]:
            archive.add(
                args.plan.parent / "protocols" / fold["file"],
                arcname="protocols/" + fold["file"],
            )
    subprocess.run(
        connection + ["mkdir -p /workspace/slp-r2 && tar xzf - -C /workspace/slp-r2"],
        input=payload.getvalue(),
        capture_output=True,
        check=True,
        timeout=90,
    )
    config = settings()
    config["SLP_STORAGE_URL"] = "https://slp-corpus-prep.potteryrage.workers.dev"
    tickets = call(
        config, "/tickets/readiness-r2-20260912-v3", "POST", max_bytes=4 * 1024**2
    )
    send("/workspace/slp-r2/transfers.json", json.dumps(tickets).encode())
    deadline = str(record["terminate_at"])
    wrapper = """import json,subprocess,time
from pathlib import Path
commands = COMMANDS
code=1
try:
    for command in commands:
        print(json.dumps({'event':'start','command':command,'at':time.time()}),flush=True)
        if 'campaign.py' in command[1] and '--execute-research' in command:
            with open('/workspace/slp-r2-campaign.log','a') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        else:
            subprocess.run(command,check=True)
    code=0
finally:
    p=Path('/workspace/slp-r2-exit.json'); t=p.with_suffix('.tmp')
    t.write_text(json.dumps({'code':code,'at':time.time()})); t.replace(p)
"""
    base = [
        "python3",
        "/workspace/slp-r2/campaign.py",
        "--plan",
        "/workspace/slp-r2/campaign/plan.json",
        "--root",
        "/workspace/slp-r2",
        "--run-id",
        args.run_id,
        "--deadline",
        deadline,
    ]
    commands = [
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "--require-hashes",
            "-r",
            "/workspace/slp-r2/requirements-linux-cu128.lock",
        ],
        [
            "python3",
            "-m",
            "pip",
            "uninstall",
            "--break-system-packages",
            "-y",
            "torchvision",
            "torchaudio",
        ],
        ["python3", "-m", "pip", "check"],
        [
            "python3",
            "/workspace/slp-r2/finish_readiness.py",
            "prepare",
            "--root",
            "/workspace/slp-r2",
            "--deadline",
            deadline,
        ],
        base,
        base + ["--execute-research", "--allow-outer-test"],
    ]
    send(
        "/workspace/slp-r2-research-wrapper.py",
        wrapper.replace("COMMANDS", repr(commands)).encode(),
    )
    launched = remote(
        connection,
        "import subprocess,json; f=open('/workspace/slp-r2-bootstrap.log','a'); p=subprocess.Popen(['python3','/workspace/slp-r2-research-wrapper.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({'pid':p.pid}))",
    )
    record["bootstrap_pid"] = launched["pid"]
    save(record_path, record)
    supervisor = detached(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            "--state",
            str(args.state),
            "--plan",
            str(args.plan),
            "--run-id",
            args.run_id,
        ],
        args.state / "supervisor.log",
    )
    record["supervisor_pid"] = supervisor.pid
    awake = detached(
        ["caffeinate", "-i", "-w", str(supervisor.pid)], args.state / "caffeinate.log"
    )
    record["caffeinate_pid"] = awake.pid
    save(record_path, record)
    print(
        json.dumps(
            {
                "bootstrap_pid": launched["pid"],
                "supervisor_pid": supervisor.pid,
                "source_archive_bytes": len(payload.getvalue()),
            }
        ),
        flush=True,
    )


STATUS_CODE = """
import json,os,time
from pathlib import Path
root=Path('/workspace/slp-r2')
def tail(p,n=12):
    if not p.exists(): return ''
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-12000))
        return '\\n'.join(f.read().decode(errors='replace').splitlines()[-n:])
logs=sorted(root.glob('runs/**/fit.log'),key=lambda p:p.stat().st_mtime)
j=root/'campaign-state/journal.json'
state=json.loads(j.read_text()) if j.exists() else {}
exitfile=Path('/workspace/slp-r2-exit.json')
metrics=[]
for p in root.glob('runs/**/inner-scores/metrics.json'):
    metrics.append({'path':str(p.relative_to(root)),'metrics':json.loads(p.read_text())})
print(json.dumps({'checked_at':time.time(),'runner_exit':json.loads(exitfile.read_text()) if exitfile.exists() else None,
 'bootstrap_tail':tail(Path('/workspace/slp-r2-bootstrap.log'),5),
 'runner_tail':tail(Path('/workspace/slp-r2-campaign.log'),6),
 'latest_fit':str(logs[-1].relative_to(root)) if logs else None,
 'fit_mtime':logs[-1].stat().st_mtime if logs else None,
 'fit_tail':tail(logs[-1],4) if logs else '',
 'stages_complete':len(state.get('stages',{})), 'folds_complete':len(state.get('folds',{})),
 'inner_metrics':metrics,'pending_transfers':[p.name for p in (root/'transfer-requests').glob('*.json') if not p.name.endswith('.ready.json')],
 'disk_free_bytes':os.statvfs('/workspace').f_bavail*os.statvfs('/workspace').f_frsize}))
"""


def supervise(args):
    record_path = args.state / "campaign.json"
    record = json.loads(record_path.read_text())
    connection = ssh(record)
    broker = None
    failure_since = None
    while time.time() < record["terminate_at"] - 300:
        if broker is None or broker.poll() is not None:
            broker = detached(
                [
                    sys.executable,
                    str(ROOT / "scripts/slp12_r2_broker.py"),
                    "--state",
                    str(args.state),
                    "--plan",
                    str(args.plan),
                    "--run-id",
                    args.run_id,
                    "--allow-outer-test",
                ],
                args.state / "broker.log",
            )
            record = json.loads(record_path.read_text())
            record["broker_pid"] = broker.pid
            save(record_path, record)
        try:
            status = remote(connection, STATUS_CODE)
            status["gpu_disk_elapsed_estimate_usd"] = (
                (time.time() - record["created_at"])
                / 3600
                * record["gpu_and_disk_hourly_estimate_usd"]
            )
            save(args.state / "latest-status.json", status)
            if status["runner_exit"] is not None:
                # Mirror the final journal before cleanup; only metadata comes home.
                journal = remote(
                    connection,
                    "import json; from pathlib import Path; p=Path('/workspace/slp-r2/campaign-state/journal.json'); print(p.read_text() if p.exists() else '{}')",
                )
                save(args.state / "campaign-journal.json", journal)
                complete = (
                    status["runner_exit"]["code"] == 0
                    and len(journal.get("folds", {})) == 30
                )
                if not complete and failure_since is None:
                    failure_since = time.time()
                if complete or time.time() - failure_since >= 1800:
                    print(run("pod", "delete", record["pod_id"]), flush=True)
                    pods = json.loads(run("pod", "list"))
                    pods = pods.get("pods", []) if isinstance(pods, dict) else pods
                    if any(p["id"] == record["pod_id"] for p in pods):
                        raise RuntimeError("Deletion not yet confirmed")
                    record = json.loads(record_path.read_text())
                    record.update(
                        termination_confirmed_at=time.time(),
                        termination_reason="research_complete"
                        if complete
                        else "failed_runner_unresolved_30_minutes",
                    )
                    save(record_path, record)
                    broker.terminate()
                    pid = record.get("local_guard_pid")
                    if pid:
                        command = subprocess.run(
                            ["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True,
                            text=True,
                        ).stdout
                        if (
                            str(ROOT / "scripts/slp12_runpod.py") in command
                            and str(args.state) in command
                            and command.rstrip().endswith(" guard")
                        ):
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                    return
                print(
                    json.dumps(
                        {
                            "event": "intervention_required",
                            "exit": status["runner_exit"],
                        }
                    ),
                    flush=True,
                )
            else:
                failure_since = None
        except Exception as error:
            print(
                json.dumps(
                    {"event": "supervisor_retry", "error": type(error).__name__}
                ),
                flush=True,
            )
        time.sleep(60)
    if broker is not None:
        broker.terminate()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("create", "bootstrap", "supervise"))
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    a.state, a.plan = a.state.resolve(), a.plan.resolve()
    if not a.state.is_relative_to(ROOT / "data"):
        raise ValueError(
            "Allocation state must remain in this repository's data directory"
        )
    if a.action == "create":
        create(a)
    elif a.action == "bootstrap":
        bootstrap(a)
    else:
        supervise(a)
