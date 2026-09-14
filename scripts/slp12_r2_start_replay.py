"""Start one bounded bundle replay once the first benchmark fold is published."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from slp12_r2_broker import mint, permissions_for
from slp12_r2_research import ROOT, remote, save, ssh


def main(args):
    state = args.state.resolve()
    if not state.is_relative_to(ROOT / "data"):
        raise ValueError("Use the existing repository allocation state")
    job_path = state / "bundle-replay-job.json"
    if job_path.exists():
        raise ValueError("A replay is already recorded; inspect it before retrying")
    record = json.loads((state / "campaign.json").read_text())
    plan_path = Path(record["plan"])
    plan = json.loads(plan_path.read_text())
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan_sha != record["plan_sha256"]:
        raise ValueError("Active plan identity changed")
    root = record["active_remote_root"]
    fold = plan["protocols"][0]["name"]
    connection = ssh(record)
    code = "root=" + repr(root) + "\nfold=" + repr(fold) + "\n" + """
import json
from pathlib import Path
p=Path(root)/'campaign-state/journal.json'
j=json.loads(p.read_text()) if p.exists() else {}
completed=j.get('folds',{}).get(fold)
print(json.dumps({'fold':fold,'plan_sha256':j.get('plan_sha256'),'families':
 {name:{'bundle':value['outputs']['bundle'],'scores':value['outputs']['scores']}
  for name,value in completed['families'].items()} if completed else None}))
"""
    request = remote(connection, code)
    if request["plan_sha256"] != plan_sha:
        raise ValueError("Remote campaign identity differs")
    if request["families"] is None:
        print(json.dumps({"state": "waiting-for-first-completed-fold", "fold": fold}))
        return
    deadline = min(time.time() + 1180, record["terminate_at"] - 600)
    if deadline - time.time() < 900:
        raise ValueError("Insufficient time for bounded replay before shutdown")
    permissions = []
    for artifacts in request["families"].values():
        for receipt in artifacts.values():
            permissions.extend(permissions_for({"kind": "restore", **receipt}, plan,
                                               record["active_run_id"]))
    driver = ROOT / "scripts/slp12_r2_replay_bundles.py"
    request_path = state / "bundle-replay-request.json"
    save(request_path, request)
    prepared = {"prepared_at": time.time(), "deadline": deadline,
                "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
                "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "fold": fold, "remote_output": "/workspace/slp-r2-bundle-replay",
                "permission_count": len(permissions), "state": "preparing"}
    save(job_path, prepared)

    def send(destination, body):
        code = "import sys;from pathlib import Path;p=Path(" + repr(destination) + ");" + (
            "b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;"
            "p.write_bytes(b);p.chmod(0o600)"
        )
        subprocess.run(connection + ["python3 -c " + shlex.quote(code)], input=body,
                       capture_output=True, check=True, timeout=60)

    send("/workspace/slp-r2-replay-bundles.py", driver.read_bytes())
    send("/workspace/slp-r2-bundle-replay-request.json", request_path.read_bytes())
    send("/workspace/slp-r2-bundle-replay-tickets.json", json.dumps(mint(permissions)).encode())
    command = ["timeout", "1200", "python3", "/workspace/slp-r2-replay-bundles.py",
               "--root", root, "--receipt", "/workspace/slp-r2-bundle-replay-request.json",
               "--tickets", "/workspace/slp-r2-bundle-replay-tickets.json",
               "--output", prepared["remote_output"], "--transport", "/workspace/slp-r2-transport.py",
               "--deadline", str(deadline)]
    launched = remote(connection, "command=" + repr(command) + "\n" + """
import json,subprocess,time
from pathlib import Path
claim=Path('/workspace/slp-r2-bundle-replay-launch.json')
with claim.open('x') as f:json.dump({'state':'launching','at':time.time()},f)
log=open('/workspace/slp-r2-bundle-replay.log','x')
p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'pid':p.pid,'at':time.time()};claim.write_text(json.dumps(result));print(json.dumps(result))
""")
    prepared.update(state="launched", launch=launched, command=command)
    save(job_path, prepared)
    print(json.dumps({"state": "launched", "fold": fold, "launch": launched, "deadline": deadline}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    main(parser.parse_args())
