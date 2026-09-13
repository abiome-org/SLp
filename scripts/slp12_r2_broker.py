"""Local exact-file ticket broker. Account credentials never enter the GPU pod."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request

from cloud_data import settings
from slp12_runpod import run

URL = "https://slp-corpus-prep.potteryrage.workers.dev"


def allowed_labels(plan):
    labels = {"source"}
    for job in plan["inner_jobs"]:
        prefix = job["fold"] + "-" + job["candidate"]
        labels.add(prefix + ("-pretrain" if job["pretraining_updates"] else "-adapt"))
        for probe in job["probes"]:
            argv = probe["scoring_command"]
            candidate = argv[argv.index("--candidate") + 1]
            labels.update(
                {
                    job["fold"] + "-" + candidate + "-adapt",
                    job["fold"] + "-" + candidate + "-inner-scores",
                }
            )
    for fold in plan["protocols"]:
        labels.update(
            fold["name"] + suffix
            for suffix in (
                "-outer-pretrain",
                "-outer-adapt",
                "-outer-scores",
                "-outer-bundle",
                "-selection",
                "-input-train",
                "-input-valid",
                "-input-test",
            )
        )
    return {label.replace("_", "-") for label in labels}


def permissions_for(request, plan, run_id, *, allow_outer_test=False):
    prefix = "campaign-r2-" + run_id + "-"
    job = request.get("job", "")
    label = job[len(prefix) :]
    base_label = re.sub(r"-u\d{6}$", "", label)
    if (
        not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", run_id)
        or not job.startswith(prefix)
        or base_label not in allowed_labels(plan)
    ):
        raise ValueError("Request is not a declared campaign operation")
    if request.get("kind") == "partition":
        fold = next(f for f in plan["protocols"] if f["name"] == request["fold"])
        partition = request["partition"]
        if label != fold["name"].replace("_", "-") + "-input-" + partition:
            raise ValueError("Partition request does not match its declared operation")
        if partition == "test" and not allow_outer_test:
            raise ValueError("Official test access is not authorized")
        spec = fold["partition_manifests"][partition]
        return [
            {
                "job": fold["job"],
                "name": spec["name"],
                "method": "GET",
                "max_bytes": 4 * 1024 * 1024,
            },
            *[
                {
                    "job": fold["job"],
                    "name": s["name"],
                    "method": "GET",
                    "max_bytes": s["bytes"],
                }
                for s in spec["manifest"]["shards"]
            ],
        ]
    permissions = []
    method = "GET" if request.get("kind") == "restore" else "PUT"
    files = request.get("files", {})
    if (
        not files
        or len(files) > 128
        or sum(v["bytes"] for v in files.values()) > 8 * 1024**3
    ):
        raise ValueError("Invalid bounded publication")
    for name, spec in files.items():
        if (
            not re.fullmatch(r"[a-zA-Z0-9_.-]+", name)
            or name.startswith(".")
            or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])
        ):
            raise ValueError("Invalid publication filename/hash")
        if not isinstance(spec["bytes"], int) or not 0 < spec["bytes"] <= 2 * 1024**3:
            raise ValueError("Invalid file size")
        for part in range(math.ceil(spec["bytes"] / (16 * 1024**2))):
            permissions.append(
                {
                    "job": job,
                    "name": name.replace(".", "-") + f"-part{part:05d}.bin.gz",
                    "method": method,
                    "max_bytes": 17 * 1024**2,
                }
            )
    permissions.extend(
        {
            "job": job,
            "name": "artifact.json",
            "method": method,
            "max_bytes": 4 * 1024**2,
        }
        for method in (("GET",) if request.get("kind") == "restore" else ("GET", "PUT"))
    )
    return permissions


def mint(permissions):
    config = settings()
    request = urllib.request.Request(
        URL + "/tickets/campaign",
        method="POST",
        data=json.dumps({"permissions": permissions}).encode(),
        headers={
            "Authorization": "Bearer " + config["SLP_STORAGE_TOKEN"],
            "Content-Type": "application/json",
            "User-Agent": "SLp-Cloud-Storage/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(2 * 1024**2 + 1)
    if len(body) > 2 * 1024**2:
        raise ValueError("Oversized ticket response")
    return json.loads(body)


def main(args):
    record = json.loads(Path(args.state, "campaign.json").read_text())
    plan_bytes = Path(args.plan).read_bytes()
    plan, plan_sha = json.loads(plan_bytes), hashlib.sha256(plan_bytes).hexdigest()
    info = json.loads(run("ssh", "info", record["pod_id"]))
    ssh = [
        "ssh",
        "-i",
        info["ssh_key"]["path"],
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(info["port"]),
        "root@" + info["ip"],
    ]
    queue = "/workspace/slp-r2/transfer-requests"

    def remote(code, body=None):
        result = subprocess.run(
            ssh + ["python3 -c " + shlex.quote(code)],
            input=body,
            capture_output=True,
            timeout=60,
            check=True,
        )
        return result.stdout

    issued = {}
    while time.time() < record["terminate_at"] - 300:
        raw = remote(
            "import json; from pathlib import Path; p=Path("
            + repr(queue)
            + "); print(json.dumps([json.loads(f.read_text()) for f in p.glob('*.json') if not f.name.endswith('.ready.json') and f.stat().st_size<262144]))"
        )
        for request in json.loads(raw):
            if request.get("plan_sha256") != plan_sha:
                raise ValueError("GPU requested tickets for a different plan")
            fingerprint = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest()
            if time.time() - issued.get(fingerprint, 0) < 1800:
                continue
            permissions = permissions_for(
                request, plan, args.run_id, allow_outer_test=args.allow_outer_test
            )
            response = {
                **mint(permissions),
                "job": request["job"],
                "files": request.get("files"),
                "partition": request.get("partition"),
                "kind": request.get("kind"),
            }
            ready = queue + "/" + request["job"] + ".ready.json"
            remote(
                "import sys,os; from pathlib import Path; p=Path("
                + repr(ready)
                + "); t=p.with_suffix('.tmp'); t.write_bytes(sys.stdin.buffer.read()); t.chmod(0o600); t.replace(p)",
                json.dumps(response).encode(),
            )
            issued[fingerprint] = time.time()
        journal = remote(
            "from pathlib import Path; p=Path('/workspace/slp-r2/campaign-state/journal.json'); print(p.read_text() if p.exists() and p.stat().st_size<32*1024*1024 else '{}')"
        )
        if json.loads(journal).get("plan_sha256") == plan_sha:
            path = Path(args.state) / "campaign-journal.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(journal)
            temporary.replace(path)
        if args.once:
            return
        time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--allow-outer-test", action="store_true")
    p.add_argument("--once", action="store_true")
    main(p.parse_args())
