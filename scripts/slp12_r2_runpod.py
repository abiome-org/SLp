"""Own-resource-only, bounded RunPod preparation; no research launch operation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from slp12_runpod import run

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/slp12-r2-readiness"
RECORD = STATE / "campaign.json"
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
NAME = "slp-r2-features-20260912"


def save(value):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = RECORD.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(RECORD)


def create(*, max_seconds=7200, prior_usd=0.0):
    if not 900 <= max_seconds <= 7200 or not 0 <= prior_usd < 5:
        raise ValueError("Require bounded preparation within the existing $5 ceiling")
    if RECORD.exists():
        raise ValueError("Preparation already recorded; inspect before retrying")
    pods = json.loads(run("pod", "list"))
    if isinstance(pods, dict):
        pods = pods.get("pods", [])
    if any(p.get("name") == NAME for p in pods):
        raise ValueError("Matching pod exists; creation outcome needs reconciliation")
    account = json.loads(run("user"))
    gpu = next(
        g
        for g in json.loads(run("gpu", "list", "--include-unavailable"))
        if g["gpuId"] == "NVIDIA GeForce RTX 4090"
    )
    rate = float(gpu["securePricePerHr"])
    upper = max_seconds / 3600 * (rate + 80 * 0.20 / 730)
    if rate > 0.80 or float(account["clientBalance"]) < 10 or prior_usd + upper > 5:
        raise ValueError("Preparation price/credit bound exceeded")
    record = {
        "schema": "slp.r2-preparation-pod/v1",
        "purpose": "frozen features and disposable readiness checks only",
        "created_at": time.time(),
        "terminate_at": time.time() + max_seconds,
        "max_seconds": max_seconds,
        "gpu_hourly_usd": rate,
        "container_disk_gb": 80,
        "gpu_and_disk_upper_estimate_usd": upper,
        "prior_preparation_usd": prior_usd,
        "initial_preparation_ceiling_usd": 5,
        "image": IMAGE,
        "credit_stop_usd": 5,
        "creation_needs_inspection": True,
        "research_training_authorized": False,
    }
    save(record)
    pod = json.loads(
        run(
            "pod",
            "create",
            "--name",
            NAME,
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
    record["pod_id"] = pod["id"]
    record["creation_needs_inspection"] = False
    save(record)
    with (STATE / "local-guard.log").open("a") as log:
        guard = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/slp12_runpod.py"),
                "--state",
                str(STATE),
                "guard",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record["local_guard_pid"] = guard.pid
    save(record)
    print(json.dumps(record, indent=2))


def status():
    record = json.loads(RECORD.read_text())
    pod = json.loads(run("pod", "get", record["pod_id"]))
    print(
        json.dumps(
            {
                k: pod.get(k)
                for k in (
                    "id",
                    "name",
                    "desiredStatus",
                    "costPerHr",
                    "publicIp",
                    "portMappings",
                    "runtime",
                )
            },
            indent=2,
        )
    )
    print(run("ssh", "info", record["pod_id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("create-features", "create-readiness", "status", "terminate")
    )
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--prior-preparation-usd", type=float, default=0.0)
    args = parser.parse_args()
    STATE = args.state.resolve()
    if not STATE.is_relative_to(ROOT / "data"):
        raise ValueError(
            "Preparation state must stay in this repository's data directory"
        )
    RECORD = STATE / "campaign.json"
    NAME = "slp-r2-" + STATE.name.removeprefix("slp12-r2-")
    if args.action in ("create-features", "create-readiness"):
        create(max_seconds=args.max_seconds, prior_usd=args.prior_preparation_usd)
    elif args.action == "status":
        status()
    else:
        record = json.loads(RECORD.read_text())
        print(run("pod", "delete", record["pod_id"]))
        record["termination_confirmed_at"] = time.time()
        record["termination_reason"] = "preparation_complete_or_stopped"
        save(record)
