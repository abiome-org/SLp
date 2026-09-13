"""Metadata-only controller for the fixed, bounded Cloudflare prep job."""

import argparse
import json
from pathlib import Path
import time

from cloud_data import call, settings

ROOT = Path(__file__).resolve().parents[1]
URL = "https://slp-corpus-prep.potteryrage.workers.dev"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "run", "reports", "batch"))
    parser.add_argument("--job", help="One exact declared preparation job")
    parser.add_argument(
        "--jobs", nargs="+", help="Exact declared jobs, executed sequentially"
    )
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    config = settings()
    config["SLP_STORAGE_URL"] = URL
    if args.command == "batch":
        declared = json.loads((ROOT / "storage/prep/prep-jobs.json").read_text())
        if (
            not args.jobs
            or any(j not in declared for j in args.jobs)
            or not 0 < args.max_seconds <= 7200
        ):
            raise ValueError(
                "Require exact declared jobs and a finite two-hour controller limit"
            )
        deadline = time.monotonic() + args.max_seconds
        for job in args.jobs:
            while time.monotonic() < deadline:
                status = call(config, "/status/" + job)
                names = {Path(o["key"]).name for o in status["objects"]}
                if "failed.json" in names or "runner-failed.json" in names:
                    raise RuntimeError(
                        "Preparation failed; inspect immutable receipts for " + job
                    )
                if "complete.json" in names:
                    print(json.dumps({"job": job, "state": "complete"}), flush=True)
                    break
                if "claim.json" not in names:
                    try:
                        result = call(config, "/run/" + job, "POST")
                    except RuntimeError as exc:
                        if "image rollout is not ready" not in str(exc):
                            raise
                        result = {"state": "waiting-for-image-rollout"}
                    print(json.dumps({"job": job, **result}), flush=True)
                time.sleep(15)
            else:
                raise TimeoutError("Bounded preparation controller deadline reached")
        return
    if args.command == "run":
        print(
            json.dumps(
                call(config, "/run" + ("/" + args.job if args.job else ""), "POST")
            )
        )
        return
    status = call(config, "/status" + ("/" + args.job if args.job else ""))
    if args.command == "status":
        print(json.dumps(status, indent=2))
        return
    directory = ROOT / "results" / status["job"]
    directory.mkdir(parents=True, exist_ok=True)
    for obj in status["objects"]:
        name = Path(obj["key"]).name
        if (
            name == "claim.json"
            or not name.endswith(".json")
            or obj["bytes"] > 4 * 1024 * 1024
            or name in ("basal.json", "aliases.json", "musl-roster.json")
        ):
            continue
        # This API exposes only bounded JSON metadata, never source payloads.
        value = call(
            config, "/outputs/" + status["job"] + "/" + name, max_bytes=4 * 1024 * 1024
        )
        (directory / name).write_text(json.dumps(value, indent=2) + "\n")
    (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps(
            {"report_directory": str(directory), "objects": len(status["objects"])}
        )
    )


if __name__ == "__main__":
    main()
