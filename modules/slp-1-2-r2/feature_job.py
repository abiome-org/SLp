"""Finite frozen-feature workload. There is no SLp optimizer in this job."""

import argparse
import json
from pathlib import Path
import time
import traceback

from cloud_io import fetch_shards, put_json, upload_directory
from extract_features import extract


def main(args):
    manifest = fetch_shards(
        "identity-r2-20260912-v5",
        "sequence-inputs-manifest.json",
        Path(args.root) / "sequence-inputs",
    )
    remaining = args.deadline - time.time() - 600
    if remaining < 300:
        raise RuntimeError("Insufficient guarded time for frozen extraction")
    options = argparse.Namespace(
        inputs=str(manifest),
        output=str(Path(args.root) / "features"),
        max_seconds=min(6000, int(remaining)),
        token_budget=4096,
        max_batch=16,
    )
    extract(options)
    receipt = upload_directory(options.output, "esm-r2-20260912-v1")
    result = {
        "state": "complete",
        "artifact": receipt,
        "frozen_encoder_only": True,
        "slp_optimizer_updates": 0,
    }
    put_json("esm-r2-20260912-v1", "complete.json", result)
    print(
        json.dumps(
            {
                "state": "complete",
                "artifact_job": receipt["job"],
                "files": len(receipt["files"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/slp-r2")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        report = {
            "state": "failed",
            "error": type(exc).__name__,
            "frames": [
                {"file": Path(f.filename).name, "line": f.lineno}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        }
        put_json("esm-r2-20260912-v1", "failed.json", report)
        print(json.dumps(report), flush=True)
        raise SystemExit(1) from None
