"""Publish or restore one immutable optimizer checkpoint using exact R2 links.

Run beside the captured model source on the pod (or set --module-directory).
This never starts an optimizer or provisions compute. The caller must supply
short-lived exact-object capabilities through SLP_TRANSFER_TICKETS.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main(args):
    sys.path.insert(0, str(Path(args.module_directory).resolve()))
    import torch
    from cloud_io import upload_directory, download_directory
    from data import digest
    from train import identity_digest

    def verify(path):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state.get("schema") != "slp.r2-checkpoint/v1" or state[
            "identity_sha256"
        ] != identity_digest(state["identity"]):
            raise ValueError("Optimizer checkpoint identity mismatch")
        if (
            not {"model", "optimizer", "sampler", "torch_rng", "cuda_rng", "python_rng"}
            <= state.keys()
        ):
            raise ValueError("Incomplete resumable state")
        return {
            "update": state["update"],
            "identity_sha256": state["identity_sha256"],
            "sha256": digest(path),
            "bytes": Path(path).stat().st_size,
        }

    if args.action == "publish":
        path = Path(args.path).resolve()
        # A hard link captures the atomically replaced checkpoint inode even
        # while a later checkpoint is being written by the fitting process.
        with tempfile.TemporaryDirectory(
            prefix=".checkpoint-publish-", dir=path.parent
        ) as temporary:
            snapshot = Path(temporary) / "checkpoint.pt"
            os.link(path, snapshot)
            receipt = verify(snapshot)
            manifest = upload_directory(temporary, args.job)
            if manifest["files"]["checkpoint.pt"]["sha256"] != receipt["sha256"]:
                raise ValueError("Checkpoint changed during publication")
    else:
        directory = Path(args.path)
        if directory.exists():
            raise ValueError("Restore requires a new directory")
        manifest = download_directory(args.job, directory)
        receipt = verify(directory / "checkpoint.pt")
        if manifest["files"]["checkpoint.pt"]["sha256"] != receipt["sha256"]:
            raise ValueError("Restored checkpoint hash mismatch")
    print(json.dumps({"action": args.action, "job": args.job, **receipt}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("publish", "restore"))
    p.add_argument("--path", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--module-directory", default="/workspace/slp-r2")
    main(p.parse_args())
