"""Bounded parallel artifact transport for the frozen r2 campaign controller.

This adjunct runs outside the captured model directory. Fitting/scoring continue
in their original subprocesses with their original source and checkpoint identity.
The adjunct source and activation receipt are separately archived in private R2.
Artifact bytes, part names, compression, hashes and manifest format are unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


def upload_directory(directory, job):
    import cloud_io

    def upload(chunk, name):
        body = gzip.compress(chunk, compresslevel=1, mtime=0)
        with cloud_io.request(job, name, body) as response:
            response.read(4096)
        return {
            "name": name,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }

    files = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for file in sorted(Path(directory).iterdir()):
            if not file.is_file() or not re.fullmatch(r"[a-zA-Z0-9_.-]+", file.name):
                raise ValueError("Only flat artifact directories may be published")
            sha, size, parts, pending = hashlib.sha256(), 0, [], []
            with file.open("rb") as stream:
                index = 0
                while chunk := stream.read(16 * 1024 * 1024):
                    sha.update(chunk)
                    size += len(chunk)
                    name = file.name.replace(".", "-") + f"-part{index:05d}.bin.gz"
                    pending.append(pool.submit(upload, chunk, name))
                    index += 1
                    # Bound resident raw/compressed payloads; retain original part order.
                    if len(pending) == 4:
                        parts.append(pending.pop(0).result())
                parts.extend(future.result() for future in pending)
            files[file.name] = {
                "sha256": sha.hexdigest(),
                "bytes": size,
                "parts": parts,
            }
    manifest = {"schema": "slp.artifact-parts/v1", "job": job, "files": files}
    cloud_io.put_json(job, "artifact.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--root", default="/workspace/slp-r2")
    p.add_argument("--run-id", required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--execute-research", action="store_true")
    p.add_argument("--allow-outer-test", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, args.root)
    import campaign

    campaign.upload_directory = upload_directory
    print(
        json.dumps(
            {
                "event": "parallel_transport",
                "workers": 4,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        ),
        flush=True,
    )
    print(json.dumps(campaign.Runner(args).run()), flush=True)
