"""Immutable, checksum-verified R2 transport via the SLp-only Worker API."""

import gzip
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

URL = "https://slp-corpus-prep.potteryrage.workers.dev"


def request(job, name, body=None):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", job) or not re.fullmatch(
        r"[a-zA-Z0-9_-]+\.(?:json|jsonl\.gz|bin\.gz)", name
    ):
        raise ValueError("Invalid artifact address")
    method = "PUT" if body is not None else "GET"
    tickets = json.loads(
        Path(
            os.environ.get("SLP_TRANSFER_TICKETS", "/workspace/slp-r2/transfers.json")
        ).read_text()
    )
    permission = next(
        (
            p
            for p in tickets["tickets"]
            if p["job"] == job and p["name"] == name and p["method"] == method
        ),
        None,
    )
    if permission is None:
        raise ValueError("No exact-object transfer capability for this operation")
    if not permission["url"].startswith(URL + "/transfer/"):
        raise ValueError("Transfer URL is outside the SLp artifact service")
    headers = {"User-Agent": "SLp-Cloud-Storage/1.0"}
    if body is not None:
        headers["X-Content-SHA256"] = hashlib.sha256(body).hexdigest()
    return urllib.request.urlopen(
        urllib.request.Request(
            permission["url"], data=body, headers=headers, method=method
        ),
        timeout=180,
    )


def get_json(job, name):
    with request(job, name) as response:
        return json.load(response)


def put_json(job, name, value):
    body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with request(job, name, body) as response:
        return json.load(response)


def fetch_shards(job, manifest_name, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = get_json(job, manifest_name)

    def fetch_one(shard):
        path = root / shard["name"]
        if path.name != shard["name"]:
            raise ValueError("Invalid shard path")
        if (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
        ):
            return
        with request(job, shard["name"]) as response:
            body = response.read(32 * 1024 * 1024 + 1)
        if (
            len(body) != shard["bytes"]
            or hashlib.sha256(body).hexdigest() != shard["sha256"]
        ):
            raise ValueError("Cloud shard checksum mismatch")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(body)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch_one, manifest["shards"]))
    (root / manifest_name).write_text(json.dumps(manifest, sort_keys=True))
    return root / manifest_name


def upload_directory(directory, job):
    """Bounded parts support retry/resume; publish the complete manifest last."""
    root = Path(directory)
    files = {}
    for file in sorted(root.iterdir()):
        if not file.is_file() or not re.fullmatch(r"[a-zA-Z0-9_.-]+", file.name):
            raise ValueError("Only flat artifact directories may be published")
        digest = hashlib.sha256()
        parts = []
        size = 0
        with file.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                body = gzip.compress(chunk, compresslevel=1, mtime=0)
                name = file.name.replace(".", "-") + f"-part{len(parts):05d}.bin.gz"
                with request(job, name, body) as response:
                    response.read(4096)
                parts.append(
                    {
                        "name": name,
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "bytes": len(body),
                    }
                )
        files[file.name] = {"sha256": digest.hexdigest(), "bytes": size, "parts": parts}
    manifest = {"schema": "slp.artifact-parts/v1", "job": job, "files": files}
    put_json(job, "artifact.json", manifest)
    return manifest


def download_directory(job, directory):
    manifest = get_json(job, "artifact.json")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact path")
        path = root / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        digest = hashlib.sha256()
        count = 0
        with tmp.open("wb") as file:
            for part in spec["parts"]:
                with request(job, part["name"]) as response:
                    body = response.read(32 * 1024 * 1024 + 1)
                if (
                    len(body) != part["bytes"]
                    or hashlib.sha256(body).hexdigest() != part["sha256"]
                ):
                    raise ValueError("Artifact part checksum mismatch")
                chunk = gzip.decompress(body)
                digest.update(chunk)
                count += len(chunk)
                file.write(chunk)
        if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise ValueError("Reassembled artifact mismatch")
        tmp.replace(path)
    return manifest
