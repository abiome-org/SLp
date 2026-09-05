"""Bounded acquisition of public processed Frangieh Perturb-CITE-seq H5ADs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/frangieh-2021-scp1064-v1"
ZENODO_RECORD = SOURCE / "zenodo-record-10044268.json"
FILES = {
    "FrangiehIzar2021_RNA.h5ad": {
        "size": 1_458_928_348,
        "md5": "dc438f53c476a3dd562898654d48ed15",
        "url": "https://zenodo.org/api/records/10044268/files/FrangiehIzar2021_RNA.h5ad/content",
    },
    "FrangiehIzar2021_protein.h5ad": {
        "size": 24_714_445,
        "md5": "9d9337e30af69ffdcf5e831f96616ed1",
        "url": "https://zenodo.org/api/records/10044268/files/FrangiehIzar2021_protein.h5ad/content",
    },
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _fetch_range(url: str, path: Path, start: int, end: int) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as output:
        if response.status != 206:
            raise RuntimeError(f"server ignored byte range {start}-{end}")
        expected = f"bytes {start}-{end}/"
        if not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError("server returned an unexpected content range")
        shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
    if path.stat().st_size != end - start + 1:
        raise RuntimeError(f"short range response: {start}-{end}")


def ranged_download(url: str, destination: Path, size: int, workers: int = 8) -> None:
    part_dir = destination.with_name(destination.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    width = (size + workers - 1) // workers
    jobs = []
    for index in range(workers):
        start = index * width
        end = min(size, start + width) - 1
        if start >= size:
            break
        path = part_dir / f"part-{index:03d}"
        if not path.exists() or path.stat().st_size != end - start + 1:
            jobs.append((url, path, start, end))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_range, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    temporary = destination.with_name(destination.name + ".complete")
    with temporary.open("wb") as output:
        for index in range(workers):
            part = part_dir / f"part-{index:03d}"
            if not part.exists():
                break
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
    if temporary.stat().st_size != size:
        raise RuntimeError("assembled download has wrong size")
    os.replace(temporary, destination)
    shutil.rmtree(part_dir)


def main() -> int:
    record = json.loads(ZENODO_RECORD.read_text(encoding="utf-8"))
    if record.get("id") != 10044268 or record.get("metadata", {}).get("license", {}).get("id") != "cc-by-4.0":
        raise RuntimeError("Zenodo record identity or license drift")
    report = {}
    for name, expected in FILES.items():
        path = SOURCE / name
        if not (path.exists() and path.stat().st_size == expected["size"] and digest(path, "md5") == expected["md5"]):
            ranged_download(str(expected["url"]), path, int(expected["size"]))
        if digest(path, "md5") != expected["md5"]:
            raise RuntimeError(f"upstream MD5 mismatch: {name}")
        report[name] = {
            "bytes": path.stat().st_size,
            "md5": expected["md5"],
            "sha256": digest(path, "sha256"),
            "url": expected["url"],
        }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
