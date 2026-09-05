"""Resume-safe bounded acquisition of Replogle RPE1 essential raw single cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/sources/replogle-2022-rpe1-essential-singlecell-v1"
URL = "https://ndownloader.figshare.com/files/35775606"
FILENAME = "rpe1_raw_singlecell_01.h5ad"
BYTES = 8_700_873_216
MD5 = "6a2a9d0d2bf4ec147f4d1104043b268c"
SOURCE_VERSION = "figshare-plus:20029387:v1:file:35775606"
MIB = 1024 * 1024
GIB = 1024 * MIB
RANGE_BYTES = 64 * MIB
CONNECTIONS = 4
DERIVED_RESERVE = 8 * GIB
FREE_RESERVE = 16 * GIB


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_hashes(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * MIB):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha.hexdigest(), size


class Acquisition:
    def __init__(self, output: Path, maximum_seconds: float) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.partial = output / f"{FILENAME}.partial"
        self.final = output / FILENAME
        self.ledger_path = output / "download-ledger.json"
        self.profile_path = output / "throughput-profile.json"
        self.complete_path = output / "complete.json"
        self.maximum_seconds = maximum_seconds
        self.started = time.monotonic()
        self.lock = threading.Lock()
        if self.final.exists() or self.complete_path.exists():
            raise FileExistsError("immutable verified source already exists")
        self._check_disk()
        self.ledger = self._load_or_create()

    def _check_disk(self) -> None:
        free = shutil.disk_usage(self.output.parent if self.output.parent.exists() else ROOT).free
        required = (0 if self.partial.exists() else BYTES) + DERIVED_RESERVE + FREE_RESERVE
        if free < required:
            raise RuntimeError(f"insufficient disk: free={free}, required={required}")

    def _new_ledger(self) -> dict:
        return {
            "schema": "slp.parallel-range-download-ledger/v1", "status": "partial",
            "source_version": SOURCE_VERSION, "url": URL, "filename": FILENAME,
            "bytes": BYTES, "expected_md5": MD5, "range_bytes": RANGE_BYTES,
            "maximum_connections": CONNECTIONS, "maximum_seconds": self.maximum_seconds,
            "completed_ranges": {},
        }

    def _load_or_create(self) -> dict:
        if self.ledger_path.exists():
            value = json.loads(self.ledger_path.read_text())
            contract = (value.get("url"), value.get("filename"), value.get("bytes"), value.get("expected_md5"), value.get("range_bytes"), value.get("maximum_connections"))
            if contract != (URL, FILENAME, BYTES, MD5, RANGE_BYTES, CONNECTIONS):
                raise RuntimeError("resume ledger contract drift")
            if not self.partial.is_file() or self.partial.stat().st_size != BYTES:
                raise RuntimeError("resume partial file missing or wrong size")
            self._verify_completed(value)
            return value
        if self.partial.exists():
            raise FileExistsError("partial source exists without ledger")
        with self.partial.open("xb") as stream:
            stream.truncate(BYTES)
        value = self._new_ledger()
        atomic_json(self.ledger_path, value)
        return value

    def _verify_completed(self, ledger: dict) -> None:
        with self.partial.open("rb") as stream:
            for key, item in sorted(ledger["completed_ranges"].items(), key=lambda pair: pair[1]["start"]):
                length = item["end"] - item["start"] + 1
                stream.seek(item["start"])
                value = hashlib.sha256()
                remaining = length
                while remaining:
                    block = stream.read(min(MIB, remaining))
                    if not block:
                        raise RuntimeError(f"short resumed range {key}")
                    value.update(block)
                    remaining -= len(block)
                if value.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"resumed range checksum drift {key}")

    def _covered(self, start: int, end: int) -> bool:
        return f"{start}-{end}" in self.ledger["completed_ranges"]

    def _record(self, result: dict) -> None:
        key = f"{result['start']}-{result['end']}"
        with self.lock:
            self.ledger["completed_ranges"][key] = result
            atomic_json(self.ledger_path, self.ledger)

    def _fetch_once(self, start: int, end: int) -> dict:
        expected = end - start + 1
        expected_range = f"bytes {start}-{end}/{BYTES}"
        began = time.monotonic()
        received = 0
        value = hashlib.sha256()
        with requests.get(URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, stream=True, timeout=(20, 60)) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                raise RuntimeError(f"range {start}-{end}: response contract {response.status_code} {response.headers.get('Content-Range')!r}")
            if response.headers.get("Content-Length") != str(expected):
                raise RuntimeError(f"range {start}-{end}: content length drift")
            with self.partial.open("r+b", buffering=0) as destination:
                destination.seek(start)
                for chunk in response.iter_content(MIB):
                    if time.monotonic() - self.started > self.maximum_seconds:
                        raise TimeoutError("acquisition wall-time cap reached")
                    if not chunk:
                        continue
                    if received + len(chunk) > expected:
                        raise RuntimeError("range response exceeded request")
                    destination.write(chunk)
                    value.update(chunk)
                    received += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        if received != expected:
            raise RuntimeError(f"short range {start}-{end}: {received}/{expected}")
        return {"start": start, "end": end, "bytes": received, "sha256": value.hexdigest(), "seconds": time.monotonic() - began, "attempts": 1}

    def _fetch(self, start: int, end: int) -> dict:
        errors = []
        for attempt in range(1, 4):
            try:
                result = self._fetch_once(start, end)
                result["attempts"] = attempt
                self._record(result)
                return result
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                if attempt == 3:
                    raise RuntimeError(f"range {start}-{end} failed bounded retries: {errors}") from error
                time.sleep(attempt)
        raise AssertionError("unreachable")

    def ranges(self) -> list[tuple[int, int]]:
        return [
            (start, min(start + RANGE_BYTES, BYTES) - 1)
            for start in range(0, BYTES, RANGE_BYTES)
            if not self._covered(start, min(start + RANGE_BYTES, BYTES) - 1)
        ]

    def _parallel(self, tasks: list[tuple[int, int]]) -> list[dict]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as pool:
            futures = [pool.submit(self._fetch, *task) for task in tasks]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                if count % 8 == 0 or count == len(futures):
                    completed = sum(item["bytes"] for item in self.ledger["completed_ranges"].values())
                    print(json.dumps({"event": "progress", "bytes": completed, "fraction": completed / BYTES, "elapsedSeconds": time.monotonic() - self.started}), flush=True)
        return results

    def profile(self) -> dict:
        if self.profile_path.exists():
            return json.loads(self.profile_path.read_text())
        tasks = self.ranges()[:CONNECTIONS]
        began = time.monotonic()
        results = self._parallel(tasks)
        elapsed = time.monotonic() - began
        transferred = sum(item["bytes"] for item in results)
        throughput = transferred / elapsed
        remaining = BYTES - sum(item["bytes"] for item in self.ledger["completed_ranges"].values())
        projected = remaining / throughput + 180.0
        result = {"schema": "slp.range-throughput-profile/v1", "connections": CONNECTIONS, "rangeBytes": RANGE_BYTES, "bytes": transferred, "seconds": elapsed, "bytesPerSecond": throughput, "projectedRemainingDownloadPlusHashSeconds": projected, "maximumSeconds": self.maximum_seconds}
        atomic_json(self.profile_path, result)
        if time.monotonic() - self.started + projected > self.maximum_seconds:
            raise TimeoutError(f"profile projects {projected:.1f}s remaining beyond cap")
        print(json.dumps({"event": "profile", **result}), flush=True)
        return result

    def run(self) -> dict:
        profile = self.profile()
        network_started = time.monotonic()
        results = self._parallel(self.ranges())
        network_seconds = time.monotonic() - network_started
        if self.ranges():
            raise RuntimeError("range coverage incomplete")
        md5, sha, size = file_hashes(self.partial)
        total = time.monotonic() - self.started
        if size != BYTES or md5 != MD5:
            raise RuntimeError(f"whole-file identity mismatch: bytes={size}, md5={md5}")
        if total > self.maximum_seconds:
            raise TimeoutError(f"verified acquisition exceeded cap: {total:.1f}s")
        os.replace(self.partial, self.final)
        self.ledger.update({"status": "complete", "whole_file_md5": md5, "whole_file_sha256": sha, "total_elapsed_seconds": total})
        atomic_json(self.ledger_path, self.ledger)
        result = {"schema": "slp.immutable-source-download/v1", "status": "complete-verified", "sourceVersion": SOURCE_VERSION, "url": URL, "path": self.final.as_posix(), "bytes": size, "md5": md5, "sha256": sha, "maximumConnections": CONNECTIONS, "rangeBytes": RANGE_BYTES, "downloadedRangesThisRun": len(results), "networkSecondsAfterProfile": network_seconds, "totalSeconds": total, "profile": profile}
        atomic_json(self.complete_path, result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--maximum-seconds", type=float, default=3500.0)
    args = parser.parse_args()
    print(json.dumps(Acquisition(args.output, args.maximum_seconds).run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
