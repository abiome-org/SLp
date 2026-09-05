"""Resume-safe four-range download of the pinned Nadal-Ribelles archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests
from profile_nadal_ribelles_seus_split_download import (
    ARCHIVE_BYTES,
    EXPECTED_MD5,
    PROFILE_BYTES,
    URL,
)

MIB = 1024 * 1024
RANGE_BYTES = 64 * MIB
MAX_CONNECTIONS = 4
SOURCE_VERSION = "zenodo:14062629;doi:10.5281/zenodo.14062629"


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = MIB if remaining is None else min(MIB, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise RuntimeError(f"short local file while hashing {path}")
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Acquisition:
    def __init__(
        self,
        output_dir: Path,
        first_prefix: Path,
        parallel_profile_dir: Path,
        maximum_seconds: float,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.partial = output_dir / "seus_split.RData.partial"
        self.final = output_dir / "seus_split.RData"
        self.ledger_path = output_dir / "download-ledger.json"
        self.complete_path = output_dir / "complete.json"
        self.first_prefix = first_prefix
        self.parallel_profile_dir = parallel_profile_dir
        self.maximum_seconds = maximum_seconds
        self.started = time.monotonic()
        self.lock = threading.Lock()
        if self.final.exists() or self.complete_path.exists():
            raise FileExistsError("immutable complete output already exists")
        self.ledger = self._load_or_create_ledger()

    def _new_ledger(self) -> dict:
        return {
            "schema": "slp.parallel-range-download-ledger/v1",
            "status": "partial",
            "source_version": SOURCE_VERSION,
            "url": URL,
            "archive_bytes": ARCHIVE_BYTES,
            "expected_md5": EXPECTED_MD5,
            "range_bytes": RANGE_BYTES,
            "maximum_connections": MAX_CONNECTIONS,
            "completed_ranges": {},
        }

    def _load_or_create_ledger(self) -> dict:
        if self.ledger_path.exists():
            ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            expected = (URL, ARCHIVE_BYTES, EXPECTED_MD5, RANGE_BYTES, MAX_CONNECTIONS)
            observed = (
                ledger.get("url"),
                ledger.get("archive_bytes"),
                ledger.get("expected_md5"),
                ledger.get("range_bytes"),
                ledger.get("maximum_connections"),
            )
            if observed != expected:
                raise RuntimeError("resume ledger does not match frozen download contract")
            if not self.partial.exists() or self.partial.stat().st_size != ARCHIVE_BYTES:
                raise RuntimeError("resume partial file missing or wrong size")
            return ledger
        if self.partial.exists():
            raise FileExistsError("partial archive exists without a resume ledger")
        with self.partial.open("xb") as stream:
            stream.truncate(ARCHIVE_BYTES)
        ledger = self._new_ledger()
        write_json_atomic(self.ledger_path, ledger)
        self._seed_verified_profiles(ledger)
        return ledger

    def _record_range(self, start: int, end: int, digest: str, origin: str) -> None:
        key = f"{start}-{end}"
        with self.lock:
            self.ledger["completed_ranges"][key] = {
                "start": start,
                "end": end,
                "bytes": end - start + 1,
                "sha256": digest,
                "origin": origin,
            }
            write_json_atomic(self.ledger_path, self.ledger)

    def _write_local_part(self, source: Path, start: int, expected_bytes: int) -> None:
        if source.stat().st_size != expected_bytes:
            raise RuntimeError(f"wrong seed size: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as incoming, self.partial.open("r+b", buffering=0) as destination:
            destination.seek(start)
            copied = 0
            while chunk := incoming.read(MIB):
                destination.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if copied != expected_bytes:
            raise RuntimeError(f"short seed copy: {source}")
        self._record_range(start, start + expected_bytes - 1, digest.hexdigest(), "verified-profile")

    def _seed_verified_profiles(self, ledger: dict) -> None:
        self.ledger = ledger
        if self.first_prefix.stat().st_size != PROFILE_BYTES:
            raise RuntimeError("first profile has wrong size")
        self._write_local_part(self.first_prefix, 0, PROFILE_BYTES)
        for index in range(4):
            start = PROFILE_BYTES + index * 16 * MIB
            end = start + 16 * MIB - 1
            source = self.parallel_profile_dir / f"seus_split-range-{start:010d}-{end:010d}.bin"
            self._write_local_part(source, start, 16 * MIB)

    def _covered(self, start: int, end: int) -> bool:
        intervals = sorted(
            (item["start"], item["end"]) for item in self.ledger["completed_ranges"].values()
        )
        cursor = start
        for left, right in intervals:
            if right < cursor:
                continue
            if left > cursor:
                return False
            cursor = max(cursor, right + 1)
            if cursor > end:
                return True
        return cursor > end

    def _fetch_range(self, start: int, end: int) -> dict:
        if time.monotonic() - self.started > self.maximum_seconds:
            raise TimeoutError("download wall-time cap reached before request")
        expected_bytes = end - start + 1
        expected_content_range = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        local_started = time.monotonic()
        digest = hashlib.sha256()
        received = 0
        with requests.get(
            URL,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(15, 30),
        ) as response:
            if response.status_code != 206:
                raise RuntimeError(f"range {start}-{end}: HTTP {response.status_code}")
            if response.headers.get("Content-Range") != expected_content_range:
                raise RuntimeError(
                    f"range {start}-{end}: Content-Range {response.headers.get('Content-Range')!r}"
                )
            if response.headers.get("Content-Length") != str(expected_bytes):
                raise RuntimeError(
                    f"range {start}-{end}: Content-Length {response.headers.get('Content-Length')!r}"
                )
            with self.partial.open("r+b", buffering=0) as destination:
                destination.seek(start)
                for chunk in response.iter_content(chunk_size=MIB):
                    if not chunk:
                        continue
                    if time.monotonic() - self.started > self.maximum_seconds:
                        raise TimeoutError("download wall-time cap reached during request")
                    if received + len(chunk) > expected_bytes:
                        raise RuntimeError(f"range {start}-{end}: response exceeded request")
                    destination.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        if received != expected_bytes:
            raise RuntimeError(f"range {start}-{end}: received {received}/{expected_bytes}")
        result = {
            "start": start,
            "end": end,
            "bytes": received,
            "sha256": digest.hexdigest(),
            "elapsed_seconds": time.monotonic() - local_started,
        }
        self._record_range(start, end, result["sha256"], "http-range")
        return result

    def download(self) -> dict:
        tasks = []
        for start in range(0, ARCHIVE_BYTES, RANGE_BYTES):
            end = min(start + RANGE_BYTES, ARCHIVE_BYTES) - 1
            if not self._covered(start, end):
                tasks.append((start, end))
        network_started = time.monotonic()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as executor:
            futures = [executor.submit(self._fetch_range, start, end) for start, end in tasks]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        network_elapsed = time.monotonic() - network_started
        if not self._covered(0, ARCHIVE_BYTES - 1):
            raise RuntimeError("download finished without complete range coverage")

        hash_started = time.monotonic()
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        bytes_hashed = 0
        with self.partial.open("rb") as stream:
            while chunk := stream.read(8 * MIB):
                md5.update(chunk)
                sha256.update(chunk)
                bytes_hashed += len(chunk)
        hash_elapsed = time.monotonic() - hash_started
        total_elapsed = time.monotonic() - self.started
        if bytes_hashed != ARCHIVE_BYTES:
            raise RuntimeError(f"whole-file hash covered {bytes_hashed}/{ARCHIVE_BYTES} bytes")
        if md5.hexdigest() != EXPECTED_MD5:
            raise RuntimeError(f"whole-file MD5 mismatch: {md5.hexdigest()}")
        if total_elapsed > self.maximum_seconds:
            raise TimeoutError(f"download and verification took {total_elapsed:.3f} seconds")
        os.replace(self.partial, self.final)
        self.ledger["status"] = "complete"
        self.ledger["whole_file_md5"] = md5.hexdigest()
        self.ledger["whole_file_sha256"] = sha256.hexdigest()
        self.ledger["network_elapsed_seconds"] = network_elapsed
        self.ledger["hash_elapsed_seconds"] = hash_elapsed
        self.ledger["total_elapsed_seconds"] = total_elapsed
        write_json_atomic(self.ledger_path, self.ledger)
        complete = {
            "schema": "slp.immutable-source-download/v1",
            "status": "complete-verified",
            "source_version": SOURCE_VERSION,
            "url": URL,
            "bytes": ARCHIVE_BYTES,
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(),
            "maximum_connections": MAX_CONNECTIONS,
            "range_bytes": RANGE_BYTES,
            "seeded_profile_bytes": 128 * MIB,
            "downloaded_ranges_this_run": len(results),
            "network_elapsed_seconds": network_elapsed,
            "hash_elapsed_seconds": hash_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "path": self.final.as_posix(),
        }
        write_json_atomic(self.complete_path, complete)
        return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--first-prefix", required=True, type=Path)
    parser.add_argument("--parallel-profile-dir", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    acquisition = Acquisition(
        args.output_dir,
        args.first_prefix,
        args.parallel_profile_dir,
        args.maximum_seconds,
    )
    print(json.dumps(acquisition.download(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
