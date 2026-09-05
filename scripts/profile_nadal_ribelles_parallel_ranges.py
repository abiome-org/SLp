"""Profile four strict disjoint HTTP ranges of the Nadal-Ribelles archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from profile_nadal_ribelles_seus_split_download import ARCHIVE_BYTES, URL

MIB = 1024 * 1024
FIRST_BYTE = 64 * MIB
PART_BYTES = 16 * MIB
PARTS = 4


def fetch_part(index: int, output_dir: Path, global_start: float, maximum_seconds: float) -> dict:
    start = FIRST_BYTE + index * PART_BYTES
    end = start + PART_BYTES - 1
    expected_range = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
    destination = output_dir / f"seus_split-range-{start:010d}-{end:010d}.bin"
    partial = destination.with_suffix(".bin.part")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"immutable range output already exists: {destination}")
    digest = hashlib.sha256()
    received = 0
    local_start = time.monotonic()
    try:
        with requests.get(
            URL,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(15, 30),
        ) as response:
            if response.status_code != 206:
                raise RuntimeError(f"part {index}: expected HTTP 206, got {response.status_code}")
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError(
                    f"part {index}: unexpected Content-Range {response.headers.get('Content-Range')!r}"
                )
            if response.headers.get("Content-Length") != str(PART_BYTES):
                raise RuntimeError(
                    f"part {index}: unexpected Content-Length {response.headers.get('Content-Length')!r}"
                )
            with partial.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=MIB):
                    if not chunk:
                        continue
                    if time.monotonic() - global_start > maximum_seconds:
                        raise TimeoutError(f"parallel profile exceeded {maximum_seconds} seconds")
                    if received + len(chunk) > PART_BYTES:
                        raise RuntimeError(f"part {index}: server exceeded requested range")
                    stream.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if received != PART_BYTES:
                raise RuntimeError(f"part {index}: received {received}, expected {PART_BYTES}")
            os.replace(partial, destination)
            elapsed = time.monotonic() - local_start
            return {
                "index": index,
                "start": start,
                "end": end,
                "bytes": received,
                "http_status": response.status_code,
                "content_range": expected_range,
                "content_length": PART_BYTES,
                "sha256": digest.hexdigest(),
                "elapsed_seconds": elapsed,
                "path": destination.as_posix(),
            }
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def profile(output_dir: Path, report_path: Path, maximum_seconds: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f"immutable report exists: {report_path}")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARTS) as executor:
        futures = [
            executor.submit(fetch_part, index, output_dir, started, maximum_seconds)
            for index in range(PARTS)
        ]
        parts = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    total_bytes = sum(part["bytes"] for part in parts)
    throughput = total_bytes / elapsed
    result = {
        "schema": "slp.parallel-http-range-download-profile/v1",
        "status": "complete-prefix-ranges-only",
        "url": URL,
        "archive_bytes": ARCHIVE_BYTES,
        "connection_count": PARTS,
        "maximum_elapsed_seconds": maximum_seconds,
        "elapsed_seconds": elapsed,
        "profile_bytes": total_bytes,
        "aggregate_throughput_bytes_per_second": throughput,
        "aggregate_throughput_mib_per_second": throughput / MIB,
        "projected_full_download_seconds_at_profile_rate": ARCHIVE_BYTES / throughput,
        "parts": sorted(parts, key=lambda part: part["start"]),
        "interpretation": "Parallel transfer feasibility only; no complete archive or full checksum exists.",
    }
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=180.0)
    args = parser.parse_args()
    print(json.dumps(profile(args.output_dir, args.report, args.maximum_seconds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
