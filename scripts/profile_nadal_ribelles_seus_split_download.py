"""Strict bounded HTTP range profile for the Nadal-Ribelles Seurat archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

URL = "https://zenodo.org/api/records/14062629/files/seus_split.RData/content"
ARCHIVE_BYTES = 5_907_877_873
EXPECTED_MD5 = "65bb56efd8120f32f65c044de5f040aa"
PROFILE_BYTES = 64 * 1024 * 1024
EXPECTED_CONTENT_RANGE = f"bytes 0-{PROFILE_BYTES - 1}/{ARCHIVE_BYTES}"
SOURCE_VERSION = "zenodo:14062629;doi:10.5281/zenodo.14062629"


def profile(output: Path, report_path: Path, maximum_seconds: float = 180.0) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists() or report_path.exists():
        raise FileExistsError("immutable profile output or report already exists")
    if partial.exists():
        partial.unlink()

    headers = {"Range": f"bytes=0-{PROFILE_BYTES - 1}", "Accept-Encoding": "identity"}
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    try:
        with requests.get(URL, headers=headers, stream=True, timeout=(15, 30)) as response:
            elapsed_headers = time.monotonic() - started
            content_range = response.headers.get("Content-Range")
            content_length = response.headers.get("Content-Length")
            if response.status_code != 206:
                raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
            if content_range != EXPECTED_CONTENT_RANGE:
                raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
            if content_length != str(PROFILE_BYTES):
                raise RuntimeError(f"unexpected Content-Length: {content_length!r}")
            with partial.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    if time.monotonic() - started > maximum_seconds:
                        raise TimeoutError(f"profile exceeded {maximum_seconds} seconds")
                    if received + len(chunk) > PROFILE_BYTES:
                        raise RuntimeError("server delivered bytes beyond requested range")
                    stream.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if received != PROFILE_BYTES:
                raise RuntimeError(f"expected {PROFILE_BYTES} bytes, received {received}")
            elapsed = time.monotonic() - started
            if elapsed > maximum_seconds:
                raise TimeoutError(f"profile took {elapsed:.3f} seconds")
            os.replace(partial, output)
            result = {
                "schema": "slp.http-range-download-profile/v1",
                "status": "complete-prefix-only",
                "source_version": SOURCE_VERSION,
                "url": URL,
                "final_url": response.url,
                "request_range": headers["Range"],
                "http_status": response.status_code,
                "content_range": content_range,
                "content_length": int(content_length),
                "accept_ranges": response.headers.get("Accept-Ranges"),
                "archive_bytes": ARCHIVE_BYTES,
                "expected_full_archive_md5": EXPECTED_MD5,
                "full_archive_md5_verified": False,
                "profile_bytes": received,
                "profile_sha256": digest.hexdigest(),
                "elapsed_to_headers_seconds": elapsed_headers,
                "elapsed_seconds": elapsed,
                "throughput_bytes_per_second": received / elapsed,
                "throughput_mib_per_second": received / elapsed / (1024 * 1024),
                "projected_full_download_seconds_at_profile_rate": ARCHIVE_BYTES / (received / elapsed),
                "maximum_elapsed_seconds": maximum_seconds,
                "output": output.as_posix(),
                "interpretation": "Transfer feasibility only; the prefix does not verify full-file integrity or parser feasibility.",
            }
            report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=180.0)
    args = parser.parse_args()
    print(json.dumps(profile(args.output, args.report, args.maximum_seconds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
