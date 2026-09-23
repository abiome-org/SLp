"""Parallel ranged download for slow single-stream hosts (figshare signed S3 redirects).

Each chunk re-requests the stable URL, so short-lived signed redirects are fine.
Usage: python -m slpbench.pget URL DEST [--conns 16]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path

import requests

UA = {"User-Agent": "slpbench/0.1"}


def size_of(url: str) -> int:
    r = requests.get(url, headers={**UA, "Range": "bytes=0-0"}, stream=True, allow_redirects=True, timeout=60)
    r.close()
    cr = r.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"no range support: {r.status_code} {r.headers}")
    return int(cr.rsplit("/", 1)[1])


def get_chunk(url: str, dest: Path, start: int, end: int) -> None:
    for attempt in range(8):
        try:
            with open(dest, "r+b") as f:
                f.seek(start)
                pos = start
                r = requests.get(url, headers={**UA, "Range": f"bytes={pos}-{end}"}, stream=True, timeout=120)
                r.raise_for_status()
                for block in r.iter_content(1 << 20):
                    f.write(block)
                    pos += len(block)
                if pos == end + 1:
                    return
                start = pos
        except requests.RequestException:
            pass
    raise RuntimeError(f"chunk {start}-{end} failed")


def pget(url: str, dest: Path, conns: int = 16) -> None:
    n = size_of(url)
    part = dest.with_suffix(dest.suffix + ".pget")
    with open(part, "wb") as f:
        f.truncate(n)
    step = -(-n // conns)
    with cf.ThreadPoolExecutor(conns) as ex:
        futs = [ex.submit(get_chunk, url, part, s, min(s + step, n) - 1) for s in range(0, n, step)]
        for fu in cf.as_completed(futs):
            fu.result()
    part.rename(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--conns", type=int, default=16)
    a = ap.parse_args()
    pget(a.url, a.dest, a.conns)
