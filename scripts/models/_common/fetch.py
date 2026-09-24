"""Download a file into data/raw/<key>/ and record URL + sha256 in data/raw/<key>/SOURCES.tsv.

usage: uv run python scripts/models/_common/fetch.py <key> <url> [filename]
"""
import datetime, hashlib, os, subprocess, sys
from pathlib import Path

key, url = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else url.rstrip("/").split("/")[-1].split("?")[0]
d = Path("data/raw") / key
d.mkdir(parents=True, exist_ok=True)
out = d / name
if not out.exists() or out.stat().st_size == 0:
    tmp = out.with_suffix(out.suffix + ".part")
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(tmp), url], check=True)
    tmp.rename(out)
h = hashlib.sha256()
with open(out, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
man = d / "SOURCES.tsv"
rows = man.read_text().splitlines() if man.exists() else []
rows = [r for r in rows if r.split("\t")[0] != name]
rows.append(f"{name}\t{out.stat().st_size}\t{h.hexdigest()}\t{url}\t{datetime.date.today()}")
man.write_text("\n".join(rows) + "\n")
print(name, out.stat().st_size, h.hexdigest())
