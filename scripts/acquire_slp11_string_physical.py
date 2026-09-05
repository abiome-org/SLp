"""Acquire versioned human physical protein relations from official STRING."""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

FILES = {
    "9606.protein.physical.links.full.v12.0.txt.gz":
        "https://stringdb-downloads.org/download/protein.physical.links.full.v12.0/9606.protein.physical.links.full.v12.0.txt.gz",
    "9606.protein.aliases.v12.0.txt.gz":
        "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
}


def main():
    root = Path("data/sources/string-human-physical-v12.0")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("source directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"version":"12.0", "taxon":9606, "license":"CC-BY-4.0",
                "license_url":"https://version-12.string-db.org/cgi/access",
                "files":{}, "quantitative_intervention_outcomes":False}
    for name, url in FILES.items():
        path = root/name
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(url, timeout=90) as response, path.open("xb") as output:
            expected = int(response.headers.get("Content-Length", 0))
            while block := response.read(1024*1024):
                total += len(block)
                if total > 100*1024*1024:
                    raise ValueError("source exceeds bounded human download size")
                output.write(block)
                digest.update(block)
        if expected and expected != total:
            raise ValueError("incomplete official source")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            header = stream.readline().strip()
        manifest["files"][name] = {"url":url, "bytes":total,
                                   "sha256":digest.hexdigest(), "header":header}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
