"""Materialize a pinned shared-panel development snapshot without imputation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1"))
from complete_panel import select_complete_panel


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if sha(args.data) != args.sha256:
        raise ValueError("source snapshot checksum mismatch")
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    result, indices = select_complete_panel(data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"development.npz", **result)
    manifest = {
        "source_sha256":args.sha256, "output_sha256":sha(output/"development.npz"),
        "query_selection":"observed for every training row; no target values used for selection",
        "query_count":len(indices), "original_query_count":len(data["query_ids"]),
        "original_query_indices":indices.tolist(),
        "excluded_query_ids":np.setdiff1d(data["query_ids"], result["query_ids"]).tolist(),
        "test_accessed":False, "imputation":False,
        "source_hashes":{p.name:sha(p) for p in (Path(__file__), Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1/complete_panel.py")},
    }
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key:value for key,value in manifest.items() if key not in ("original_query_indices", "excluded_query_ids")}))


if __name__ == "__main__":
    main()
