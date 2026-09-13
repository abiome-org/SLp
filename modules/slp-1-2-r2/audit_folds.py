"""One-pass exposure coverage for every declared inner and outer gene mask."""

import json
from pathlib import Path
import numpy as np
from data import digest
from cloud_io import get_json


def load_protocols(ticket_path):
    tickets = json.loads(Path(ticket_path).read_text())["tickets"]
    addresses = sorted(
        {
            (p["job"], p["name"])
            for p in tickets
            if p["method"] == "GET" and p["name"].endswith("-fold.json")
        }
    )
    result = []
    for job, name in addresses:
        spec = get_json(job, name)
        for scope in ("inner", "outer"):
            forbidden = set(spec["outer_held"]) | (
                set(spec["inner_held"]) if scope == "inner" else set()
            )
            result.append(
                {
                    "name": spec["name"],
                    "benchmark": spec["benchmark"],
                    "scope": scope,
                    "job": job,
                    "file": name,
                    "forbidden": sorted(forbidden),
                    "protocol_sha256": __import__("hashlib")
                    .sha256(json.dumps(spec, sort_keys=True).encode())
                    .hexdigest(),
                }
            )
    return result


def audit(protocols, genes, arrays):
    if not 0 < len(protocols) <= 64:
        raise ValueError("A single audit supports 1–64 declared views")
    index = {g: i for i, g in enumerate(genes)}
    bits = np.zeros(len(genes) + 1, np.uint64)
    for i, p in enumerate(protocols):
        for gene in p["forbidden"]:
            if gene not in index:
                raise ValueError("Unresolved held gene")
            bits[index[gene]] |= np.uint64(1) << np.uint64(i)
    reports = []
    for array_path, templates, label in arrays:
        rows = np.load(array_path, mmap_mode="r", allow_pickle=False)
        taxon = np.array([t["taxon"] for t in templates])
        admitted = np.array(
            [t.get("training_allowed") is True and t["kind"] != "sl" for t in templates]
        )
        counts = np.zeros((len(protocols), 2), np.int64)
        excluded = np.zeros(len(protocols), np.int64)
        for lo in range(0, len(rows), 262144):
            part = rows[lo : lo + 262144]
            tid = part["template"]
            human = taxon[tid] == 9606
            exposure = np.bitwise_or.reduce(bits[part["targets"]], axis=1)
            exposure = np.where(human, exposure, np.uint64(0))
            for i in range(len(protocols)):
                held = (exposure & (np.uint64(1) << np.uint64(i))) != 0
                kept = admitted[tid] & ~held
                counts[i, 0] += int(kept.sum())
                counts[i, 1] += int((kept & human).sum())
                excluded[i] += int((admitted[tid] & held).sum())
        reports.append(
            {
                "source": label,
                "input_units_or_measurements": len(rows),
                "array_sha256": digest(array_path),
                "views": [
                    {
                        "protocol": p["name"],
                        "scope": p["scope"],
                        "fitting_pretrain": int(counts[i, 0]),
                        "fitting_human_adaptation": int(counts[i, 1]),
                        "excluded_human": int(excluded[i]),
                    }
                    for i, p in enumerate(protocols)
                ],
            }
        )
    return {
        "schema": "slp.all-fold-exposure-coverage/v1",
        "view_count": len(protocols),
        "protocols": protocols,
        "sources": reports,
        "definition": "Strict masks before fitted transforms; counts are population units for dense RNA and measurements for packed outcomes, not unique genes or independent experiments.",
    }
