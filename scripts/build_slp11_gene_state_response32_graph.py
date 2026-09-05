"""Build the immutable response32-augmented source-three gene-state graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "modules/slp-1-1-world-transition-v1/gene_state_response_features.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gene_state_response_features", MODULE_PATH
)
FEATURES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FEATURES)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for label, path in (("graph", args.graph), ("reference", args.reference)):
        if sha256(path) != protocol["inputs"][label]["sha256"]:
            raise ValueError(f"{label} SHA-256 mismatch")
    if (args.output_dir / "source3-gene-state-response32-graph.npz").exists():
        raise FileExistsError("immutable augmented graph already exists")
    with (
        np.load(args.graph, allow_pickle=False) as graph,
        np.load(args.reference, allow_pickle=False) as reference,
    ):
        arrays, audit = FEATURES.augment_graph(graph, reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "source3-gene-state-response32-graph.npz"
    FEATURES.write_npz(output, arrays)
    with (
        np.load(args.graph, allow_pickle=False) as graph,
        np.load(output, allow_pickle=False) as augmented,
    ):
        fields = FEATURES.graph_field_audit(graph, augmented)
    if not all(item["exact"] for item in fields.values()):
        raise RuntimeError("original graph field drift after serialization")
    manifest = {
        "schema": "slp.source3-gene-state-response32-graph/v1",
        "status": "prepared-not-trained-not-scored",
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        },
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "audit": audit,
        "original_graph_fields": fields,
        "limitations": [
            "The response32 block is fitted from source-three training RNA outcomes and is not a static prior.",
            "A future model will expose the block to the shared node encoder, including action and query nodes; this is not an isolated decoder-only ablation.",
            "No new response basis, model fit, prediction, score, HepG2, Jurkat, Frangieh, or benchmark outcome is used here.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
