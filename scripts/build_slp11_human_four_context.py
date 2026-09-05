"""Build the immutable Replogle plus retired-HepG2 development snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/four_context_data.py"
SPEC = importlib.util.spec_from_file_location("four_context_data", MODULE_PATH)
DATA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATA)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replogle", type=Path, required=True)
    parser.add_argument("--hepg2", type=Path, required=True)
    parser.add_argument("--hepg2-normalization", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inputs = {
        "replogle": args.replogle,
        "hepg2": args.hepg2,
        "hepg2_normalization": args.hepg2_normalization,
    }
    for label, path in inputs.items():
        if digest(path) != protocol["inputs"][label]["sha256"]:
            raise ValueError(f"{label} SHA-256 mismatch")
    with (
        np.load(args.replogle, allow_pickle=False) as replogle,
        np.load(args.hepg2, allow_pickle=False) as hepg2,
        np.load(args.hepg2_normalization, allow_pickle=False) as normalization,
    ):
        arrays, audit = DATA.build_four_context_arrays(replogle, hepg2, normalization)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "development.npz"
    DATA.write_npz(output, arrays)
    manifest = {
        "schema": "slp.human-four-context-development/v1",
        "status": "adaptive-development; HepG2 transfer diagnostic retired",
        "development": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": digest(output),
        },
        "protocol": {"path": str(args.protocol), "sha256": digest(args.protocol)},
        "audit": audit,
        "endpoint_compatibility": {
            "shared_query_identity": True,
            "shared_target_numeric_space": False,
            "replogle": str(arrays["target_value_space_by_context"][0]),
            "hepg2": str(arrays["target_value_space_by_context"][3]),
            "implication": "context-specific heads/references are required; pooled target scale is not assumed",
        },
        "uncertainty": {
            "replogle_control_target_pseudobulks": True,
            "hepg2_control_target_pseudobulks": False,
            "hepg2_available_control_statistics": "per-GEM control cell counts and aligned linear normalization mean/std/support",
            "compatibility": "HepG2 normalization statistics are not Replogle-space control pseudobulk residuals",
        },
        "rights": [
            "rights/figshare-replogle-2022-processed-perturb-seq-cc-by-4.0.yaml",
            "rights/figshare-replogle-2022-k562-gwps-cc-by-4.0.yaml",
            "rights/nadig-2025-gse264667-public-molecular.yaml",
        ],
        "excluded_sources": ["Frangieh2021", "Nadig2025 Jurkat"],
        "claim_limit": "HepG2 is adaptive development after failed frozen transfer and cannot be reported again as unseen-context confirmation.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
