"""Verify a built benchmark and export the model-facing files without private answers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from slpbench.evaluate import SCORER_VERSION, file_sha256

PUBLIC_FILES = (
    "train.parquet", "dev.parquet", "dev_semi.parquet", "test_inputs.parquet",
    "test_semi_inputs.parquet", "contexts.parquet", "gene_single_effects.parquet",
    "held_out_families.parquet", "manifest.json",
)


def verify_benchmark(bench: Path, raw: bool = False) -> dict:
    manifest = json.loads((bench / "manifest.json").read_text())
    checks = {}
    for name, expected in manifest["files"].items():
        path = bench / name
        if not path.is_file():
            raise ValueError(f"missing benchmark artifact: {path}")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"sha256 mismatch: {path}")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != expected["rows"]:
            raise ValueError(f"row count mismatch: {path}: {rows} != {expected['rows']}")
        checks[name] = digest
    if raw:
        root = Path("data/raw")
        for line in Path("reference/raw_sha256sums.txt").read_text().splitlines():
            digest, name = line.split("  ./", 1)
            path = root / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"missing or changed pinned raw file: {path}")
    return {"benchmark": manifest["version"], "manifest_sha256": file_sha256(bench / "manifest.json"),
            "scorer_version": SCORER_VERSION, "files_verified": len(checks), "raw_verified": raw}


def export_public(bench: Path, out: Path) -> dict:
    verification = verify_benchmark(bench)
    if out.resolve() == bench.resolve() or bench.resolve() in out.resolve().parents:
        raise ValueError("public export must not be inside the private benchmark directory")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"public export destination must be empty: {out}")
    for name in PUBLIC_FILES:
        shutil.copyfile(bench / name, out / name)
    lock = {**verification, "public_files": {name: file_sha256(out / name) for name in PUBLIC_FILES}}
    (out / "release.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock
