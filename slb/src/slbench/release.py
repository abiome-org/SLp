"""Verify a built benchmark and export the model-facing files without private answers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from slbench.evaluate import SCORER_VERSION, file_sha256

PUBLIC_FILES = (
    "train.parquet", "dev.parquet", "dev_semi.parquet", "test_inputs.parquet",
    "test_semi_inputs.parquet", "contexts.parquet", "gene_single_effects.parquet",
    "held_out_families.parquet", "hidden/dev_propensity.parquet",
    "hidden/dev_semi_propensity.parquet", "manifest.json",
)


def _verify_public(bench: Path, raw: bool) -> dict:
    if raw:
        raise ValueError("raw source verification requires the private benchmark directory")
    lock = json.loads((bench / "release.lock.json").read_text())
    manifest_path = bench / "manifest.json"
    if file_sha256(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("public manifest sha256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (lock.get("benchmark") != manifest["version"] or lock.get("scorer_version") != SCORER_VERSION
            or set(lock.get("public_files", {})) != set(PUBLIC_FILES)):
        raise ValueError("public release lock does not match this benchmark or scorer")
    found = {p.relative_to(bench).as_posix() for p in bench.rglob("*") if p.is_file()}
    if found != set(PUBLIC_FILES) | {"release.lock.json"}:
        raise ValueError("public release contains missing or extra files")
    for name, digest in lock["public_files"].items():
        path = bench / name
        if file_sha256(path) != digest:
            raise ValueError(f"sha256 mismatch: {path}")
        if name != "manifest.json":
            expected = manifest["files"][name]
            if digest != expected["sha256"] or pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
                raise ValueError(f"manifest mismatch: {path}")
    return {"benchmark": manifest["version"], "manifest_sha256": lock["manifest_sha256"],
            "scorer_version": SCORER_VERSION, "files_verified": len(PUBLIC_FILES),
            "raw_verified": False, "public_verified": True}


def verify_benchmark(bench: Path, raw: bool = False) -> dict:
    if (bench / "release.lock.json").is_file():
        return _verify_public(bench, raw)
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
        try:
            old = json.loads((out / "release.lock.json").read_text())
            _verify_public(out, raw=False)
            if old["manifest_sha256"] == verification["manifest_sha256"]:
                return old
        except (OSError, KeyError, ValueError):
            pass
        raise ValueError(f"public export destination contains different files: {out}")
    for name in PUBLIC_FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bench / name, out / name)
    lock = {**verification, "public_files": {name: file_sha256(out / name) for name in PUBLIC_FILES}}
    (out / "release.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock
