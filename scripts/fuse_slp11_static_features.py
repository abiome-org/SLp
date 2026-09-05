#!/usr/bin/env python3
"""Concatenate static feature packs only when their composite keys match exactly."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ARRAYS = {"feature_values", "entity_taxon", "entity_id"}
BLOCK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
OUTPUT_SCHEMA = "slp.static-feature-fusion/v1"


class FeatureFusionError(ValueError):
    """Raised when feature packs cannot be fused without changing identity."""


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    path: Path
    manifest_path: Path
    values: np.ndarray
    taxa: np.ndarray
    identifiers: np.ndarray
    metadata: dict[str, Any]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Write a compressed NPZ with fixed member order and ZIP metadata."""
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in ("feature_values", "entity_taxon", "entity_id"):
            member = io.BytesIO()
            np.lib.format.write_array(member, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def parse_block_spec(raw: str) -> tuple[str, Path]:
    name, separator, path_text = raw.partition("=")
    if not separator or BLOCK_NAME_RE.fullmatch(name) is None or not path_text:
        raise FeatureFusionError("--block must be lowercase_name=path/to/features.npz")
    return name, Path(path_text)


def _load_manifest(path: Path, npz_path: Path, npz_payload: bytes) -> dict[str, Any]:
    if not path.is_file():
        raise FeatureFusionError(f"missing source manifest for {npz_path.name}: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureFusionError(f"invalid JSON source manifest: {path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("schema"), str):
        raise FeatureFusionError(f"source manifest lacks a schema: {path}")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise FeatureFusionError(f"source manifest lacks artifact identity: {path}")
    if (
        artifact.get("path") != npz_path.name
        or artifact.get("bytes") != len(npz_payload)
        or artifact.get("sha256") != sha256_bytes(npz_payload)
    ):
        raise FeatureFusionError(f"source manifest artifact identity mismatch: {path}")
    return manifest


def load_block(name: str, path: Path) -> FeatureBlock:
    if not path.is_file():
        raise FeatureFusionError(f"missing feature pack: {path}")
    payload = path.read_bytes()
    manifest_path = path.with_suffix(".manifest.json")
    manifest_payload = manifest_path.read_bytes() if manifest_path.is_file() else b""
    manifest = _load_manifest(manifest_path, path, payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_ARRAYS or len(archive.files) != 3:
                raise FeatureFusionError(f"unexpected arrays in feature pack: {path}")
            values = np.asarray(archive["feature_values"])
            taxa = np.asarray(archive["entity_taxon"])
            identifiers = np.asarray(archive["entity_id"])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, FeatureFusionError):
            raise
        raise FeatureFusionError(f"invalid non-pickle NPZ feature pack: {path}") from exc
    if values.ndim != 2 or values.dtype != np.dtype("float32"):
        raise FeatureFusionError(f"feature_values must be a 2D float32 array: {path}")
    if taxa.ndim != 1 or taxa.dtype != np.dtype("int64"):
        raise FeatureFusionError(f"entity_taxon must be a 1D int64 array: {path}")
    if identifiers.ndim != 1 or identifiers.dtype.kind != "U":
        raise FeatureFusionError(f"entity_id must be a 1D Unicode array: {path}")
    if len(values) != len(taxa) or len(values) != len(identifiers) or not len(values):
        raise FeatureFusionError(f"feature arrays have inconsistent or empty rows: {path}")
    if not values.shape[1] or not np.isfinite(values).all():
        raise FeatureFusionError(f"feature_values must be finite and nonempty: {path}")
    keys = [(int(taxon), str(identifier)) for taxon, identifier in zip(taxa, identifiers)]
    if keys != sorted(set(keys)):
        raise FeatureFusionError(f"composite keys must be uniquely sorted: {path}")
    metadata = {
        "artifact": {
            "fileName": path.name,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
        "manifest": {
            "fileName": manifest_path.name,
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
            "schema": manifest["schema"],
        },
    }
    return FeatureBlock(
        name=name,
        path=path,
        manifest_path=manifest_path,
        values=values,
        taxa=taxa,
        identifiers=identifiers,
        metadata=metadata,
    )


def fuse_blocks(blocks: list[FeatureBlock]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(blocks) < 2:
        raise FeatureFusionError("at least two feature blocks are required")
    if len({block.name for block in blocks}) != len(blocks):
        raise FeatureFusionError("feature block names must be unique")
    reference = blocks[0]
    for block in blocks[1:]:
        if not np.array_equal(block.taxa, reference.taxa) or not np.array_equal(
            block.identifiers, reference.identifiers
        ):
            raise FeatureFusionError(
                f"composite key rows differ between {reference.name} and {block.name}"
            )
    values = np.concatenate([block.values for block in blocks], axis=1).astype(
        np.dtype("<f4"), copy=False
    )
    arrays = {
        "feature_values": values,
        "entity_taxon": reference.taxa.astype(np.dtype("<i8"), copy=False),
        "entity_id": reference.identifiers.copy(),
    }
    start = 0
    block_metadata: list[dict[str, Any]] = []
    for block in blocks:
        end = start + block.values.shape[1]
        nonzero = np.any(block.values != 0, axis=1)
        block_metadata.append(
            {
                "name": block.name,
                "columns": {"startInclusive": start, "endExclusive": end},
                "dimensions": block.values.shape[1],
                "rowsWithAnyNonzeroFeature": int(nonzero.sum()),
                "zeroRows": int((~nonzero).sum()),
                "source": block.metadata,
            }
        )
        start = end
    fused_nonzero = np.any(values != 0, axis=1)
    metadata = {
        "featureBlocks": block_metadata,
        "coverage": {
            "rows": len(values),
            "rowsWithAnyNonzeroFeature": int(fused_nonzero.sum()),
            "zeroRows": int((~fused_nonzero).sum()),
        },
    }
    return arrays, metadata


def build_fusion(block_specs: list[str], output: Path, manifest_output: Path) -> dict[str, Any]:
    parsed = [parse_block_spec(spec) for spec in block_specs]
    blocks = [load_block(name, path) for name, path in parsed]
    arrays, fusion_metadata = fuse_blocks(blocks)
    if output.exists() or manifest_output.exists():
        raise FeatureFusionError("refusing to overwrite an existing fusion artifact")
    if output.resolve() == manifest_output.resolve():
        raise FeatureFusionError("NPZ and manifest output paths must differ")
    npz_payload = deterministic_npz_bytes(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(npz_payload)
    key_lines = "".join(
        f"{int(taxon)}\t{identifier}\n"
        for taxon, identifier in zip(
            arrays["entity_taxon"], arrays["entity_id"], strict=True
        )
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "status": "exploratory-static-feature-artifact-not-omf-admitted",
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "rows": len(arrays["entity_id"]),
            "orderedCompositeKeyListSha256": sha256_bytes(key_lines),
            "joinRule": "require-identical-ordered-full-row-coverage",
            "symbolJoinsUsed": False,
        },
        "arrays": {
            "feature_values": {
                "dtype": "little-endian-float32",
                "shape": list(arrays["feature_values"].shape),
            },
            "entity_taxon": {
                "dtype": "little-endian-int64",
                "shape": list(arrays["entity_taxon"].shape),
            },
            "entity_id": {
                "dtype": str(arrays["entity_id"].dtype),
                "shape": list(arrays["entity_id"].shape),
                "pickleRequired": False,
            },
        },
        **fusion_metadata,
        "construction": {
            "operation": "column-concatenation-without-scaling-or-fitting",
            "quantitativeOutcomesConsumed": False,
            "benchmarkLabelsConsumed": False,
        },
        "artifact": {
            "path": output.name,
            "bytes": len(npz_payload),
            "sha256": sha256_bytes(npz_payload),
            "compression": "zip-deflate-level-9-fixed-metadata",
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        action="append",
        required=True,
        help="Repeat as lowercase_name=path/to/features.npz in desired column order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    try:
        manifest = build_fusion(args.block, args.output, manifest_output)
    except (FeatureFusionError, OSError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
