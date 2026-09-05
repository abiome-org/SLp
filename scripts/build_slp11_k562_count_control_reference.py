#!/usr/bin/env python3
"""Build positive GEM-specific control CP10k rates from frozen raw moments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
ROUTING_SHA256 = "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad"
QUERY_ROSTER_SHA256 = "9182efe0304204a30418c55d364de3178557d6b0813748436d9fa81b54da4d79"
PSEUDOCOUNT = 0.5
QUERY_COUNT = 8563
GEM_COUNT = 48


class ControlReferenceError(ValueError):
    """Raised when control moments violate the frozen reference contract."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def positive_control_rate(
    raw_count_sum: np.ndarray,
    library_count_sum: np.ndarray,
    num_cells: np.ndarray,
    *,
    pseudocount: float = PSEUDOCOUNT,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the frozen additive smoothing in source-count space."""
    raw = np.asarray(raw_count_sum)
    library = np.asarray(library_count_sum)
    cells = np.asarray(num_cells)
    if raw.ndim != 2 or raw.shape[1] != QUERY_COUNT:
        raise ControlReferenceError("raw_count_sum must be [GEM,8563]")
    if library.shape != (len(raw),) or cells.shape != (len(raw),):
        raise ControlReferenceError("library/cell moments must align to GEM rows")
    raw64 = raw.astype(np.float64)
    library64 = library.astype(np.float64)
    if (
        not np.isfinite(raw64).all()
        or not np.isfinite(library64).all()
        or np.any(raw64 < 0)
        or np.any(library64 <= 0)
        or np.any(cells <= 0)
        or not np.equal(raw64, np.floor(raw64)).all()
        or not np.equal(library64, np.floor(library64)).all()
    ):
        raise ControlReferenceError("control counts must be finite integer moments")
    row_total = raw64.sum(axis=1, dtype=np.float64)
    if not np.array_equal(row_total, library64):
        raise ControlReferenceError("full-panel raw sums do not equal source libraries")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ControlReferenceError("pseudocount must be positive")
    denominator = library64 + pseudocount * QUERY_COUNT
    rate64 = 10000.0 * (raw64 + pseudocount) / denominator[:, None]
    if not np.isfinite(rate64).all() or np.any(rate64 <= 0):
        raise ControlReferenceError("control smoothing did not produce positive rates")
    audit = {
        "minimumRate": float(rate64.min()),
        "maximumRate": float(rate64.max()),
        "maximumFloat64MassError": float(np.max(np.abs(rate64.sum(1) - 10000.0))),
    }
    return rate64.astype(np.float32), audit


def scalar_text(arrays: dict[str, np.ndarray], name: str) -> str:
    if name not in arrays or arrays[name].ndim != 0:
        raise ControlReferenceError(f"missing scalar {name}")
    return str(arrays[name].item())


def build(args: argparse.Namespace) -> dict[str, object]:
    actual = sha256_file(args.moments)
    if actual != args.moments_sha256:
        raise ControlReferenceError(f"control moments SHA-256 mismatch: {actual}")
    with np.load(args.moments, allow_pickle=False) as source:
        moments = {name: source[name].copy() for name in source.files}
    if scalar_text(moments, "source_sha256") != SOURCE_SHA256:
        raise ControlReferenceError("source identity mismatch")
    if scalar_text(moments, "routing_sha256") != ROUTING_SHA256:
        raise ControlReferenceError("routing identity mismatch")
    query_ids = moments["query_ids"].astype(str)
    payload = "".join(f"{item}\n" for item in query_ids).encode("ascii")
    if len(query_ids) != QUERY_COUNT or hashlib.sha256(payload).hexdigest() != QUERY_ROSTER_SHA256:
        raise ControlReferenceError("ordered query roster mismatch")
    gem_group = np.asarray(moments["gem_group"])
    if not np.array_equal(gem_group, np.arange(1, GEM_COUNT + 1)):
        raise ControlReferenceError("all 48 ordered GEM groups are required")
    rate, audit = positive_control_rate(
        moments["raw_count_sum"], moments["library_count_sum"], moments["num_cells"]
    )
    if len(rate) != GEM_COUNT:
        raise ControlReferenceError("all 48 GEM groups require control support")
    arrays = {
        "schema": np.asarray("slp.k562-essential-gem-control-reference/v1"),
        "query_ids": query_ids,
        "query_taxon": np.full(QUERY_COUNT, 9606, dtype=np.int64),
        "gem_group": gem_group.astype(np.int16),
        "basal_rate": rate,
        "basal_mask": np.ones((GEM_COUNT, QUERY_COUNT), dtype=np.bool_),
        "control_num_cells": np.asarray(moments["num_cells"], dtype=np.int64),
        "control_raw_count_sum": np.asarray(moments["raw_count_sum"]),
        "control_library_count_sum": np.asarray(moments["library_count_sum"]),
        "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256),
        "control_moments_sha256": np.asarray(actual),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "gem-control-reference.npz"
    output.write_bytes(deterministic_npz(arrays))
    manifest = {
        "schema": "slp.k562-essential-gem-control-reference-manifest/v1",
        "artifact": {
            "path": output.name,
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rateShape": [GEM_COUNT, QUERY_COUNT],
        },
        "formula": "basal_rate[g,q]=10000*(pooled_raw_control_count[g,q]+0.5)/(pooled_full_source_library_count[g]+0.5*8563)",
        "domain": "reconstruction-training verified non-targeting controls only; positive-library cells; separately in every GEM group",
        "support": {
            "gemGroups": GEM_COUNT,
            "queries": QUERY_COUNT,
            "minimumCellsPerGem": int(np.min(arrays["control_num_cells"])),
            "maximumCellsPerGem": int(np.max(arrays["control_num_cells"])),
            "basalMaskAllTrue": True,
            **audit,
        },
        "inputs": {
            "moments": {"path": args.moments.as_posix(), "sha256": actual},
            "source": {"sha256": SOURCE_SHA256},
            "routing": {"sha256": ROUTING_SHA256},
            "queryRoster": {"sha256": QUERY_ROSTER_SHA256},
        },
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "source": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "accessBoundary": {
            "sourceCountMatrixReadByThisBuilder": False,
            "controlSufficientStatisticsRead": True,
            "reconstructionHeldControlsExcludedUpstream": True,
            "targetingDevelopmentAndTestCellsExcluded": True,
        },
        "limitation": "Exact empty-action identity is relative to this explicitly smoothed reconstruction-training control estimator.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--moments", type=Path, required=True)
    result.add_argument("--moments-sha256", required=True)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/derived/slp11-human-k562-essential-count-control/reconstruction-train-nt-gem-v1",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
