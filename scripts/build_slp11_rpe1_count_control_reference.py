#!/usr/bin/env python3
"""Build the frozen RPE1 GEM-specific positive control CP10k reference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1"
ROUTING_SHA256 = "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2"
MOMENTS_SHA256 = "5aceba5fb4874811aac797be14d1947a9fca866d11178d5f8fe2bdc534df6f61"
QUERY_ROSTER_SHA256 = "20f22e3f4c58981d6805911e4dc1f2069a387b2b2be695c8eabd155d62432e79"
QUERY_COUNT = 8749
GEM_COUNT = 56
PSEUDOCOUNT = 0.5


class RpeControlError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
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
    """Apply Jeffreys-style additive smoothing in full-panel count space."""
    raw = np.asarray(raw_count_sum)
    library = np.asarray(library_count_sum)
    cells = np.asarray(num_cells)
    if raw.ndim != 2 or library.shape != (len(raw),) or cells.shape != (len(raw),):
        raise RpeControlError("control moments must align on GEM rows")
    raw64, library64 = raw.astype(np.float64), library.astype(np.float64)
    if (
        not np.isfinite(raw64).all()
        or not np.isfinite(library64).all()
        or np.any(raw64 < 0)
        or np.any(library64 <= 0)
        or np.any(cells <= 0)
        or not np.equal(raw64, np.floor(raw64)).all()
        or not np.equal(library64, np.floor(library64)).all()
        or not np.array_equal(raw64.sum(1, dtype=np.float64), library64)
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise RpeControlError("control counts/libraries must be exact supported moments")
    queries = raw.shape[1]
    rate64 = 10000.0 * (raw64 + pseudocount) / (
        library64[:, None] + pseudocount * queries
    )
    if not np.isfinite(rate64).all() or np.any(rate64 <= 0):
        raise RpeControlError("control smoothing did not produce finite positive rates")
    return rate64.astype(np.float32), {
        "minimumRate": float(rate64.min()),
        "maximumRate": float(rate64.max()),
        "maximumFloat64MassError": float(np.max(np.abs(rate64.sum(1) - 10000.0))),
    }


def scalar_text(values: dict[str, np.ndarray], key: str) -> str:
    if key not in values or values[key].ndim != 0:
        raise RpeControlError(f"missing scalar {key}")
    return str(values[key].item())


def build(args: argparse.Namespace) -> dict[str, object]:
    if sha256_file(args.moments) != MOMENTS_SHA256:
        raise RpeControlError("RPE1 control moments hash mismatch")
    with np.load(args.moments, allow_pickle=False) as archive:
        moments = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        scalar_text(moments, "source_sha256") != SOURCE_SHA256
        or scalar_text(moments, "routing_sha256") != ROUTING_SHA256
        or scalar_text(moments, "schema")
        != "slp.replogle-rpe1-essential-raw-cell-shards/v1.control-gem-moments"
    ):
        raise RpeControlError("RPE1 control moment lineage mismatch")
    query_ids = moments["query_ids"].astype(str)
    query_bytes = "".join(gene + "\n" for gene in query_ids).encode("ascii")
    gem = np.asarray(moments["gem_group"], dtype=np.int16)
    if (
        query_ids.shape != (QUERY_COUNT,)
        or hashlib.sha256(query_bytes).hexdigest() != QUERY_ROSTER_SHA256
        or not np.all(moments["query_taxon"] == 9606)
        or not np.array_equal(gem, np.arange(1, GEM_COUNT + 1))
        or int(np.asarray(moments["num_cells"]).sum()) != 10350
    ):
        raise RpeControlError("RPE1 query/GEM/control support drift")
    rate, audit = positive_control_rate(
        moments["raw_count_sum"], moments["library_count_sum"], moments["num_cells"]
    )
    arrays = {
        "schema": np.asarray("slp.rpe1-essential-gem-control-reference/v1"),
        "query_ids": query_ids,
        "query_taxon": np.full(QUERY_COUNT, 9606, dtype=np.int64),
        "gem_group": gem,
        "basal_rate": rate,
        "basal_mask": np.ones((GEM_COUNT, QUERY_COUNT), dtype=np.bool_),
        "control_num_cells": np.asarray(moments["num_cells"], dtype=np.int64),
        "control_raw_count_sum": np.asarray(moments["raw_count_sum"], dtype=np.int64),
        "control_library_count_sum": np.asarray(moments["library_count_sum"], dtype=np.int64),
        "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256),
        "control_moments_sha256": np.asarray(MOMENTS_SHA256),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact = args.output_dir / "gem-control-reference.npz"
    artifact.write_bytes(deterministic_npz(arrays))
    manifest = {
        "schema": "slp.rpe1-essential-gem-control-reference-manifest/v1",
        "artifact": {"path": artifact.name, "sha256": sha256_file(artifact), "shape": [GEM_COUNT, QUERY_COUNT]},
        "formula": "basal_rate[g,q]=10000*(pooled_raw_control_count[g,q]+0.5)/(pooled_full_8749_query_library_count[g]+0.5*8749)",
        "domain": "10,350 reconstruction-training verified non-targeting controls only, separately in all 56 GEM groups",
        "support": {
            "gemGroups": GEM_COUNT, "queries": QUERY_COUNT,
            "minimumCellsPerGem": int(arrays["control_num_cells"].min()),
            "maximumCellsPerGem": int(arrays["control_num_cells"].max()),
            "basalMaskAllTrue": True, **audit,
        },
        "inputs": {
            "moments": {"path": args.moments.as_posix(), "sha256": MOMENTS_SHA256},
            "source": {"sha256": SOURCE_SHA256}, "routing": {"sha256": ROUTING_SHA256},
            "queryRoster": {"sha256": QUERY_ROSTER_SHA256},
        },
        "runtime": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "source": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256_file(Path(__file__).resolve())}},
        "accessBoundary": {"sourceCountMatrixReadByThisBuilder": False, "controlSufficientStatisticsRead": True, "reconstructionHeldControlsExcludedUpstream": True, "developmentTestUnresolvedCellsExcluded": True},
        "limitation": "Exact empty-action identity is relative to this explicitly smoothed reconstruction-training control estimator.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", type=Path, default=ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/control-gem-moments.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1")
    return p


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
