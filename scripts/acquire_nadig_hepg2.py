#!/usr/bin/env python3
"""Acquire and metadata-audit the official Nadig 2025 HepG2 H5AD.

The audit deliberately never indexes ``X``.  It reads only HDF5 structure,
``obs`` metadata, and ``var/gene_id`` so a future streaming normalizer can be
specified without inspecting perturbation-expression outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad"
)
SOURCE_NAME = "GSE264667_hepg2_raw_singlecell_01.h5ad"
EXPECTED_BYTES = 5_614_460_941
EXPECTED_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
DEFAULT_OUTPUT = Path("data/sources/nadig-2025-gse264667-hepg2-v1")
DEFAULT_REFERENCE = Path(
    "data/derived/slp11-human/"
    "replogle-k562-rpe1-author-normalized-development-v2.npz"
)


class AcquisitionError(RuntimeError):
    """Raised when the pinned source or metadata contract differs."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(path: Path) -> None:
    """Resume the official HTTPS download and enforce byte/hash identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != EXPECTED_BYTES:
        subprocess.run(
            [
                "curl.exe",
                "-L",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                SOURCE_URL,
                "-o",
                str(path),
            ],
            check=True,
        )
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise AcquisitionError(f"source has {size} bytes; expected {EXPECTED_BYTES}")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SHA256:
        raise AcquisitionError(
            f"source SHA-256 is {actual_hash}; expected {EXPECTED_SHA256}"
        )


def _decode(values: h5py.Dataset) -> np.ndarray:
    return np.asarray(values[...]).astype(str)


def _write_lines(path: Path, values: list[str]) -> dict[str, object]:
    payload = "".join(f"{value}\n" for value in values).encode()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "entries": len(values),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def audit_metadata(path: Path, reference_path: Path | None) -> dict[str, object]:
    """Audit identities and control counts without reading expression values."""

    with h5py.File(path, "r") as source:
        if set(source) != {"X", "obs", "var"}:
            raise AcquisitionError(f"unexpected H5AD top-level keys: {list(source)}")
        matrix = source["X"]
        if not isinstance(matrix, h5py.Dataset):
            raise AcquisitionError("X must be a dense HDF5 dataset")
        if matrix.shape != (145_473, 9_624) or matrix.dtype != np.dtype("float32"):
            raise AcquisitionError(
                f"unexpected X contract: shape={matrix.shape}, dtype={matrix.dtype}"
            )

        obs = source["obs"]
        required_obs = {
            "UMI_count",
            "cell_barcode",
            "gem_group",
            "gene",
            "gene_id",
            "gene_transcript",
            "mitopercent",
            "sgID_AB",
            "transcript",
            "z_gemgroup_UMI",
        }
        if not required_obs.issubset(obs):
            raise AcquisitionError("required observation metadata is missing")
        gene_ids = _decode(obs["__categories/gene_id"])
        gene_codes = np.asarray(obs["gene_id"][...], dtype=np.int64)
        gem_groups = np.asarray(obs["gem_group"][...], dtype=np.int64)
        umi_counts = np.asarray(obs["UMI_count"][...], dtype=np.float64)
        query_ids = _decode(source["var/gene_id"])

    control_indices = np.flatnonzero(gene_ids == "non-targeting")
    if control_indices.size != 1:
        raise AcquisitionError("expected exactly one non-targeting gene_id category")
    control_code = int(control_indices[0])
    is_control = gene_codes == control_code
    unique_gems, gem_cell_counts = np.unique(gem_groups, return_counts=True)
    control_by_gem = np.asarray(
        [np.sum(is_control & (gem_groups == gem)) for gem in unique_gems]
    )
    target_ids = sorted(value for value in gene_ids.tolist() if value != "non-targeting")
    all_gene_counts = np.bincount(gene_codes, minlength=len(gene_ids))
    target_cell_counts = np.delete(all_gene_counts, control_code)
    action_gem_pairs = np.unique((gene_codes << 8) | gem_groups)

    action_roster = _write_lines(path.parent / "observed_action_roster.txt", target_ids)
    query_roster = _write_lines(
        path.parent / "expression_query_roster.txt", sorted(query_ids.tolist())
    )
    overlap: dict[str, object] | None = None
    if reference_path is not None:
        with np.load(reference_path, allow_pickle=False) as reference:
            current_actions = {str(value).split(".")[0] for value in reference["action_ids"]}
            current_queries = {str(value).split(".")[0] for value in reference["query_ids"]}
        observed_actions = set(target_ids)
        observed_queries = {value.split(".")[0] for value in query_ids.tolist()}
        overlap = {
            "referencePath": str(reference_path),
            "currentInterventionGenes": len(current_actions),
            "observedInterventionOverlap": len(current_actions & observed_actions),
            "missingCurrentInterventions": sorted(current_actions - observed_actions),
            "currentReadoutGenes": len(current_queries),
            "readoutOverlap": len(current_queries & observed_queries),
            "missingCurrentReadouts": len(current_queries - observed_queries),
        }

    return {
        "schema": "slp.nadig-hepg2-source-metadata-audit/v1",
        "source": {
            "url": SOURCE_URL,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "officialChecksumPublished": False,
        },
        "outcomeAccess": {
            "matrixValuesRead": False,
            "perturbationProfilesSummarized": False,
            "fittingOrScoringPerformed": False,
        },
        "matrixStructure": {
            "shape": [145_473, 9_624],
            "dtype": "float32",
            "storage": "contiguous-uncompressed-dense",
        },
        "observationFields": sorted(required_obs),
        "cells": int(gene_codes.size),
        "gemGroups": {
            "count": int(unique_gems.size),
            "minimumCells": int(gem_cell_counts.min()),
            "medianCells": float(np.median(gem_cell_counts)),
            "maximumCells": int(gem_cell_counts.max()),
        },
        "controls": {
            "identity": "non-targeting",
            "cells": int(is_control.sum()),
            "presentGemGroups": int(np.sum(control_by_gem > 0)),
            "minimumCellsPerGemGroup": int(control_by_gem.min()),
            "medianCellsPerGemGroup": float(np.median(control_by_gem)),
            "maximumCellsPerGemGroup": int(control_by_gem.max()),
        },
        "targets": {
            "observedEnsemblGenes": len(target_ids),
            "minimumCells": int(target_cell_counts.min()),
            "medianCells": float(np.median(target_cell_counts)),
            "maximumCells": int(target_cell_counts.max()),
            "observedActionGemPairsIncludingControls": int(action_gem_pairs.size),
        },
        "queries": {
            "ensemblGenes": int(query_ids.size),
            "uniqueEnsemblGenes": len(set(query_ids.tolist())),
        },
        "umiCountMetadata": {
            "finite": bool(np.isfinite(umi_counts).all()),
            "minimum": float(umi_counts.min()),
            "median": float(np.median(umi_counts)),
            "maximum": float(umi_counts.max()),
        },
        "rosters": {"actions": action_roster, "queries": query_roster},
        "currentDevelopmentOverlap": overlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-development", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    source_path = args.output_dir / SOURCE_NAME
    if args.no_download:
        if not source_path.is_file():
            raise AcquisitionError(f"source does not exist: {source_path}")
        if source_path.stat().st_size != EXPECTED_BYTES:
            raise AcquisitionError("existing source has the wrong byte count")
        if sha256_file(source_path) != EXPECTED_SHA256:
            raise AcquisitionError("existing source has the wrong SHA-256")
    else:
        acquire(source_path)
    reference = args.reference_development
    report = audit_metadata(source_path, reference if reference.is_file() else None)
    report_path = args.output_dir / "h5ad-metadata-audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
