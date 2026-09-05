#!/usr/bin/env python3
"""Acquire and metadata-audit the official Nadig 2025 Jurkat H5AD.

The audit never indexes ``X``.  It reads only HDF5 structure, ``obs``
metadata, and ``var/gene_id``.  Perturbed expression therefore remains
unopened until a separately frozen protocol authorizes its use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
    "GSE264667_jurkat_raw_singlecell_01.h5ad"
)
SOURCE_NAME = "GSE264667_jurkat_raw_singlecell_01.h5ad"
EXPECTED_BYTES = 9_366_490_264
EXPECTED_SHA256 = "ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3"
EXPECTED_SHAPE = (262_956, 8_882)
DEFAULT_OUTPUT = Path("data/sources/nadig-2025-gse264667-jurkat-v1")
DEFAULT_REFERENCE = Path(
    "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
)
ENSG_RE = re.compile(r"^ENSG[0-9]+$")


class AcquisitionError(RuntimeError):
    """Raised when the pinned source or metadata contract differs."""


def sha256_file(path: Path) -> str:
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
                "3",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                "900",
                SOURCE_URL,
                "-o",
                str(path),
            ],
            check=True,
        )
    if path.stat().st_size != EXPECTED_BYTES:
        raise AcquisitionError(
            f"source has {path.stat().st_size} bytes; expected {EXPECTED_BYTES}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SHA256:
        raise AcquisitionError(
            f"source SHA-256 is {actual_hash}; expected {EXPECTED_SHA256}"
        )


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[...]).astype(str)


def _categorical(group: h5py.Group, name: str) -> np.ndarray:
    categories = _decode(group[f"__categories/{name}"])
    codes = np.asarray(group[name][...], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise AcquisitionError(f"obs/{name} has invalid categorical codes")
    return categories[codes]


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
        if matrix.shape != EXPECTED_SHAPE or matrix.dtype != np.dtype("float32"):
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
        gene_ids = _categorical(obs, "gene_id")
        populations = _categorical(obs, "gene_transcript")
        constructs = _categorical(obs, "sgID_AB")
        transcripts = _categorical(obs, "transcript")
        gem_groups = np.asarray(obs["gem_group"][...], dtype=np.int64)
        umi_counts = np.asarray(obs["UMI_count"][...], dtype=np.float64)
        query_ids = _decode(source["var/gene_id"])

    stable = np.asarray([ENSG_RE.fullmatch(value) is not None for value in gene_ids])
    controls = gene_ids == "non-targeting"
    unresolved = ~(stable | controls)
    unique_gems, gem_cell_counts = np.unique(gem_groups, return_counts=True)
    control_by_gem = np.asarray(
        [np.sum(controls & (gem_groups == gem)) for gem in unique_gems]
    )
    stable_ids = sorted(set(gene_ids[stable].tolist()))
    population_identity: dict[str, tuple[str, str, str]] = {}
    conflicting_populations: set[str] = set()
    for action, population, construct, transcript in zip(
        gene_ids[stable], populations[stable], constructs[stable], transcripts[stable]
    ):
        identity = (action, construct, transcript)
        previous = population_identity.setdefault(population, identity)
        if previous != identity:
            conflicting_populations.add(population)
    population_ids = sorted(population_identity)

    action_roster = _write_lines(path.parent / "observed_action_roster.txt", stable_ids)
    query_roster = _write_lines(
        path.parent / "expression_query_roster.txt", sorted(query_ids.tolist())
    )
    overlap: dict[str, object] | None = None
    if reference_path is not None:
        with np.load(reference_path, allow_pickle=False) as reference:
            current_actions = {str(value).split(".")[0] for value in reference["action_ids"]}
            current_queries = {str(value).split(".")[0] for value in reference["query_ids"]}
        observed_actions = set(stable_ids)
        observed_queries = {value.split(".")[0] for value in query_ids.tolist()}
        missing_actions = sorted(current_actions - observed_actions)
        missing_action_payload = "".join(f"{value}\n" for value in missing_actions).encode()
        overlap = {
            "referencePath": str(reference_path),
            "currentInterventionGenes": len(current_actions),
            "observedInterventionOverlap": len(current_actions & observed_actions),
            "missingCurrentInterventions": len(missing_actions),
            "missingCurrentInterventionIdsSha256": hashlib.sha256(
                missing_action_payload
            ).hexdigest(),
            "currentReadoutGenes": len(current_queries),
            "readoutOverlap": len(current_queries & observed_queries),
            "missingCurrentReadouts": len(current_queries - observed_queries),
        }

    stable_cell_counts = np.unique(gene_ids[stable], return_counts=True)[1]
    return {
        "schema": "slp.nadig-jurkat-source-metadata-audit/v1",
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
            "controlExpressionRead": False,
            "fittingOrScoringPerformed": False,
        },
        "matrixStructure": {
            "shape": list(EXPECTED_SHAPE),
            "dtype": "float32",
            "storage": "contiguous-uncompressed-dense",
        },
        "cells": int(gene_ids.size),
        "gemGroups": {
            "count": int(unique_gems.size),
            "minimumCells": int(gem_cell_counts.min()),
            "medianCells": float(np.median(gem_cell_counts)),
            "maximumCells": int(gem_cell_counts.max()),
        },
        "controls": {
            "identity": "non-targeting",
            "cells": int(controls.sum()),
            "presentGemGroups": int(np.count_nonzero(control_by_gem)),
            "minimumCellsPerGemGroup": int(control_by_gem.min()),
            "medianCellsPerGemGroup": float(np.median(control_by_gem)),
            "maximumCellsPerGemGroup": int(control_by_gem.max()),
        },
        "targets": {
            "stableEnsemblGenes": len(stable_ids),
            "stableTargetCells": int(stable.sum()),
            "unresolvedTargetCellsQuarantined": int(unresolved.sum()),
            "exactGeneTranscriptPopulations": len(population_ids),
            "populationIdentityConflicts": len(conflicting_populations),
            "minimumCellsPerGene": int(stable_cell_counts.min()),
            "medianCellsPerGene": float(np.median(stable_cell_counts)),
            "maximumCellsPerGene": int(stable_cell_counts.max()),
        },
        "queries": {
            "ensemblGenes": int(query_ids.size),
            "uniqueEnsemblGenes": len(set(query_ids.tolist())),
        },
        "umiCountMetadata": {
            "finite": bool(np.isfinite(umi_counts).all()),
            "positive": bool(np.all(umi_counts > 0.0)),
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
