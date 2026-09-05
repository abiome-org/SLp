#!/usr/bin/env python3
"""Build the frozen control-only Jurkat per-GEM normalization artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from control_normalization import (
    BASAL_VALUE_SPACE,
    VALUE_SPACE,
    control_basal_expression,
    fit_control_normalizer,
)

SOURCE_SHA256 = "ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3"
SOURCE_SHAPE = (262_956, 8_882)
CONTROL_ROWS = 12_013
DEFAULT_SOURCE = ROOT / (
    "data/sources/nadig-2025-gse264667-jurkat-v1/"
    "GSE264667_jurkat_raw_singlecell_01.h5ad"
)
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-human/nadig-jurkat-control-normalization-v1"


class ControlArtifactError(RuntimeError):
    """Raised when the Jurkat control-only source contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[...]).astype(str)


def _read_controls(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read metadata and exactly the non-targeting expression rows."""

    if _sha256(path) != SOURCE_SHA256:
        raise ControlArtifactError("Jurkat source SHA-256 does not match the pin")
    with h5py.File(path, "r") as source:
        if tuple(source["X"].shape) != SOURCE_SHAPE:
            raise ControlArtifactError("Jurkat source matrix shape drifted")
        categories = _decode(source["obs/__categories/gene_id"])
        codes = np.asarray(source["obs/gene_id"][...], dtype=np.int64)
        matches = np.flatnonzero(categories == "non-targeting")
        if matches.size != 1:
            raise ControlArtifactError("expected one non-targeting gene_id category")
        control_code = int(matches[0])
        rows = np.flatnonzero(codes == control_code)
        if rows.size != CONTROL_ROWS or np.any(codes[rows] != control_code):
            raise ControlArtifactError("Jurkat non-targeting control roster drifted")
        all_depth = np.asarray(source["obs/UMI_count"][...], dtype=np.float64)
        all_groups = np.asarray(source["obs/gem_group"][...], dtype=np.int64)
        query_ids = _decode(source["var/gene_id"])
        # This is the only X access: an exact sorted non-targeting row index.
        raw = np.asarray(source["X"][rows, :], dtype=np.float64)
    return raw, all_depth[rows], all_groups[rows], query_ids


def build(source_path: Path, output_dir: Path) -> dict[str, object]:
    raw, depth, groups, query_ids = _read_controls(source_path)
    if raw.shape != (CONTROL_ROWS, SOURCE_SHAPE[1]) or query_ids.shape != (
        SOURCE_SHAPE[1],
    ):
        raise ControlArtifactError("Jurkat control/query dimensions drifted")
    model = fit_control_normalizer(raw, depth, groups)
    basal = control_basal_expression(raw, depth)
    row_sums = raw.sum(axis=1)
    difference = depth - row_sums

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "control-normalization.npz"
    np.savez_compressed(
        artifact_path,
        schema=np.asarray("slp.nadig-jurkat-control-normalization/v1"),
        source_sha256=np.asarray(SOURCE_SHA256),
        value_space=np.asarray(VALUE_SPACE),
        basal_value_space=np.asarray(BASAL_VALUE_SPACE),
        fit_provenance=np.asarray(model.fit_provenance),
        author_endpoint_equivalent=np.asarray(False),
        query_ids=query_ids,
        gem_groups=model.gem_groups_,
        target_umi=np.asarray(model.target_umi_),
        control_mean=model.control_mean_,
        control_std=model.control_std_,
        control_observed=model.control_observed_,
        control_counts=model.control_counts_,
        context_basal_expression=basal.astype(np.float32),
    )
    report = {
        "schema": "slp.nadig-jurkat-control-normalization-report/v1",
        "source": {
            "path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": SOURCE_SHA256,
        },
        "expressionAccess": {
            "rowsRead": "non-targeting controls only",
            "controlRows": CONTROL_ROWS,
            "perturbedRowsRead": 0,
            "perturbedProfilesSummarized": False,
        },
        "normalization": {
            "formula": (
                "cell counts * median(control full UMI) / cell full UMI; then "
                "per-GEM/query control mean subtraction and sample-SD division"
            ),
            "valueSpace": VALUE_SPACE,
            "targetUmi": model.target_umi_,
            "gemGroups": int(model.gem_groups_.size),
            "controlCellsPerGemMinimum": int(model.control_counts_.min()),
            "controlCellsPerGemMedian": float(np.median(model.control_counts_)),
            "controlCellsPerGemMaximum": int(model.control_counts_.max()),
            "standardDeviationDof": 1,
            "logTransform": False,
            "clipping": False,
            "fitUsesPerturbedExpression": False,
            "supportedGemQueryFraction": float(model.control_observed_.mean()),
        },
        "rawCountAudit": {
            "allFiniteNonnegativeIntegers": bool(
                np.isfinite(raw).all()
                and np.all(raw >= 0.0)
                and np.all(raw == np.floor(raw))
            ),
            "fullUmiMinusSelectedPanel": {
                "minimum": float(difference.min()),
                "maximum": float(difference.max()),
                "allNonnegative": bool(np.all(difference >= 0.0)),
            },
        },
        "artifact": {
            "path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
        },
        "trainingPerformed": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
