#!/usr/bin/env python3
"""Build a control-only HepG2 normalization artifact from the pinned H5AD.

The HDF5 expression matrix is indexed only at rows whose ``gene_id`` category
is exactly ``non-targeting``.  No perturbed expression row is read.  The
resulting transform is an SLp control-anchored value space, not the Nadig et
al. DESeq2 differential-expression endpoint.
"""

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

SOURCE_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
REPLOGLE_CODE_COMMIT = "3b25109aeb9c0c2026bd70abd50304a0ad4e5395"
DEFAULT_SOURCE = ROOT / (
    "data/sources/nadig-2025-gse264667-hepg2-v1/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad"
)
DEFAULT_OUTPUT = ROOT / (
    "data/derived/slp11-human/nadig-hepg2-control-normalization-v1"
)


class ControlArtifactError(RuntimeError):
    """Raised when the pinned source or its control-only contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[...]).astype(str)


def build(source_path: Path, output_dir: Path) -> dict[str, object]:
    """Read non-targeting controls only and write frozen normalization data."""

    if _sha256(source_path) != SOURCE_SHA256:
        raise ControlArtifactError("HepG2 source SHA-256 does not match the pin")
    with h5py.File(source_path, "r") as source:
        categories = _decode(source["obs/__categories/gene_id"])
        codes = np.asarray(source["obs/gene_id"][...], dtype=np.int64)
        matches = np.flatnonzero(categories == "non-targeting")
        if matches.size != 1:
            raise ControlArtifactError("expected one non-targeting gene_id category")
        control_code = int(matches[0])
        control_rows = np.flatnonzero(codes == control_code)
        if control_rows.size != 4_976 or np.any(codes[control_rows] != control_code):
            raise ControlArtifactError("non-targeting control roster drifted")
        all_depth = np.asarray(source["obs/UMI_count"][...], dtype=np.float64)
        all_groups = np.asarray(source["obs/gem_group"][...], dtype=np.int64)
        depth = all_depth[control_rows]
        groups = all_groups[control_rows]
        query_ids = _decode(source["var/gene_id"])

        # This is the sole X access.  HDF5 receives the sorted exact control
        # indices; no surrounding or perturbed row is loaded.
        raw = np.asarray(source["X"][control_rows, :], dtype=np.float64)

    if raw.shape != (4_976, 9_624) or query_ids.shape != (9_624,):
        raise ControlArtifactError("HepG2 control/query dimensions drifted")
    model = fit_control_normalizer(raw, depth, groups)
    basal = control_basal_expression(raw, depth)
    row_sums = raw.sum(axis=1)
    difference = depth - row_sums

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "control-normalization.npz"
    np.savez_compressed(
        artifact_path,
        schema=np.asarray("slp.nadig-hepg2-control-normalization/v1"),
        source_sha256=np.asarray(SOURCE_SHA256),
        value_space=np.asarray(VALUE_SPACE),
        basal_value_space=np.asarray(BASAL_VALUE_SPACE),
        fit_provenance=np.asarray(model.fit_provenance),
        author_endpoint_equivalent=np.asarray(False),
        query_ids=query_ids,
        gem_groups=model.gem_groups_,
        target_umi=np.asarray(model.target_umi_, dtype=np.float64),
        control_mean=model.control_mean_.astype(np.float32),
        control_std=model.control_std_.astype(np.float32),
        control_observed=model.control_observed_,
        control_counts=model.control_counts_,
        context_basal_expression=basal.astype(np.float32),
    )
    artifact_hash = _sha256(artifact_path)
    report: dict[str, object] = {
        "schema": "slp.nadig-hepg2-control-normalization-report/v1",
        "source": {
            "path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": SOURCE_SHA256,
        },
        "expressionAccess": {
            "rowsRead": "non-targeting controls only",
            "controlRows": int(raw.shape[0]),
            "perturbedRowsRead": 0,
            "perturbedProfilesSummarized": False,
        },
        "rawCountAudit": {
            "allFiniteNonnegativeIntegers": bool(
                np.isfinite(raw).all()
                and np.all(raw >= 0.0)
                and np.all(raw == np.floor(raw))
            ),
            "selectedPanelRowSum": {
                "minimum": float(row_sums.min()),
                "median": float(np.median(row_sums)),
                "maximum": float(row_sums.max()),
            },
            "fullUmiCount": {
                "minimum": float(depth.min()),
                "median": float(np.median(depth)),
                "maximum": float(depth.max()),
            },
            "fullUmiMinusSelectedPanel": {
                "minimum": float(difference.min()),
                "maximum": float(difference.max()),
                "allStrictlyPositive": bool(np.all(difference > 0.0)),
            },
        },
        "normalization": {
            "valueSpace": VALUE_SPACE,
            "targetUmi": model.target_umi_,
            "formula": (
                "cell counts * median(control full UMI) / cell full UMI; "
                "then per-GEM/query control mean subtraction and sample-SD division"
            ),
            "logTransform": False,
            "clipping": False,
            "standardDeviationDof": 1,
            "gemGroups": int(model.gem_groups_.size),
            "controlCellsPerGemMinimum": int(model.control_counts_.min()),
            "controlCellsPerGemMaximum": int(model.control_counts_.max()),
            "supportedGemQueryFraction": float(model.control_observed_.mean()),
            "fitUsesPerturbedExpression": False,
            "formulaProvenance": {
                "repository": "https://github.com/thomasmaxwellnorman/Perturbseq_GI",
                "commit": REPLOGLE_CODE_COMMIT,
                "notebook": (
                    "GI_generate_populations.ipynb calls normalize_to_gemgroup_control"
                ),
                "implementation": (
                    "perturbseq/expression_normalization.py: "
                    "normalize_to_gemgroup_control and normalize_matrix_to_control"
                ),
            },
        },
        "basalContext": {
            "valueSpace": BASAL_VALUE_SPACE,
            "formula": "mean over controls of log2(1 + 10000 * count / full UMI)",
        },
        "compatibility": {
            "reploglePinnedTransform": (
                "same linear-UMI scaling and per-GEM control z-score formula"
            ),
            "replogleControlRosterEquivalent": False,
            "reasonControlRosterNotEquivalent": (
                "Nadig marks non-targeting cells but supplies no Replogle core-control flag"
            ),
            "nadigAuthorEndpointEquivalent": False,
            "reasonAuthorEndpointNotEquivalent": (
                "Nadig sums per-condition/per-GEM counts and fits DESeq2 median-of-ratios "
                "negative-binomial models with fixed GEM effects to report log2 fold change/SE"
            ),
            "existingSlpContextDescriptorEquivalent": False,
            "reasonExistingDescriptorNotEquivalent": (
                "the existing K562/RPE1 adapter used a selected-panel denominator; this "
                "artifact uses the required original full-cell UMI denominator"
            ),
        },
        "artifact": {
            "path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": artifact_path.stat().st_size,
            "sha256": artifact_hash,
        },
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
