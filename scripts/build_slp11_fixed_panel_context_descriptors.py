#!/usr/bin/env python3
"""Build fixed-panel control descriptors for three Replogle contexts and HepG2.

The panel is selected from source identity metadata only: it is the exact
intersection of the frozen 7,036 fitting-complete target queries and all four
raw control source gene rosters. Both numerator and denominator use this same
panel. Only control expression rows are read.
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
sys.path.insert(0, str(ROOT / "scripts"))

from build_slp11_common_context_descriptors import (
    CONTROL_SUFFIX,
    HEPG2_SHA256,
    REPLOGLE_CODE_COMMIT,
    REPLOGLE_SOURCES,
    _copy_snapshot_with_context,
    _decode,
    _sha256,
)
from context_descriptor import (
    FIXED_PANEL_VALUE_SPACE,
    pooled_control_fixed_panel_log2_cp10k,
)

INPUT_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
PRIOR_UNPROVEN_SHA256 = "4f9dcf74f6c84d980072121ae062ea7dcc35148a5aca1550d4e41c0e8e529d2f"
DEFAULT_INPUT = ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
DEFAULT_HEPG2 = ROOT / (
    "data/sources/nadig-2025-gse264667-hepg2-v1/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad"
)
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1"


class FixedPanelBuildError(RuntimeError):
    """Raised when a source or fixed-panel identity contract drifts."""


def _source_gene_roster(path: Path, expected_hash: str) -> np.ndarray:
    if _sha256(path) != expected_hash:
        raise FixedPanelBuildError(f"source SHA-256 drift: {path.name}")
    with h5py.File(path, "r") as source:
        genes = _decode(source["var/gene_id"])
    if len(set(genes.tolist())) != genes.size:
        raise FixedPanelBuildError(f"duplicate source query IDs: {path.name}")
    return genes


def _fixed_panel(
    query_ids: np.ndarray,
    source_dir: Path,
    hepg2_path: Path,
) -> tuple[np.ndarray, bytes, dict[str, object]]:
    roster_sets = []
    source_rosters = []
    for spec in REPLOGLE_SOURCES:
        genes = _source_gene_roster(source_dir / spec.name, spec.sha256)
        roster_sets.append(set(genes.tolist()))
        payload = "".join(f"{gene}\n" for gene in genes).encode("ascii")
        source_rosters.append(
            {
                "source": spec.name,
                "genes": int(genes.size),
                "depositedOrderRosterSha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    hepg2_genes = _source_gene_roster(hepg2_path, HEPG2_SHA256)
    roster_sets.append(set(hepg2_genes.tolist()))
    hepg2_payload = "".join(f"{gene}\n" for gene in hepg2_genes).encode("ascii")
    source_rosters.append(
        {
            "source": hepg2_path.name,
            "genes": int(hepg2_genes.size),
            "depositedOrderRosterSha256": hashlib.sha256(hepg2_payload).hexdigest(),
        }
    )
    common = set(query_ids.tolist()).intersection(*roster_sets)
    mask = np.asarray([gene in common for gene in query_ids], dtype=np.bool_)
    roster = query_ids[mask]
    payload = "".join(f"{gene}\n" for gene in roster).encode("ascii")
    return mask, payload, {
        "startingFittingCompleteQueries": int(query_ids.size),
        "fixedPanelQueries": int(mask.sum()),
        "excludedContextTokens": int((~mask).sum()),
        "ordering": "frozen fitting-complete query order",
        "rosterBytes": len(payload),
        "rosterSha256": hashlib.sha256(payload).hexdigest(),
        "sourceRosters": source_rosters,
        "selectionUsesExpressionValues": False,
    }


def _aligned_replogle_controls(
    source_dir: Path,
    spec: object,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = source_dir / spec.name
    if _sha256(path) != spec.sha256:
        raise FixedPanelBuildError(f"source SHA-256 drift: {spec.name}")
    with h5py.File(path, "r") as source:
        records = _decode(source["obs/gene_transcript"])
        core = np.asarray(source["obs/core_control"][...], dtype=np.bool_)
        rows = np.flatnonzero(core & np.char.endswith(records, CONTROL_SUFFIX))
        weights = np.asarray(
            source["obs/num_cells_filtered"][...], dtype=np.float64
        )[rows]
        genes = _decode(source["var/gene_id"])
        raw_means = np.asarray(source["X"][rows, :], dtype=np.float64)
    lookup = {gene: index for index, gene in enumerate(genes)}
    aligned = raw_means[:, np.asarray([lookup[gene] for gene in query_ids], dtype=np.int64)]
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0):
        raise FixedPanelBuildError(f"invalid Replogle controls: {spec.name}")
    return aligned, weights, {
        "contextId": spec.context_id,
        "source": f"data/sources/human/{spec.name}",
        "sha256": spec.sha256,
        "controlPopulations": int(rows.size),
        "filteredControlCells": int(weights.sum()),
        "depositedRows": "arithmetic means over filtered member cells",
        "pooledTotals": "stored mean * num_cells_filtered",
    }


def _aligned_hepg2_controls(
    path: Path,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if _sha256(path) != HEPG2_SHA256:
        raise FixedPanelBuildError("HepG2 source SHA-256 drift")
    with h5py.File(path, "r") as source:
        categories = _decode(source["obs/__categories/gene_id"])
        codes = np.asarray(source["obs/gene_id"][...], dtype=np.int64)
        control_codes = np.flatnonzero(categories == "non-targeting")
        if control_codes.size != 1:
            raise FixedPanelBuildError("HepG2 control identity drift")
        rows = np.flatnonzero(codes == int(control_codes[0]))
        if rows.size != 4_976 or np.any(codes[rows] != int(control_codes[0])):
            raise FixedPanelBuildError("HepG2 control roster drift")
        genes = _decode(source["var/gene_id"])
        raw_cells = np.asarray(source["X"][rows, :], dtype=np.float64)
    lookup = {gene: index for index, gene in enumerate(genes)}
    present = np.asarray([gene in lookup for gene in query_ids], dtype=np.bool_)
    aligned = np.zeros((rows.size, query_ids.size), dtype=np.float64)
    aligned[:, present] = raw_cells[
        :, np.asarray([lookup[gene] for gene in query_ids[present]], dtype=np.int64)
    ]
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0):
        raise FixedPanelBuildError("invalid HepG2 controls")
    return aligned, np.ones(rows.size, dtype=np.int64), {
        "contextId": "nadig-2025-hepg2-day-7",
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": HEPG2_SHA256,
        "controlCells": int(rows.size),
        "perturbedExpressionRowsRead": 0,
    }


def _write_hepg2(
    path: Path,
    query_ids: np.ndarray,
    descriptor: np.ndarray,
    observed: np.ndarray,
    panel_hash: str,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("slp.unseen-context-fixed-panel-control-descriptor/v1"),
        context_ids=np.asarray(["nadig-2025-hepg2-day-7"]),
        query_ids=query_ids,
        context_basal_expression=descriptor[None, :].astype(np.float32),
        context_basal_observed=observed[None, :],
        context_value_space=np.asarray(FIXED_PANEL_VALUE_SPACE),
        fixed_panel_query_sha256=np.asarray(panel_hash),
        source_sha256=np.asarray(HEPG2_SHA256),
        perturbed_expression_rows_read=np.asarray(0, dtype=np.int64),
    )


def build(
    input_path: Path,
    source_dir: Path,
    hepg2_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the fixed-panel context descriptors and immutable snapshot."""

    if _sha256(input_path) != INPUT_SHA256:
        raise FixedPanelBuildError("fitting-complete development SHA-256 drift")
    with np.load(input_path, allow_pickle=False) as source:
        query_ids = source["query_ids"]
        context_ids = source["context_ids"]
    if query_ids.shape != (7_036,) or context_ids.tolist() != [
        spec.context_id for spec in REPLOGLE_SOURCES
    ]:
        raise FixedPanelBuildError("fitting-complete identity contract drift")

    panel_mask, roster_payload, panel_report = _fixed_panel(
        query_ids, source_dir, hepg2_path
    )
    if int(panel_mask.sum()) != 6_789:
        raise FixedPanelBuildError("fixed shared-panel size drift")
    panel_hash = hashlib.sha256(roster_payload).hexdigest()

    descriptors = []
    masks = []
    replogle_reports = []
    for spec in REPLOGLE_SOURCES:
        raw, weights, source_report = _aligned_replogle_controls(
            source_dir, spec, query_ids
        )
        descriptor, observed = pooled_control_fixed_panel_log2_cp10k(
            raw, weights, panel_mask
        )
        descriptors.append(descriptor)
        masks.append(observed)
        replogle_reports.append(source_report)
    descriptor_matrix = np.stack(descriptors)
    mask_matrix = np.stack(masks)

    hepg2_raw, hepg2_weights, hepg2_report = _aligned_hepg2_controls(
        hepg2_path, query_ids
    )
    hepg2_descriptor, hepg2_observed = pooled_control_fixed_panel_log2_cp10k(
        hepg2_raw, hepg2_weights, panel_mask
    )
    if not np.array_equal(mask_matrix, np.broadcast_to(panel_mask, mask_matrix.shape)):
        raise FixedPanelBuildError("Replogle context masks differ from fixed panel")
    if not np.array_equal(hepg2_observed, panel_mask):
        raise FixedPanelBuildError("HepG2 context mask differs from fixed panel")

    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = output_dir / "fixed-context-query-ids.txt"
    roster_path.write_bytes(roster_payload)
    development_path = output_dir / (
        "replogle-k562-rpe1-gwps-complete-panel-development-"
        "v2-fixed-control-context.npz"
    )
    unchanged_hashes = _copy_snapshot_with_context(
        input_path,
        development_path,
        descriptor_matrix,
        mask_matrix,
        value_space=FIXED_PANEL_VALUE_SPACE,
    )
    hepg2_output = output_dir / "nadig-hepg2-fixed-panel-control-context-v1.npz"
    _write_hepg2(
        hepg2_output,
        query_ids,
        hepg2_descriptor,
        hepg2_observed,
        panel_hash,
    )

    report: dict[str, object] = {
        "schema": "slp.fixed-panel-common-context-build-report/v1",
        "valueSpace": FIXED_PANEL_VALUE_SPACE,
        "formula": (
            "log2(1 + 10000 * pooled control query count / pooled control "
            "count summed over the exact 6789-gene fixed panel)"
        ),
        "panel": panel_report,
        "inputDevelopment": {
            "path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": INPUT_SHA256,
            "targetQueriesRetained": int(query_ids.size),
        },
        "replogleSources": replogle_reports,
        "replogleAggregateSemantics": {
            "pinnedRepository": "https://github.com/thomasmaxwellnorman/Perturbseq_GI",
            "commit": REPLOGLE_CODE_COMMIT,
            "method": "CellPopulation.average stores arithmetic expression means",
        },
        "supersedes": {
            "artifactSha256": PRIOR_UNPROVEN_SHA256,
            "claim": "cross-source full-library denominator compatibility",
            "reason": (
                "Replogle raw_bulk exposes only 8248-8749 genes and no original "
                "filtered-cell full UMI metadata or proof that every omitted gene is zero"
            ),
            "priorArtifactPreserved": True,
        },
        "hepg2": hepg2_report,
        "outputs": {
            "fixedPanelRoster": {
                "path": str(roster_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": roster_path.stat().st_size,
                "sha256": panel_hash,
            },
            "development": {
                "path": str(development_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": development_path.stat().st_size,
                "sha256": _sha256(development_path),
                "unchangedNpyPayloadSha256": unchanged_hashes,
                "all7036TargetArraysAndSplitsUnchanged": True,
            },
            "hepg2": {
                "path": str(hepg2_output.relative_to(ROOT)).replace("\\", "/"),
                "bytes": hepg2_output.stat().st_size,
                "sha256": _sha256(hepg2_output),
            },
        },
        "expressionAccess": {
            "replogle": "core-control aggregate rows only",
            "hepg2": "non-targeting single-cell rows only",
            "hepg2PerturbedRowsRead": 0,
        },
        "trainingPerformed": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/sources/human")
    parser.add_argument("--hepg2", type=Path, default=DEFAULT_HEPG2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(
        args.input.resolve(),
        args.source_dir.resolve(),
        args.hepg2.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
