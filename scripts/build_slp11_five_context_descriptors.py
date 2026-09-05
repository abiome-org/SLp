#!/usr/bin/env python3
"""Build one fixed-panel basal descriptor shared by five assay contexts.

Panel membership uses only query identity metadata.  Expression access is
restricted to Replogle core-control aggregate rows and Nadig non-targeting
single-cell rows.  All existing development arrays other than the three basal
descriptor fields are copied as exact NPY payloads.
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
HEPG2_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
JURKAT_SHA256 = "ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3"
PANEL_SHA256 = "1a863ba69f514ba9c1f3752cebde707af4a54ecbf1b54000d7e1320207838a79"
PANEL_QUERIES = 6_517
DEFAULT_INPUT = ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
DEFAULT_HEPG2 = ROOT / (
    "data/sources/nadig-2025-gse264667-hepg2-v1/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad"
)
DEFAULT_JURKAT = ROOT / (
    "data/sources/nadig-2025-gse264667-jurkat-v1/"
    "GSE264667_jurkat_raw_singlecell_01.h5ad"
)
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-human-gwps-five-context-v1"


class FiveContextBuildError(RuntimeError):
    """Raised when a pinned source or five-context contract drifts."""


def _source_gene_roster(path: Path, expected_hash: str) -> np.ndarray:
    if _sha256(path) != expected_hash:
        raise FiveContextBuildError(f"source SHA-256 drift: {path.name}")
    with h5py.File(path, "r") as source:
        genes = _decode(source["var/gene_id"])
    if len(set(genes.tolist())) != genes.size:
        raise FiveContextBuildError(f"duplicate source query IDs: {path.name}")
    return genes


def _fixed_panel(
    query_ids: np.ndarray,
    source_dir: Path,
    hepg2_path: Path,
    jurkat_path: Path,
) -> tuple[np.ndarray, bytes, list[dict[str, object]]]:
    specs = [(source_dir / spec.name, spec.sha256) for spec in REPLOGLE_SOURCES]
    specs.extend([(hepg2_path, HEPG2_SHA256), (jurkat_path, JURKAT_SHA256)])
    rosters = []
    reports = []
    for path, expected_hash in specs:
        genes = _source_gene_roster(path, expected_hash)
        rosters.append(set(genes.tolist()))
        payload = "".join(f"{gene}\n" for gene in genes).encode("ascii")
        reports.append(
            {
                "source": path.name,
                "genes": int(genes.size),
                "depositedOrderRosterSha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    common = set(query_ids.tolist()).intersection(*rosters)
    mask = np.asarray([gene in common for gene in query_ids], dtype=np.bool_)
    payload = "".join(f"{gene}\n" for gene in query_ids[mask]).encode("ascii")
    if int(mask.sum()) != PANEL_QUERIES or hashlib.sha256(payload).hexdigest() != PANEL_SHA256:
        raise FiveContextBuildError("five-context fixed-panel identity drift")
    return mask, payload, reports


def _aligned_replogle_controls(
    source_dir: Path, spec: object, query_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = source_dir / spec.name
    if _sha256(path) != spec.sha256:
        raise FiveContextBuildError(f"source SHA-256 drift: {spec.name}")
    with h5py.File(path, "r") as source:
        records = _decode(source["obs/gene_transcript"])
        core = np.asarray(source["obs/core_control"][...], dtype=np.bool_)
        rows = np.flatnonzero(core & np.char.endswith(records, CONTROL_SUFFIX))
        weights = np.asarray(source["obs/num_cells_filtered"][...], dtype=np.float64)[rows]
        genes = _decode(source["var/gene_id"])
        raw_means = np.asarray(source["X"][rows, :], dtype=np.float64)
    lookup = {gene: index for index, gene in enumerate(genes)}
    aligned = raw_means[:, np.asarray([lookup[gene] for gene in query_ids], dtype=np.int64)]
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0):
        raise FiveContextBuildError(f"invalid Replogle controls: {spec.name}")
    return aligned, weights, {
        "contextId": spec.context_id,
        "source": f"data/sources/human/{spec.name}",
        "sha256": spec.sha256,
        "controlPopulations": int(rows.size),
        "filteredControlCells": int(weights.sum()),
        "expressionRowsRead": int(rows.size),
        "rowPolicy": "core_control and exact non-targeting suffix",
    }


def _aligned_nadig_controls(
    path: Path,
    expected_hash: str,
    expected_rows: int,
    expected_shape: tuple[int, int],
    context_id: str,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Read exactly non-targeting rows and align them to the 7,036-query axis."""

    if _sha256(path) != expected_hash:
        raise FiveContextBuildError(f"source SHA-256 drift: {path.name}")
    with h5py.File(path, "r") as source:
        if tuple(source["X"].shape) != expected_shape:
            raise FiveContextBuildError(f"source matrix shape drift: {path.name}")
        categories = _decode(source["obs/__categories/gene_id"])
        codes = np.asarray(source["obs/gene_id"][...], dtype=np.int64)
        matches = np.flatnonzero(categories == "non-targeting")
        if matches.size != 1:
            raise FiveContextBuildError(f"non-targeting identity drift: {path.name}")
        control_code = int(matches[0])
        rows = np.flatnonzero(codes == control_code)
        if rows.size != expected_rows or np.any(codes[rows] != control_code):
            raise FiveContextBuildError(f"non-targeting row roster drift: {path.name}")
        genes = _decode(source["var/gene_id"])
        # The sole X access uses the exact sorted non-targeting row index.
        raw_cells = np.asarray(source["X"][rows, :], dtype=np.float64)
    lookup = {gene: index for index, gene in enumerate(genes)}
    present = np.asarray([gene in lookup for gene in query_ids], dtype=np.bool_)
    aligned = np.zeros((rows.size, query_ids.size), dtype=np.float64)
    aligned[:, present] = raw_cells[
        :, np.asarray([lookup[gene] for gene in query_ids[present]], dtype=np.int64)
    ]
    if not np.isfinite(aligned[:, present]).all() or np.any(aligned[:, present] < 0.0):
        raise FiveContextBuildError(f"invalid Nadig controls: {path.name}")
    return aligned, np.ones(rows.size, dtype=np.int64), {
        "contextId": context_id,
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": expected_hash,
        "controlCells": int(rows.size),
        "sourceQueries": int(genes.size),
        "alignedQueries": int(present.sum()),
        "expressionRowsRead": int(rows.size),
        "rowPolicy": "gene_id == non-targeting",
        "perturbedExpressionRowsRead": 0,
    }


def _write_descriptor(
    path: Path,
    context_id: str,
    query_ids: np.ndarray,
    descriptor: np.ndarray,
    observed: np.ndarray,
    source_hash: str,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("slp.unseen-context-five-context-panel-descriptor/v1"),
        context_ids=np.asarray([context_id]),
        query_ids=query_ids,
        context_basal_expression=descriptor[None, :].astype(np.float32),
        context_basal_observed=observed[None, :],
        context_value_space=np.asarray(FIXED_PANEL_VALUE_SPACE),
        fixed_panel_query_sha256=np.asarray(PANEL_SHA256),
        source_sha256=np.asarray(source_hash),
        perturbed_expression_rows_read=np.asarray(0, dtype=np.int64),
    )


def build(
    input_path: Path,
    source_dir: Path,
    hepg2_path: Path,
    jurkat_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if _sha256(input_path) != INPUT_SHA256:
        raise FiveContextBuildError("three-context development SHA-256 drift")
    with np.load(input_path, allow_pickle=False) as source:
        query_ids = source["query_ids"].astype(str)
        context_ids = source["context_ids"].astype(str)
    if query_ids.shape != (7_036,) or context_ids.tolist() != [
        spec.context_id for spec in REPLOGLE_SOURCES
    ]:
        raise FiveContextBuildError("three-context identity contract drift")

    panel_mask, roster_payload, roster_reports = _fixed_panel(
        query_ids, source_dir, hepg2_path, jurkat_path
    )
    descriptors = []
    masks = []
    source_reports = []
    for spec in REPLOGLE_SOURCES:
        raw, weights, source_report = _aligned_replogle_controls(source_dir, spec, query_ids)
        descriptor, observed = pooled_control_fixed_panel_log2_cp10k(
            raw, weights, panel_mask
        )
        descriptors.append(descriptor)
        masks.append(observed)
        source_reports.append(source_report)

    nadig_specs = [
        (hepg2_path, HEPG2_SHA256, 4_976, (145_473, 9_624), "nadig-2025-hepg2-day-7"),
        (jurkat_path, JURKAT_SHA256, 12_013, (262_956, 8_882), "nadig-2025-jurkat-day-7"),
    ]
    nadig_outputs = []
    for path, source_hash, rows, shape, context_id in nadig_specs:
        raw, weights, source_report = _aligned_nadig_controls(
            path, source_hash, rows, shape, context_id, query_ids
        )
        descriptor, observed = pooled_control_fixed_panel_log2_cp10k(
            raw, weights, panel_mask
        )
        descriptors.append(descriptor)
        masks.append(observed)
        source_reports.append(source_report)
        nadig_outputs.append((context_id, descriptor, observed, source_hash))

    descriptor_matrix = np.stack(descriptors)
    mask_matrix = np.stack(masks)
    if descriptor_matrix.shape != (5, 7_036) or not np.array_equal(
        mask_matrix, np.broadcast_to(panel_mask, mask_matrix.shape)
    ):
        raise FiveContextBuildError("five-context descriptor support drift")

    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = output_dir / "fixed-context-query-ids.txt"
    roster_path.write_bytes(roster_payload)
    development_path = output_dir / (
        "replogle-k562-rpe1-gwps-complete-panel-development-"
        "v3-five-context-control-panel.npz"
    )
    unchanged_hashes = _copy_snapshot_with_context(
        input_path,
        development_path,
        descriptor_matrix[:3],
        mask_matrix[:3],
        value_space=FIXED_PANEL_VALUE_SPACE,
    )
    output_records = {}
    for context_id, descriptor, observed, source_hash in nadig_outputs:
        short = "hepg2" if "hepg2" in context_id else "jurkat"
        path = output_dir / f"nadig-{short}-five-context-control-context-v1.npz"
        _write_descriptor(path, context_id, query_ids, descriptor, observed, source_hash)
        output_records[short] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "supportedQueries": int(observed.sum()),
        }

    report = {
        "schema": "slp.five-context-fixed-panel-build-report/v1",
        "hypothesis": (
            "all five sources support one identity-selected subset of the original "
            "7036-query axis and single-cell pooling equals equivalent aggregate pooling"
        ),
        "acceptanceRule": {
            "exactSupportedQueriesPerContext": PANEL_QUERIES,
            "perturbedExpressionRowsRead": 0,
            "unchangedDevelopmentNpyPayloads": True,
        },
        "valueSpace": FIXED_PANEL_VALUE_SPACE,
        "formula": (
            "log2(1 + 10000 * pooled control query count / pooled control count "
            "summed over the exact 6517-gene shared panel)"
        ),
        "denominator": "exact 6517-query shared panel; not original full-cell UMI",
        "panel": {
            "startingQueries": 7_036,
            "sharedQueries": PANEL_QUERIES,
            "maskedQueries": 7_036 - PANEL_QUERIES,
            "sha256": PANEL_SHA256,
            "selectionUsesExpressionValues": False,
            "sourceRosters": roster_reports,
        },
        "inputDevelopment": {
            "path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": INPUT_SHA256,
        },
        "sources": source_reports,
        "replogleAggregateSemantics": {
            "pinnedRepository": "https://github.com/thomasmaxwellnorman/Perturbseq_GI",
            "commit": REPLOGLE_CODE_COMMIT,
            "method": "CellPopulation.average stores arithmetic expression means",
        },
        "outputs": {
            "panelRoster": {
                "path": str(roster_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": roster_path.stat().st_size,
                "sha256": PANEL_SHA256,
            },
            "development": {
                "path": str(development_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": development_path.stat().st_size,
                "sha256": _sha256(development_path),
                "unchangedNpyPayloadSha256": unchanged_hashes,
                "onlyContextDescriptorFieldsReplaced": True,
            },
            **output_records,
        },
        "expressionAccess": {
            "replogle": "core-control aggregate rows only",
            "hepg2": "4976 non-targeting single-cell rows only",
            "jurkat": "12013 non-targeting single-cell rows only",
            "hepg2PerturbedRowsRead": 0,
            "jurkatPerturbedRowsRead": 0,
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
    parser.add_argument("--jurkat", type=Path, default=DEFAULT_JURKAT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(
        args.input.resolve(),
        args.source_dir.resolve(),
        args.hepg2.resolve(),
        args.jurkat.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
