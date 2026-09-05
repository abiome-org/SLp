#!/usr/bin/env python3
"""Build one pooled-control context statistic for Replogle and HepG2.

Only core/non-targeting control expression is read.  The existing three-context
development target arrays and splits are copied byte-for-byte at the NPY-entry
level into a new immutable snapshot; only the context descriptor/provenance is
replaced and an explicit descriptor mask is added.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_descriptor import VALUE_SPACE, pooled_control_log2_cp10k

INPUT_SHA256 = "baac863d7050fbd71ac332a680215af1e400f759ad441534019905bd521fda96"
HEPG2_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
CONTROL_SUFFIX = "_non-targeting_non-targeting_non-targeting"
REPLOGLE_CODE_COMMIT = "3b25109aeb9c0c2026bd70abd50304a0ad4e5395"


@dataclass(frozen=True)
class ReplogleRawSource:
    context_id: str
    name: str
    sha256: str
    expected_shape: tuple[int, int]


REPLOGLE_SOURCES = (
    ReplogleRawSource(
        "replogle-2022-k562-essential-day-6",
        "K562_essential_raw_bulk_01.h5ad",
        "80de95e54fcbca0e0537d569b43ec92fde6bd0482801505504baebe3118dcadf",
        (2_285, 8_563),
    ),
    ReplogleRawSource(
        "replogle-2022-rpe1-essential-day-7",
        "rpe1_raw_bulk_01.h5ad",
        "603c655f1cfa41d649baf3ae63fca224cc11f297e40d4ed59d390b1e8d2e2db2",
        (2_679, 8_749),
    ),
    ReplogleRawSource(
        "replogle-2022-k562-gwps-day-8",
        "K562_gwps_raw_bulk_01.h5ad",
        "7cec96b3b76169abbf6b6ab9d10bf00d71d942d89e63292351f745e130b154db",
        (11_258, 8_248),
    ),
)

DEFAULT_INPUT = ROOT / (
    "data/derived/slp11-human-gwps/"
    "replogle-k562-rpe1-gwps-author-normalized-development-v3.npz"
)
DEFAULT_HEPG2 = ROOT / (
    "data/sources/nadig-2025-gse264667-hepg2-v1/"
    "GSE264667_hepg2_raw_singlecell_01.h5ad"
)
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-human-gwps-common-context-v1"


class CommonContextBuildError(RuntimeError):
    """Raised when a pinned source or aggregate contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[...]).astype(str)


def _npy_payload(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _copy_snapshot_with_context(
    source_path: Path,
    destination_path: Path,
    descriptor: np.ndarray,
    observed: np.ndarray,
    *,
    value_space: str = VALUE_SPACE,
) -> dict[str, str]:
    """Copy existing NPY payloads exactly while replacing context fields."""

    replacements = {
        "context_basal_expression.npy": _npy_payload(
            np.ascontiguousarray(descriptor, dtype=np.float32)
        ),
        "context_basal_observed.npy": _npy_payload(
            np.ascontiguousarray(observed, dtype=np.bool_)
        ),
        "context_value_space.npy": _npy_payload(np.asarray(value_space)),
    }
    unchanged_hashes: dict[str, str] = {}
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd, temporary = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", dir=destination_path.parent
    )
    os.close(descriptor_fd)
    try:
        with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as destination:
            source_names = set(source.namelist())
            for name in sorted(source_names | {"context_basal_observed.npy"}):
                if name in replacements:
                    destination.writestr(_zip_info(name), replacements[name])
                    continue
                digest = hashlib.sha256()
                with source.open(name, "r") as reader, destination.open(
                    _zip_info(name), "w"
                ) as writer:
                    while block := reader.read(8 * 1024 * 1024):
                        digest.update(block)
                        writer.write(block)
                unchanged_hashes[name.removesuffix(".npy")] = digest.hexdigest()
        os.replace(temporary, destination_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return unchanged_hashes


def _replogle_descriptor(
    source_dir: Path,
    source_spec: ReplogleRawSource,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = source_dir / source_spec.name
    if _sha256(path) != source_spec.sha256:
        raise CommonContextBuildError(f"source SHA-256 drift: {source_spec.name}")
    with h5py.File(path, "r") as source:
        if source["X"].shape != source_spec.expected_shape:
            raise CommonContextBuildError(f"source shape drift: {source_spec.name}")
        records = _decode(source["obs/gene_transcript"])
        core = np.asarray(source["obs/core_control"][...], dtype=np.bool_)
        rows = np.flatnonzero(core & np.char.endswith(records, CONTROL_SUFFIX))
        cell_counts = np.asarray(
            source["obs/num_cells_filtered"][...], dtype=np.float64
        )[rows]
        unfiltered_umi = np.asarray(
            source["obs/UMI_count_unfiltered"][...], dtype=np.float64
        )[rows]
        genes = _decode(source["var/gene_id"])
        # Exact control-population rows only. Deposited X is the arithmetic
        # mean of filtered member cells, as defined by CellPopulation.average.
        raw_means = np.asarray(source["X"][rows, :], dtype=np.float64)

    if (
        rows.size == 0
        or not np.isfinite(raw_means).all()
        or np.any(raw_means < 0.0)
        or np.any(cell_counts != np.floor(cell_counts))
    ):
        raise CommonContextBuildError(f"invalid raw controls: {source_spec.name}")
    full_mean_umi = raw_means.sum(axis=1)
    lookup = {gene: index for index, gene in enumerate(genes)}
    present = np.asarray([gene in lookup for gene in query_ids], dtype=np.bool_)
    aligned = np.zeros((rows.size, query_ids.size), dtype=np.float64)
    aligned[:, present] = raw_means[
        :, np.asarray([lookup[gene] for gene in query_ids[present]], dtype=np.int64)
    ]
    aligned_observed = np.broadcast_to(present, aligned.shape).copy()
    descriptor, observed = pooled_control_log2_cp10k(
        aligned,
        full_mean_umi,
        cell_counts,
        aligned_observed,
    )
    report = {
        "contextId": source_spec.context_id,
        "source": f"data/sources/human/{source_spec.name}",
        "sha256": source_spec.sha256,
        "controlPopulations": int(rows.size),
        "filteredControlCells": int(cell_counts.sum()),
        "querySupport": int(observed.sum()),
        "rawRowsArePopulationMeans": True,
        "populationTotalReconstruction": "stored raw mean * num_cells_filtered",
        "fullMeanUmi": "sum across all deposited raw X genes before query alignment",
        "umiCountUnfilteredUsed": False,
        "unfilteredUmiMedian": float(np.median(unfiltered_umi)),
        "filteredRawMeanUmiMedian": float(np.median(full_mean_umi)),
    }
    return descriptor, observed, report


def _hepg2_descriptor(
    path: Path,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if _sha256(path) != HEPG2_SHA256:
        raise CommonContextBuildError("HepG2 source SHA-256 drift")
    with h5py.File(path, "r") as source:
        categories = _decode(source["obs/__categories/gene_id"])
        codes = np.asarray(source["obs/gene_id"][...], dtype=np.int64)
        matches = np.flatnonzero(categories == "non-targeting")
        if matches.size != 1:
            raise CommonContextBuildError("HepG2 non-targeting category drift")
        rows = np.flatnonzero(codes == int(matches[0]))
        if rows.size != 4_976 or np.any(codes[rows] != int(matches[0])):
            raise CommonContextBuildError("HepG2 control roster drift")
        full_umi = np.asarray(source["obs/UMI_count"][...], dtype=np.float64)[rows]
        genes = _decode(source["var/gene_id"])
        # Exact non-targeting rows only; no perturbed X row is indexed.
        raw_cells = np.asarray(source["X"][rows, :], dtype=np.float64)

    lookup = {gene: index for index, gene in enumerate(genes)}
    present = np.asarray([gene in lookup for gene in query_ids], dtype=np.bool_)
    aligned = np.zeros((rows.size, query_ids.size), dtype=np.float64)
    aligned[:, present] = raw_cells[
        :, np.asarray([lookup[gene] for gene in query_ids[present]], dtype=np.int64)
    ]
    aligned_observed = np.broadcast_to(present, aligned.shape).copy()
    descriptor, observed = pooled_control_log2_cp10k(
        aligned,
        full_umi,
        np.ones(rows.size, dtype=np.int64),
        aligned_observed,
    )
    report = {
        "contextId": "nadig-2025-hepg2-day-7",
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": HEPG2_SHA256,
        "controlCells": int(rows.size),
        "querySupport": int(observed.sum()),
        "fullUmi": "source obs/UMI_count",
        "selectedPanelSumUsedAsDenominator": False,
        "perturbedExpressionRowsRead": 0,
    }
    return descriptor, observed, report


def build(
    input_path: Path,
    source_dir: Path,
    hepg2_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build the common-control descriptors and replacement snapshot."""

    if _sha256(input_path) != INPUT_SHA256:
        raise CommonContextBuildError("three-context development SHA-256 drift")
    with np.load(input_path, allow_pickle=False) as archive:
        query_ids = archive["query_ids"]
        context_ids = archive["context_ids"]
    if query_ids.shape != (7_226,) or context_ids.tolist() != [
        source.context_id for source in REPLOGLE_SOURCES
    ]:
        raise CommonContextBuildError("three-context identity contract drift")

    descriptors = []
    masks = []
    replogle_reports = []
    for source in REPLOGLE_SOURCES:
        descriptor, observed, report = _replogle_descriptor(
            source_dir, source, query_ids
        )
        descriptors.append(descriptor)
        masks.append(observed)
        replogle_reports.append(report)
    descriptor_matrix = np.stack(descriptors)
    mask_matrix = np.stack(masks)

    output_dir.mkdir(parents=True, exist_ok=True)
    development_path = output_dir / (
        "replogle-k562-rpe1-gwps-author-normalized-development-"
        "v4-common-control-context.npz"
    )
    unchanged_hashes = _copy_snapshot_with_context(
        input_path,
        development_path,
        descriptor_matrix,
        mask_matrix,
    )

    hepg2_descriptor, hepg2_observed, hepg2_report = _hepg2_descriptor(
        hepg2_path, query_ids
    )
    hepg2_output = output_dir / "nadig-hepg2-common-control-context-v1.npz"
    np.savez_compressed(
        hepg2_output,
        schema=np.asarray("slp.unseen-context-control-descriptor/v1"),
        context_ids=np.asarray(["nadig-2025-hepg2-day-7"]),
        query_ids=query_ids,
        context_basal_expression=hepg2_descriptor[None, :].astype(np.float32),
        context_basal_observed=hepg2_observed[None, :],
        context_value_space=np.asarray(VALUE_SPACE),
        source_sha256=np.asarray(HEPG2_SHA256),
        perturbed_expression_rows_read=np.asarray(0, dtype=np.int64),
    )

    report: dict[str, object] = {
        "schema": "slp.common-control-context-build-report/v1",
        "valueSpace": VALUE_SPACE,
        "formula": (
            "log2(1 + 10000 * sum(control mean count * filtered cells) / "
            "sum(control full mean UMI * filtered cells))"
        ),
        "inputDevelopment": {
            "path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": INPUT_SHA256,
        },
        "replogleAggregateSemantics": {
            "sourceCodeRepository": (
                "https://github.com/thomasmaxwellnorman/Perturbseq_GI"
            ),
            "commit": REPLOGLE_CODE_COMMIT,
            "method": "CellPopulation.average stores arithmetic expression means",
            "precision": (
                "totals are exact reconstructions of deposited float32 aggregate means, "
                "not recovery of pre-rounding integer counts"
            ),
        },
        "replogleContexts": replogle_reports,
        "hepg2Context": hepg2_report,
        "outputs": {
            "development": {
                "path": str(development_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": development_path.stat().st_size,
                "sha256": _sha256(development_path),
                "unchangedNpyPayloadSha256": unchanged_hashes,
                "targetsSplitsAndOtherArraysCopiedWithoutNpyPayloadChange": True,
            },
            "hepg2": {
                "path": str(hepg2_output.relative_to(ROOT)).replace("\\", "/"),
                "bytes": hepg2_output.stat().st_size,
                "sha256": _sha256(hepg2_output),
                "alignedQueries": int(query_ids.size),
                "observedQueries": int(hepg2_observed.sum()),
            },
        },
        "expressionAccess": {
            "replogle": "core-control population rows only",
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
