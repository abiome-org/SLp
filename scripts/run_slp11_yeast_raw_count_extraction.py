"""Replay frozen eligible columns from verified yeast raw RNA/counts matrices."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "scripts/prepare_slp11_yeast_seurat_counts.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_count_prepare", PREPARE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load count extraction support")
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)

SOURCE = (
    ROOT
    / "data/sources/nadal-ribelles-2025-yeast-seus-split-v1/full-acquisition-v1/seus_split.RData"
)
SELECTION_ROOT = (
    ROOT / "results/slp11-transition/yeast-seurat-metadata-inventory-v1/selection"
)
OUTPUT = (
    ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1"
)
SOURCE_BYTES = 5_907_877_873
SOURCE_MD5 = "65bb56efd8120f32f65c044de5f040aa"
SOURCE_SHA256 = "da99869c11d1a6c034454568098aa50bc3313cd4508dbd506d43241b0fb4695d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_csc(path: Path, rows: int) -> dict[str, object]:
    i = np.load(path / "i.npy", mmap_mode="r")
    x = np.load(path / "x.npy", mmap_mode="r")
    indptr = np.load(path / "p.npy", mmap_mode="r")
    if len(i) != len(x) or indptr[-1] != len(x):
        raise p.ss.SeuratStreamError("selected raw CSC length mismatch")
    if len(i) and (i.min() < 0 or i.max() >= rows):
        raise p.ss.SeuratStreamError("selected raw CSC row index outside matrix")
    minimum = np.inf
    maximum = -np.inf
    noninteger = nonfinite = negative = 0
    for start in range(0, len(x), 1 << 20):
        values = np.asarray(x[start : start + (1 << 20)])
        if len(values):
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
        nonfinite += int(np.count_nonzero(~np.isfinite(values)))
        negative += int(np.count_nonzero(values < 0))
        noninteger += int(np.count_nonzero(values != np.floor(values)))
    if nonfinite or negative or noninteger:
        raise p.ss.SeuratStreamError(
            "RNA/counts payload is not finite nonnegative integer data"
        )
    return {
        "nnz": len(x),
        "minimum": minimum,
        "maximum": maximum,
        "nonfinite": nonfinite,
        "negative": negative,
        "noninteger": noninteger,
    }


def main() -> None:
    if OUTPUT.exists():
        raise p.ss.SeuratStreamError(f"refusing to overwrite {OUTPUT}")
    started = time.monotonic()
    source = p.verify_source(
        SOURCE,
        expected_bytes=SOURCE_BYTES,
        expected_md5=SOURCE_MD5,
        expected_sha256=SOURCE_SHA256,
    )
    inventory = p.ss.inspect_rdata(
        SOURCE,
        materialize_limit=4096,
        materialize_atomic_names={"p", "Dim"},
        max_materialized_bytes=256 << 20,
        max_rss_bytes=6 << 30,
        max_seconds=900 - (time.monotonic() - started),
    )
    if not inventory.complete or inventory.root is None:
        raise p.ss.SeuratStreamError("raw extraction inventory replay failed")
    _, candidates, _ = p.discover_structure(inventory.root)
    candidates = [
        candidate
        for candidate in candidates
        if p.ss.is_admissible_rna_counts_path(candidate.semantic_path)
    ]
    selection_report = json.loads(
        (SELECTION_ROOT / "selection-report.json").read_text()
    )
    by_context = {candidate.semantic_path[1]: candidate for candidate in candidates}
    if set(by_context) != {frame["context"] for frame in selection_report["frames"]}:
        raise p.ss.SeuratStreamError("raw RNA contexts and frozen selections disagree")
    OUTPUT.mkdir(parents=True)
    contexts: list[dict[str, object]] = []
    for frame in selection_report["frames"]:
        frame_index = frame["frameIndex"]
        context = frame["context"]
        selected = np.load(SELECTION_ROOT / f"frame-{frame_index}-selection.npz")
        selection = p.FrozenColumnSelection(
            selected["source_columns"],
            frame["selectionSha256"],
            frame["controls"],
            frame["selectedMutantCells"],
            frame["excludedProtectedCells"],
            frame["exactMapFailureCells"],
        )
        destination = OUTPUT / context.lower() / "raw-csc"
        remaining = 900 - (time.monotonic() - started)
        if remaining <= 0:
            raise p.ss.SeuratStreamError(
                "raw extraction exhausted total wall-time bound"
            )
        extraction = p.write_selected_csc(
            SOURCE,
            by_context[context],
            selection,
            destination,
            max_seconds=remaining,
            max_rss_bytes=6 << 30,
        )
        integrity = _validate_raw_csc(destination, by_context[context].rows)
        contexts.append(
            {
                "context": context,
                "extraction": extraction,
                "integrity": integrity,
                "files": {
                    name: {
                        "bytes": (destination / name).stat().st_size,
                        "sha256": _sha256(destination / name),
                    }
                    for name in ("i.npy", "x.npy", "p.npy", "source_columns.npy")
                },
            },
        )
    report = {
        "schema": "slp.yeast-seurat-raw-rna-development-extraction/v1",
        "source": source,
        "selectionReportSha256": _sha256(SELECTION_ROOT / "selection-report.json"),
        "queryMapSha256": _sha256(SELECTION_ROOT / "query-map.npz"),
        "contexts": contexts,
        "runtimeSeconds": time.monotonic() - started,
        "limits": {"totalSeconds": 900, "rssGiB": 6, "cpuThreads": 2, "gpu": False},
        "normalizationApplied": False,
        "countValuesUsedForSelection": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
