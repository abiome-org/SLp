"""Consolidate fitting-only K562 action-by-GEM CP10k moments."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/replogle-2022-k562-essential-singlecell-v1/K562_essential_raw_singlecell_01.h5ad"
ROUTING = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
PARENT = ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2"
OUTPUT = ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1"
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
ROUTING_SHA256 = "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad"
PARENT_MANIFEST_SHA256 = "859b3fb0b0aeb830e25dce17e86edfc2d8ec3fcdbcec57beeeebf6d1a8faf685"
SHAPE = (310_385, 8_563)
MATRIX_OFFSET = 2048
SCHEMA = "slp.replogle-k562-essential-fitting-action-gem-moments/v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    width = max(1, *(len(x) for x in values))
    return values.astype(f"<U{width}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            with archive.open(_zip_info(name), "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def action_moments(
    raw_counts: np.ndarray, library_size: np.ndarray, action_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    raw = np.asarray(raw_counts)
    library = np.asarray(library_size, dtype=np.float64)
    actions = np.asarray(action_ids, dtype=str)
    if raw.ndim != 2 or len(raw) != len(library) or np.any(library <= 0):
        raise ValueError("invalid moment rows")
    if raw.dtype != np.float32 or not np.all(np.isfinite(raw)) or np.any(raw < 0) or not np.array_equal(raw, np.rint(raw)):
        raise ValueError("raw counts must be exact finite nonnegative float32 integers")
    measured_library = raw.astype(np.int64).sum(axis=1)
    if not np.array_equal(measured_library, library.astype(np.int64)):
        raise ValueError("retained-panel library mismatch")
    genes, inverse = np.unique(actions, return_inverse=True)
    membership = sparse.csr_matrix(
        (np.ones(len(inverse), dtype=np.float64), (inverse, np.arange(len(inverse)))),
        shape=(len(genes), len(inverse)),
    )
    counts = sparse.csr_matrix(raw.astype(np.int32))
    rates = counts.astype(np.float64).multiply((10_000.0 / library)[:, None])
    sums = (membership @ rates).tocsr()
    sums.sort_indices()
    return genes, np.bincount(inverse, minlength=len(genes)).astype(np.int64), sums


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUTPUT}")
    if digest(ROUTING) != ROUTING_SHA256 or digest(PARENT / "manifest.json") != PARENT_MANIFEST_SHA256:
        raise ValueError("routing or canonical parent drift")
    with np.load(ROUTING, allow_pickle=False) as archive:
        routing = {name: np.asarray(archive[name]) for name in archive.files}
    fit = (routing["intervention_role"].astype(str) == "train") & (routing["reconstruction_role"].astype(str) == "train") & ~routing["is_control"]
    if int(fit.sum()) != 188_195:
        raise ValueError("fitting row count drift")
    all_genes = np.asarray(sorted(set(routing["action_ids"][fit].astype(str))))
    if len(all_genes) != 1_443 or "" in all_genes:
        raise ValueError("fitting action roster drift")
    query_ids = routing["query_ids"].astype(str)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "by-gem").mkdir()
    protocol = {
        "schema": SCHEMA, "status": "frozen-before-quantitative-source-access",
        "hypothesis": "fitting reconstruction-training raw cells can be exactly consolidated into action-by-GEM CP10k sums without accessing held roles",
        "advancementRule": "188195 positive-library fitting cells and1443 stable actions are represented exactly once across48 GEM moment shards; every retained-panel library equals the raw8563-column sum; held/dev/test/control X rows read equals zero",
        "source": {"path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": SOURCE_SHA256, "matrixOffset": MATRIX_OFFSET},
        "routing": {"path": str(ROUTING.relative_to(ROOT)).replace("\\", "/"), "sha256": ROUTING_SHA256},
        "parentManifest": {"path": str((PARENT / "manifest.json").relative_to(ROOT)).replace("\\", "/"), "sha256": PARENT_MANIFEST_SHA256},
        "rateDefinition": "CP10k=10000*raw_count/sum_raw_counts_across_ordered_8563_source_columns",
        "included": "intervention_role=train and reconstruction_role=train and not control",
        "excluded": ["reconstruction-held", "development-validation", "test-excluded", "control"],
        "implementation": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": digest(Path(__file__).resolve())},
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    pooled_sum = np.zeros((len(all_genes), SHAPE[1]), dtype=np.float64)
    cell_count = np.zeros(len(all_genes), dtype=np.int64)
    gem_cell_count = np.zeros((len(all_genes), 48), dtype=np.int64)
    gene_to_row = {gene: index for index, gene in enumerate(all_genes)}
    gem_items = []
    maximum_count = 0
    for gem in range(1, 49):
        rows = np.flatnonzero(fit & (routing["gem_group"] == gem))
        if len(rows) == 0:
            raise AssertionError("fitting GEM has no cells")
        matrix = np.memmap(SOURCE, mode="r", dtype=np.float32, offset=MATRIX_OFFSET, shape=SHAPE, order="C")
        raw = np.asarray(matrix[rows, :])
        del matrix
        maximum_count = max(maximum_count, int(raw.max(initial=0)))
        libraries = raw.astype(np.int64).sum(axis=1)
        if np.any(libraries <= 0):
            raise ValueError("zero fitting library")
        genes, counts, sums = action_moments(raw, libraries, routing["action_ids"][rows])
        destination = np.asarray([gene_to_row[gene] for gene in genes], dtype=np.int64)
        pooled_sum[destination] += sums.toarray()
        cell_count[destination] += counts
        gem_cell_count[destination, gem - 1] += counts
        path = OUTPUT / "by-gem" / f"moments-gem-{gem:02d}.npz"
        write_npz(path, {
            "schema": np.asarray(SCHEMA), "source_sha256": np.asarray(SOURCE_SHA256),
            "query_ids": _strings(query_ids), "action_ids": _strings(genes),
            "gem_group": np.asarray(gem, np.int16), "num_cells": counts,
            "sum_cp10k_data": sums.data.astype(np.float64),
            "sum_cp10k_indices": sums.indices.astype(np.int32),
            "sum_cp10k_indptr": sums.indptr.astype(np.int64),
            "sum_cp10k_shape": np.asarray(sums.shape, np.int64),
        })
        gem_items.append({"gemGroup": gem, "path": f"by-gem/{path.name}", "cells": len(rows), "actions": len(genes), "bytes": path.stat().st_size, "sha256": digest(path)})
    if int(cell_count.sum()) != 188_195 or not np.array_equal(cell_count, gem_cell_count.sum(axis=1)):
        raise AssertionError("consolidated cell counts do not close")
    pooled_path = OUTPUT / "fitting-action-moments.npz"
    write_npz(pooled_path, {
        "schema": np.asarray(f"{SCHEMA}.pooled-actions"), "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256), "query_ids": _strings(query_ids),
        "query_taxon": np.full(SHAPE[1], 9606, np.int64), "action_ids": _strings(all_genes),
        "gem_group": np.arange(1, 49, dtype=np.int16), "cp10k_sum": pooled_sum,
        "cell_count": cell_count, "gem_cell_count": gem_cell_count,
        "rate_definition": np.asarray(protocol["rateDefinition"]),
    })
    manifest = {
        "schema": SCHEMA, "status": "complete", "protocolSha256": digest(OUTPUT / "protocol.json"),
        "sourceSha256": SOURCE_SHA256, "routingSha256": ROUTING_SHA256,
        "counts": {"cells": int(cell_count.sum()), "actions": len(all_genes), "queries": SHAPE[1], "gemGroups": 48, "maximumRawCount": maximum_count, "heldRowsRead": 0, "developmentValidationRowsRead": 0, "testRowsRead": 0, "controlRowsRead": 0},
        "pooledActions": {"path": pooled_path.name, "bytes": pooled_path.stat().st_size, "sha256": digest(pooled_path), "shape": [len(all_genes), SHAPE[1]]},
        "byGem": gem_items, "originalSquaredMomentFragments": {"parentManifest": PARENT_MANIFEST_SHA256, "path": "data/derived/slp11-human-k562-essential-raw-cells-v2/fit-moments"},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()["counts"], indent=2))
