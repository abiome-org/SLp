"""Describe measured WT batch variation; read only selected WT count columns."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1"
SELECTION = ROOT / "results/slp11-transition/yeast-seurat-metadata-inventory-v1/selection"
OUTPUT = ROOT / "results/slp11-transition/yeast-wildtype-batch-diagnostic-v1"
CORE = ROOT / "modules/slp-1-1-count-moments-v1/count_moments.py"
PINS = {
    "manifest": "18c4b3e2f6cdfd33ce663f11a83cc2cdae65e3cad9ec56e9575cfd2783d9f148",
    "query_map": "776bd0021bbdf53243d93e2283c27bfbca07a71b4fb8c0af5cd92299a9d4f018",
    "core": "53344c00ad4a8615c796a2f41371efc46823eb59e82925e67ff2178d72d004c3",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dispersion(mean: np.ndarray, variance: np.ndarray, n: np.ndarray):
    """Weighted centroid variance and independent-cell sampling contribution."""
    mean, variance, n = (np.asarray(v, dtype=np.float64) for v in (mean, variance, n))
    if np.any(n < 2) or not np.isfinite(variance).all():
        raise ValueError("batch dispersion requires at least two observed cells each")
    weight = n / n.sum()
    center = weight @ mean
    observed = float(np.sum(weight[:, None] * (mean - center) ** 2) / mean.shape[1])
    sampling = float(np.sum((1 - weight[:, None]) * variance) / n.sum() / mean.shape[1])
    return {
        "weighted_between_batch_mean_squared_dispersion": observed,
        "estimated_independent_cell_sampling_contribution": sampling,
        "signed_excess_dispersion": observed - sampling,
        "observed_to_sampling_ratio": observed / sampling if sampling > 0 else None,
    }


def main():
    for key, path in (
        ("manifest", COUNTS / "manifest.json"),
        ("query_map", SELECTION / "query-map.npz"),
        ("core", CORE),
    ):
        if sha(path) != PINS[key]:
            raise ValueError(f"input drift: {key}")
    if OUTPUT.exists():
        raise ValueError("immutable diagnostic output already exists")
    manifest = json.loads((COUNTS / "manifest.json").read_text())
    query = dict(np.load(SELECTION / "query-map.npz", allow_pickle=False))
    spec = importlib.util.spec_from_file_location("count_moments", CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    OUTPUT.mkdir()
    protocol = {
        "schema": "slp.yeast-wildtype-batch-diagnostic-protocol/v1",
        "purpose": "Descriptive WT batch variation and finite-count uncertainty; no model selection or mutant values.",
        "inputs": PINS,
        "script_sha256": sha(Path(__file__)),
        "measurement": "Frozen all6951-denominator ln1pCP10k, strict6683 queries, no filtering.",
        "estimator": "Sum_b (n_b/N)*mean_q[(mean_b-mean_all)^2]; sampling=sum_b (1-n_b/N)*mean_q[var_b]/N.",
        "limitation": "Sampling correction assumes independent cells and ignores clone dependence; source batches are not established biological replicates.",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    contexts, arrays = {}, {"query_ids": query["query_ids"]}
    for index, name in enumerate(("control", "nacl")):
        folder = COUNTS / name / "raw-csc"
        item = manifest["contexts"][index]
        for file, details in item["files"].items():
            if sha(folder / file) != details["sha256"]:
                raise ValueError(f"selected count file drift: {name}/{file}")
        selection = dict(np.load(SELECTION / f"frame-{index}-selection.npz", allow_pickle=False))
        source_columns = np.load(folder / "source_columns.npy")
        if not np.array_equal(source_columns, selection["source_columns"]):
            raise ValueError("selection alignment drift")
        control = selection["is_control"]
        if not np.array_equal(control, selection["assignment_consensus2"] == "WT"):
            raise ValueError("WT identity drift")
        # Construct structure with memmaps, then materialize only explicit WT columns.
        matrix = sparse.csc_matrix(
            (
                np.load(folder / "x.npy", mmap_mode="r"),
                np.load(folder / "i.npy", mmap_mode="r"),
                np.load(folder / "p.npy", mmap_mode="r"),
            ),
            shape=tuple(item["extraction"]["shape"]),
        )
        cells = matrix[:, np.flatnonzero(control)].T.tocsr()
        del matrix
        batches, groups = np.unique(selection["batch"][control], return_inverse=True)
        moments = core.CountMoments(
            query["source_to_query_index"], query["denominator_mask"],
            len(query["query_ids"]), len(batches),
        )
        valid = moments.update(cells, groups)
        summary = moments.summary()
        library = np.asarray(cells.sum(axis=1)).ravel()
        result = dispersion(summary["mean"], summary["cell_variance"], summary["num_cells"])
        result.update({
            "cells": int(control.sum()),
            "zero_library_cells": int((~valid).sum()),
            "library_count_quantiles": np.quantile(library, [0, .1, .5, .9, 1]).tolist(),
            "batch_cell_counts": dict(zip(batches.astype(str), summary["num_cells"].tolist(), strict=True)),
        })
        contexts[name] = result
        for key in ("mean", "cell_variance", "num_cells"):
            arrays[f"{name}_{key}"] = summary[key]
        arrays[f"{name}_batch_ids"] = batches
    np.savez_compressed(OUTPUT / "wildtype-reference.npz", **arrays)
    report = {
        "schema": "slp.yeast-wildtype-batch-diagnostic/v1", "contexts": contexts,
        "mutant_columns_materialized": 0,
        "protocol_sha256": sha(OUTPUT / "protocol.json"),
        "reference_sha256": sha(OUTPUT / "wildtype-reference.npz"),
        "limitations": protocol["limitation"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
