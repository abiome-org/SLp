#!/usr/bin/env python3
"""Build the target-free graph adapter for the three-context gene-state pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TAXON = 9606
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
BASE_SHA256 = "f4bbfe62b73cf6362170996fcf34200cea68da106d687d3c9e994e709e951f40"
ALIASES_SHA256 = "b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680"
LINKS_SHA256 = "b28f494f58e1ace634ef1fe41734ada5be37f151e3168bb9658bc6ca1dd1a954"
GENE_RE = re.compile(r"^ENSG\d{11}$")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_physical_edges(source: Path) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    aliases_path = source / "9606.protein.aliases.v12.0.txt.gz"
    links_path = source / "9606.protein.physical.links.full.v12.0.txt.gz"
    if sha256(aliases_path) != ALIASES_SHA256 or sha256(links_path) != LINKS_SHA256:
        raise ValueError("STRING v12 physical source digest mismatch")
    aliases: dict[str, set[str]] = defaultdict(set)
    with gzip.open(aliases_path, "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            protein, gene, label = line.rstrip("\n").split("\t", 2)
            if label == "Ensembl_gene" and GENE_RE.fullmatch(gene):
                aliases[protein].add(gene)
    exact = {protein: next(iter(genes)) for protein, genes in aliases.items() if len(genes) == 1}
    pairs: dict[tuple[str, str], int] = {}
    strong_rows = 0
    with gzip.open(links_path, "rt", encoding="utf-8") as stream:
        columns = next(stream).split()
        experiment_column = columns.index("experiments")
        for line in stream:
            fields = line.split()
            confidence = int(fields[experiment_column])
            if confidence < 700:
                continue
            strong_rows += 1
            left, right = exact.get(fields[0]), exact.get(fields[1])
            if left is None or right is None or left == right:
                continue
            pair = tuple(sorted((left, right)))
            pairs[pair] = max(pairs.get(pair, 0), confidence)
    return (
        [(left, right, score / 1000.0) for (left, right), score in sorted(pairs.items())],
        {
            "strongSourceRows": strong_rows,
            "uniqueExactMappedGeneEdges": len(pairs),
            "ambiguousAliasProteins": sum(len(genes) != 1 for genes in aliases.values()),
        },
    )


def graph_arrays(
    base_ids: np.ndarray,
    base_values: np.ndarray,
    action_ids: np.ndarray,
    query_ids: np.ndarray,
    fitting_action_ids: np.ndarray,
    edges: list[tuple[str, str, float]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build one sorted universe and a row-normalized fixed physical graph."""
    base_ids = np.asarray(base_ids, dtype=str)
    base_values = np.asarray(base_values, dtype=np.float32)
    action_ids = np.asarray(action_ids, dtype=str)
    query_ids = np.asarray(query_ids, dtype=str)
    fitting_action_ids = np.asarray(fitting_action_ids, dtype=str)
    if (
        base_values.shape != (len(base_ids), 577)
        or len(set(base_ids)) != len(base_ids)
        or not np.isfinite(base_values).all()
        or any(GENE_RE.fullmatch(item) is None for item in np.concatenate((base_ids, action_ids, query_ids)))
    ):
        raise ValueError("gene-state static identity or feature contract mismatch")
    node_ids = np.asarray(sorted(set(base_ids) | set(action_ids) | set(query_ids)))
    node_row = {gene: row for row, gene in enumerate(node_ids)}
    base_row = {gene: row for row, gene in enumerate(base_ids)}
    static_observed = np.asarray([gene in base_row for gene in node_ids], dtype=np.bool_)
    raw_static = np.zeros((len(node_ids), 577), dtype=np.float32)
    present_rows = np.flatnonzero(static_observed)
    raw_static[present_rows] = base_values[[base_row[node_ids[row]] for row in present_rows]]

    fit_present = sorted(set(fitting_action_ids) & set(base_ids))
    if not fit_present:
        raise ValueError("no feature-covered fitting action genes")
    fitting_values = base_values[[base_row[gene] for gene in fit_present]].astype(np.float64)
    feature_mean = fitting_values.mean(axis=0)
    feature_scale = fitting_values.std(axis=0)
    feature_scale[feature_scale < 1e-5] = 1.0
    standardized = np.zeros_like(raw_static)
    standardized[present_rows] = (
        (raw_static[present_rows].astype(np.float64) - feature_mean) / feature_scale
    ).astype(np.float32)

    base_set = set(base_ids)
    directed: dict[tuple[int, int], float] = {}
    for left, right, weight in edges:
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("physical confidence must be finite in (0,1]")
        # Appended source-only identities are deliberately isolated.
        if left in base_set and right in base_set:
            left_row, right_row = node_row[left], node_row[right]
            directed[(left_row, right_row)] = max(directed.get((left_row, right_row), 0.0), weight)
            directed[(right_row, left_row)] = max(directed.get((right_row, left_row), 0.0), weight)
    ordered = sorted(directed)
    rows = np.asarray([pair[0] for pair in ordered], dtype=np.int64)
    columns = np.asarray([pair[1] for pair in ordered], dtype=np.int64)
    weights64 = np.asarray([directed[pair] for pair in ordered], dtype=np.float64)
    row_mass = np.bincount(rows, weights=weights64, minlength=len(node_ids))
    normalized = (weights64 / row_mass[rows]).astype(np.float32)
    row_counts = np.bincount(rows, minlength=len(node_ids))
    indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)
    normalized_mass = np.bincount(rows, weights=normalized, minlength=len(node_ids))
    nonempty = row_counts > 0
    if nonempty.any() and not np.allclose(normalized_mass[nonempty], 1.0, rtol=1e-6, atol=1e-6):
        raise RuntimeError("row normalization failed")

    arrays = {
        "node_ids": node_ids.astype("<U15"),
        "node_taxon": np.full(len(node_ids), TAXON, dtype=np.int64),
        "static_features": standardized,
        "static_feature_observed": static_observed,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "adjacency_indptr": indptr,
        "adjacency_indices": columns,
        "adjacency_weights": normalized,
        "action_node_index": np.asarray([node_row[gene] for gene in action_ids], dtype=np.int64),
        "query_node_index": np.asarray([node_row[gene] for gene in query_ids], dtype=np.int64),
    }
    stats = {
        "nodes": len(node_ids),
        "translatedFeatureNodes": int(static_observed.sum()),
        "explicitMissingFeatureNodes": int((~static_observed).sum()),
        "missingActionGenes": len(set(action_ids) - base_set),
        "missingQueryGenes": len(set(query_ids) - base_set),
        "featureCoveredFittingGenes": len(fit_present),
        "directedEdges": len(rows),
        "nodesWithIncomingPhysicalEdges": int(nonempty.sum()),
        "maximumDegree": int(row_counts.max(initial=0)),
        "adjacencyDirection": "adjacency[i,j] sends node j to node i",
    }
    return arrays, stats


def build(args: argparse.Namespace) -> dict[str, object]:
    if sha256(args.data) != DATA_SHA256 or sha256(args.base_features) != BASE_SHA256:
        raise ValueError("source-three data or static base digest mismatch")
    if args.output.exists():
        raise FileExistsError("immutable graph adapter output already exists")
    with np.load(args.data, allow_pickle=False) as data, np.load(
        args.base_features, allow_pickle=False,
    ) as base:
        if len(data["split_test"]) or not np.all(base["entity_taxon"] == TAXON):
            raise ValueError("adapter accepts development identities and human static features only")
        action_ids = data["action_ids"].astype(str)
        query_ids = data["query_ids"].astype(str)
        fitting_ids = action_ids[data["split_train"]]
        base_ids = base["entity_id"].astype(str)
        base_values = base["feature_values"]
    edges, edge_stats = load_physical_edges(args.string_source)
    arrays, graph_stats = graph_arrays(
        base_ids, base_values, action_ids, query_ids, fitting_ids, edges,
    )
    args.output.mkdir(parents=True)
    graph_path = args.output / "source3-gene-state-graph.npz"
    np.savez_compressed(graph_path, **arrays)
    manifest = {
        "schema": "slp.source3-gene-state-graph-adapter/v1",
        "graph": graph_stats,
        "edgeSource": {**edge_stats, "confidenceColumn": "experiments", "minimum": 700},
        "featureStandardization": (
            "mean/SD over unique source fitting action genes with translated static coverage; "
            "SD<1e-5 replaced by1; appended missing-feature nodes remain exact zero"
        ),
        "inputs": {
            "development": {"path": str(args.data.resolve()), "sha256": DATA_SHA256},
            "baseFeatures": {"path": str(args.base_features.resolve()), "sha256": BASE_SHA256},
            "stringAliases": ALIASES_SHA256,
            "stringPhysicalLinks": LINKS_SHA256,
        },
        "output": {"path": graph_path.name, "sha256": sha256(graph_path)},
        "access": {
            "quantitativePerturbationOutcomesRead": False,
            "developmentIdentityAndSplitRostersRead": True,
            "testRowsPresent": False,
        },
        "limitations": [
            "STRING physical associations are static undirected evidence, not causal edges.",
            "Action strength one later denotes intervention presence, not measured efficacy or knockdown.",
            "Appended source-only identities have missing static features and no graph edges.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--string-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
