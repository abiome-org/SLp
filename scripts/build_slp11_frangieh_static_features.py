#!/usr/bin/env python3
"""Build fixed-universe human ESM/GO/physical features for Frangieh metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_slp11_human_go_features as go
import build_slp11_human_sequence_features as sequence
import build_slp11_norman_static_features as frozen

TAXON = 9606
OLD_FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
QUERY_SHA256 = "87bac5ddbe3a1546d49896b3e1135efd087260a3db5704d6946d9de7a36fc14a"
ADT_SHA256 = "a23afdb6c0214fc79d429aac28a9b0b17599196f7768ab460d16a3ed34d1e3f8"
ORIGINAL_GO_SHA256 = "208be756b81229b3881af8229e18ba2f5e806f5be85180b6f5560c3f2d07c0ea"
GO_COMPONENT_SHA256 = "44dc50187681703238b66a905750cfd25decbba8e9adb457a8f77bb69a2f5f2d"
STRING_HASHES = {
    "9606.protein.aliases.v12.0.txt.gz": "b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680",
    "9606.protein.physical.links.full.v12.0.txt.gz": "b28f494f58e1ace634ef1fe41734ada5be37f151e3168bb9658bc6ca1dd1a954",
}
OUTPUT_COUNT = 18_893
GRAPH_COUNT = 23_879
GENE_RE = re.compile(r"^ENSG\d{11}$")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lf_payload(ids: list[str] | tuple[str, ...]) -> bytes:
    return "".join(f"{item}\n" for item in ids).encode("ascii")


def read_lf_ids(path: Path, expected_sha: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("ID roster SHA-256 mismatch")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("ID roster must be LF-terminated ASCII")
    ids = tuple(payload.decode("ascii").splitlines())
    if len(ids) != len(set(ids)) or any(GENE_RE.fullmatch(item) is None for item in ids):
        raise ValueError("ID roster identity contract mismatch")
    return ids


def output_arrays(ids: tuple[str, ...], values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "feature_values": values.astype(np.dtype("<f4"), copy=False),
        "entity_taxon": np.full(len(ids), TAXON, dtype=np.dtype("<i8")),
        "entity_id": np.asarray(ids, dtype="<U15"),
    }


def load_edges(source: Path) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    for name, digest in STRING_HASHES.items():
        if sha256_file(source / name) != digest:
            raise ValueError("STRING source drift")
    aliases: dict[str, set[str]] = defaultdict(set)
    with gzip.open(source / "9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            protein, gene, label = line.rstrip("\n").split("\t", 2)
            if label == "Ensembl_gene" and GENE_RE.fullmatch(gene):
                aliases[protein].add(gene)
    exact = {protein: next(iter(genes)) for protein, genes in aliases.items() if len(genes) == 1}
    pairs: dict[tuple[str, str], int] = {}
    strong_rows = 0
    with gzip.open(
        source / "9606.protein.physical.links.full.v12.0.txt.gz", "rt", encoding="utf-8"
    ) as stream:
        columns = next(stream).split()
        experiment = columns.index("experiments")
        for line in stream:
            fields = line.split()
            confidence = int(fields[experiment])
            if confidence < 700:
                continue
            strong_rows += 1
            left, right = exact.get(fields[0]), exact.get(fields[1])
            if left is None or right is None or left == right:
                continue
            pair = tuple(sorted((left, right)))
            pairs[pair] = max(pairs.get(pair, 0), confidence)
    edges = [(left, right, confidence / 1000.0) for (left, right), confidence in sorted(pairs.items())]
    return edges, {
        "strongSourceRows": strong_rows,
        "uniqueExactMappedGeneEdges": len(edges),
        "ambiguousAliasProteins": sum(len(genes) != 1 for genes in aliases.values()),
    }


def fixed_neighborhood_features(
    output_ids: tuple[str, ...],
    output_base: np.ndarray,
    graph_ids: tuple[str, ...],
    graph_base: np.ndarray,
    edges: list[tuple[str, str, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Aggregate over a fixed graph source universe, independent of output rows."""
    if (
        output_base.shape != (len(output_ids), 577)
        or graph_base.shape != (len(graph_ids), 577)
        or not np.isfinite(output_base).all()
        or not np.isfinite(graph_base).all()
        or len(output_ids) != len(set(output_ids))
        or len(graph_ids) != len(set(graph_ids))
    ):
        raise ValueError("fixed-neighborhood feature shapes or identities disagree")
    output_row = {gene: row for row, gene in enumerate(output_ids)}
    graph_row = {gene: row for row, gene in enumerate(graph_ids)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left, right, weight in edges:
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("physical confidence must be in (0,1]")
        if left in output_row and right in graph_row:
            rows.append(output_row[left])
            columns.append(graph_row[right])
            weights.append(weight)
        if right in output_row and left in graph_row:
            rows.append(output_row[right])
            columns.append(graph_row[left])
            weights.append(weight)
    adjacency = csr_matrix(
        (np.asarray(weights, dtype=np.float32), (rows, columns)),
        shape=(len(output_ids), len(graph_ids)),
    )
    degree = np.diff(adjacency.indptr).astype(np.float32)
    mass = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbors = (adjacency @ graph_base) / np.maximum(mass[:, None], 1e-12)
    values = np.concatenate(
        (
            output_base,
            neighbors,
            np.log1p(degree)[:, None],
            (degree > 0)[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    return values, {
        "directedOutputToGraphLinks": int(adjacency.nnz),
        "outputEntitiesWithNeighbors": int(np.count_nonzero(degree)),
        "maximumDegree": int(degree.max(initial=0)),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    if sha256_file(args.old_features) != OLD_FEATURE_SHA256:
        raise ValueError("old physical feature pack drift")
    queries = read_lf_ids(args.query_ids, QUERY_SHA256)
    with np.load(args.old_features, allow_pickle=False) as archive:
        old_ids = tuple(str(item) for item in archive["entity_id"])
        old_physical = archive["feature_values"]
    if old_physical.shape != (10_231, 1_156):
        raise ValueError("old physical feature shape drift")
    old_base = old_physical[:, :577].copy()
    old_row = {gene: row for row, gene in enumerate(old_ids)}
    output_ids = tuple(sorted(set(old_ids) | set(queries)))
    if len(output_ids) != OUTPUT_COUNT:
        raise ValueError("unexpected Frangieh output gene union")
    if sha256_file(args.adt_roster) != ADT_SHA256:
        raise ValueError("ADT stable component roster drift")

    fasta = sequence.verify_source_dir(args.ensembl_source)
    model_files = sequence.verify_esm_model_dir(args.esm_model)
    translations, sequence_source_counts = sequence.parse_longest_translations(fasta)
    graph_ids = tuple(sorted(translations))
    if len(graph_ids) != GRAPH_COUNT:
        raise ValueError("Ensembl116 translated graph universe drift")
    graph_row = {gene: row for row, gene in enumerate(graph_ids)}
    graph_sequence = np.zeros((len(graph_ids), 321), dtype=np.float32)
    copied_graph = []
    extract_graph = []
    for gene in graph_ids:
        prior = old_row.get(gene)
        if prior is not None and old_base[prior, 320] == 1:
            graph_sequence[graph_row[gene]] = old_base[prior, :321]
            copied_graph.append(gene)
        else:
            extract_graph.append(gene)
    cache_ids = tuple(extract_graph)
    if args.embedding_cache.exists():
        with np.load(args.embedding_cache, allow_pickle=False) as archive:
            if tuple(archive.files) != ("feature_values", "entity_taxon", "entity_id"):
                raise ValueError("embedding cache schema mismatch")
            cached_ids = tuple(str(item) for item in archive["entity_id"])
            cached_taxon = archive["entity_taxon"]
            cached = archive["feature_values"]
        if (
            cached_ids != cache_ids
            or cached.shape != (len(cache_ids), 321)
            or not np.array_equal(cached_taxon, np.full(len(cache_ids), TAXON))
            or not np.isfinite(cached).all()
            or not np.all(cached[:, 320] == 1)
        ):
            raise ValueError("embedding cache identity, value, or missingness mismatch")
        extraction = {
            "loadedFromCache": True,
            "cacheSha256": sha256_file(args.embedding_cache),
            "presentEntityPeptides": len(cache_ids),
        }
    else:
        embeddings, extraction = sequence.extract_esm(
            [sequence.normalize_for_esm(translations[gene].peptide) for gene in extract_graph],
            args.esm_model,
            device_name=args.device,
            batch_size=args.batch_size,
            max_residues=sequence.ESM_MAX_RESIDUES,
            overlap=sequence.ESM_DEFAULT_OVERLAP,
        )
        cached = np.concatenate(
            (embeddings, np.ones((len(embeddings), 1), dtype=np.float32)), axis=1
        )
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        args.embedding_cache.write_bytes(
            sequence.deterministic_npz_bytes(output_arrays(cache_ids, cached))
        )
        extraction["loadedFromCache"] = False
        extraction["cacheSha256"] = sha256_file(args.embedding_cache)
    for gene, values in zip(cache_ids, cached, strict=True):
        graph_sequence[graph_row[gene]] = values

    if sha256_file(args.original_go) != ORIGINAL_GO_SHA256:
        raise ValueError("original GO artifact drift")
    original_ids = sequence.load_entity_ids(args.original_entity_ids)
    with np.load(args.original_go, allow_pickle=False) as archive:
        original_go = archive["feature_values"]
    mapping = go.require_file(
        args.go_source / go.MAPPING_NAME,
        go.MAPPING_BYTES,
        go.MAPPING_SHA256,
        "Ensembl mapping",
    )
    gaf = go.require_file(
        args.go_source / go.GO_NAME, go.GO_BYTES, go.GO_SHA256, "GO GAF"
    )
    original_xrefs, _ = go.parse_mapping_bytes(mapping, frozenset(original_ids))
    original_terms, _ = go.parse_gaf_bytes(gaf, original_xrefs, original_ids)
    original_matrix, terms, _ = go.direct_matrix(original_terms)
    reconstructed, svd = go.fit_svd(original_matrix, 256, 731)
    component_sha = hashlib.sha256(
        svd.components_.astype(np.dtype("<f4"), copy=False).tobytes("C")
    ).hexdigest()
    if component_sha != GO_COMPONENT_SHA256 or not np.array_equal(reconstructed, original_go):
        raise ValueError("frozen GO basis failed exact reconstruction")
    projection_ids = tuple(sorted(set(graph_ids) | set(output_ids)))
    projected_xrefs, _ = go.parse_mapping_bytes(mapping, frozenset(projection_ids))
    projected_terms, go_stats = go.parse_gaf_bytes(gaf, projected_xrefs, projection_ids)
    projected_matrix, omitted_terms = frozen.fixed_term_matrix(projected_terms, terms)
    projected_go = svd.transform(projected_matrix).astype(np.float32, copy=False)
    projected_row = {gene: row for row, gene in enumerate(projection_ids)}

    graph_base = np.concatenate(
        (
            graph_sequence,
            np.stack([projected_go[projected_row[gene]] for gene in graph_ids]),
        ),
        axis=1,
    ).astype(np.float32)
    copied_graph_base = 0
    for gene, row in graph_row.items():
        prior = old_row.get(gene)
        if prior is not None:
            graph_base[row] = old_base[prior]
            copied_graph_base += 1
    output_base = np.zeros((len(output_ids), 577), dtype=np.float32)
    for row, gene in enumerate(output_ids):
        prior = old_row.get(gene)
        if prior is not None:
            output_base[row] = old_base[prior]
        else:
            translated = graph_row.get(gene)
            if translated is not None:
                output_base[row, :321] = graph_sequence[translated]
            output_base[row, 321:] = projected_go[projected_row[gene]]
    output_row = {gene: row for row, gene in enumerate(output_ids)}
    old_positions = np.asarray([output_row[gene] for gene in old_ids], dtype=np.int64)
    if output_base[old_positions].tobytes() != old_base.tobytes():
        raise ValueError("old 577-dimensional base rows changed")
    expected_presence = np.asarray([gene in translations for gene in output_ids], dtype=np.float32)
    if not np.array_equal(output_base[:, 320], expected_presence):
        raise ValueError("protein-present missingness flag mismatch")

    edges, edge_stats = load_edges(args.string_source)
    physical, physical_stats = fixed_neighborhood_features(
        output_ids, output_base, graph_ids, graph_base, edges
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), Path(frozen.__file__), Path(sequence.__file__), Path(go.__file__)):
        (source / path.name).write_bytes(path.read_bytes())

    graph_path = output / "ensembl116-translated-universe-esm-go-features.npz"
    graph_path.write_bytes(sequence.deterministic_npz_bytes(output_arrays(graph_ids, graph_base)))
    feature_path = output / "frangieh-extended-static-esm-go-fixed-physical-features.npz"
    feature_path.write_bytes(sequence.deterministic_npz_bytes(output_arrays(output_ids, physical)))
    output_entity_path = output / "entity-ids.txt"
    output_entity_path.write_bytes(lf_payload(output_ids))
    graph_entity_path = output / "graph-universe-entity-ids.txt"
    graph_entity_path.write_bytes(lf_payload(graph_ids))
    provenance_path = output / "new-embedding-provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8", newline="\n") as stream:
        for gene in extract_graph:
            item = translations[gene]
            stream.write(
                json.dumps(
                    {
                        "entityId": gene,
                        "peptideSha256": hashlib.sha256(
                            sequence.normalize_for_esm(item.peptide)
                        ).hexdigest(),
                        "selectedProteinId": f"{item.protein_id}.{item.protein_version}",
                        "selectedTranscriptId": f"{item.transcript_id}.{item.transcript_version}",
                        "residues": len(item.peptide),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    missing_ids = [gene for gene in output_ids if gene not in translations]
    manifest = {
        "schema": "slp.frangieh-fixed-universe-static-features/v1",
        "identity": {
            "taxon": TAXON,
            "namespace": "Ensembl-gene",
            "outputRows": len(output_ids),
            "outputEntityListSha256": sha256_file(output_entity_path),
            "graphUniverse": "all stable genes with a selected translation in Ensembl release 116",
            "graphUniverseRows": len(graph_ids),
            "graphUniverseEntityListSha256": sha256_file(graph_entity_path),
            "adtComponentRoster": {
                "path": str(args.adt_roster),
                "sha256": ADT_SHA256,
                "rows": 24,
                "joinToGeneFeatures": False,
            },
        },
        "outputs": {
            "features": {
                "path": feature_path.name,
                "sha256": sha256_file(feature_path),
                "shape": list(physical.shape),
            },
            "graphBaseFeatures": {
                "path": graph_path.name,
                "sha256": sha256_file(graph_path),
                "shape": list(graph_base.shape),
            },
        },
        "baseFeatures": {
            "dimensions": 577,
            "oldRowsCopiedBitwise": len(old_ids),
            "oldRowsPreservedExactly": True,
            "oldTranslatedGraphRowsCopiedBitwise": copied_graph_base,
            "sequence": {
                "dimensions": 321,
                "presentOutputRows": int(expected_presence.sum()),
                "missingOutputRows": len(missing_ids),
                "missingEntityIds": missing_ids,
                "copiedGraphEmbeddings": len(copied_graph),
                "newGraphEmbeddings": len(extract_graph),
                "extraction": extraction,
                "model": {
                    "repository": sequence.ESM_REPOSITORY,
                    "revision": sequence.ESM_REVISION,
                    "files": model_files,
                    "maxResidues": sequence.ESM_MAX_RESIDUES,
                    "overlap": sequence.ESM_DEFAULT_OVERLAP,
                    "pooling": "inverse-overlap-weighted full-residue mean; no truncation",
                },
                "sourceCounts": sequence_source_counts,
            },
            "go": {
                "dimensions": 256,
                "frozenTerms": len(terms),
                "basisComponentSha256": component_sha,
                "basisRefit": False,
                "basisReconstructedExactly": True,
                "projectionRows": len(projection_ids),
                "projectionRowsWithFrozenTerms": sum(bool(item) for item in projected_terms),
                "newOnlyTermsOmitted": omitted_terms,
                "mapping": go_stats,
            },
        },
        "physical": {
            "dimensions": 579,
            "rule": "confidence-weighted mean of 577 base features over neighbors in fixed Ensembl116 translated-gene universe; append log1p degree and presence",
            "experimentConfidenceMinimum": 700,
            **edge_stats,
            **physical_stats,
            "outputRosterAffectsNeighborUniverse": False,
        },
        "sources": {
            "oldPhysicalFeatures": {"path": str(args.old_features), "sha256": OLD_FEATURE_SHA256},
            "rnaQueryRoster": {"path": str(args.query_ids), "sha256": QUERY_SHA256},
            "go": {"gafSha256": go.GO_SHA256, "mappingSha256": go.MAPPING_SHA256},
            "string": STRING_HASHES,
            "newEmbeddingCache": {
                "path": str(args.embedding_cache),
                "sha256": sha256_file(args.embedding_cache),
                "rows": len(cache_ids),
            },
            "sourceCode": {
                path.name: sha256_file(source / path.name)
                for path in (Path(__file__), Path(frozen.__file__), Path(sequence.__file__), Path(go.__file__))
            },
        },
        "runtime": {"elapsedSeconds": time.monotonic() - started},
        "accessBoundary": {
            "molecularOutcomesRead": False,
            "testOutcomesRead": False,
            "metadataRostersOnly": True,
        },
        "limitations": [
            "Physical associations are undirected static evidence, not causal or quantitative perturbation outcomes.",
            "Zero protein vectors include noncoding and release-unresolved stable IDs and are distinguished only by the protein-present flag.",
            "ADT barcodes remain typed assay components outside the gene-feature matrix.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "finished", "manifest": manifest}, sort_keys=True))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--query-ids", type=Path, required=True)
    result.add_argument("--adt-roster", type=Path, required=True)
    result.add_argument("--old-features", type=Path, required=True)
    result.add_argument("--ensembl-source", type=Path, required=True)
    result.add_argument("--esm-model", type=Path, required=True)
    result.add_argument("--original-entity-ids", type=Path, required=True)
    result.add_argument("--original-go", type=Path, required=True)
    result.add_argument("--go-source", type=Path, required=True)
    result.add_argument("--string-source", type=Path, required=True)
    result.add_argument("--embedding-cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=16)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
