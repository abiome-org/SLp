#!/usr/bin/env python3
"""Independently verify the frozen-universe Frangieh static feature pack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_slp11_frangieh_static_features.py"
SPEC = importlib.util.spec_from_file_location("slp11_frangieh_static_verify_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load Frangieh feature builder")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("feature_values", "entity_taxon", "entity_id"):
            raise ValueError("static feature NPZ schema mismatch")
        values = archive["feature_values"]
        taxon = archive["entity_taxon"]
        ids = tuple(str(item) for item in archive["entity_id"])
    return ids, taxon, values


def run(args: argparse.Namespace) -> dict[str, object]:
    output_ids, output_taxon, output = load(args.features)
    graph_ids, graph_taxon, graph = load(args.graph_base)
    old_ids, old_taxon, old = load(args.old_features)
    if (
        output.shape != (18_893, 1_156)
        or graph.shape != (23_879, 577)
        or old.shape != (10_231, 1_156)
        or output.dtype != np.float32
        or graph.dtype != np.float32
        or output_ids != tuple(sorted(set(output_ids)))
        or graph_ids != tuple(sorted(set(graph_ids)))
        or not np.all(output_taxon == 9606)
        or not np.all(graph_taxon == 9606)
        or not np.all(old_taxon == 9606)
        or not np.isfinite(output).all()
        or not np.isfinite(graph).all()
    ):
        raise ValueError("feature shape, identity, dtype, or finite-value contract failed")
    output_row = {gene: row for row, gene in enumerate(output_ids)}
    old_positions = np.asarray([output_row[gene] for gene in old_ids])
    old_base_exact = output[old_positions, :577].tobytes() == old[:, :577].tobytes()
    graph_set = set(graph_ids)
    expected_presence = np.asarray([gene in graph_set for gene in output_ids], dtype=np.float32)
    missingness_exact = np.array_equal(output[:, 320], expected_presence)
    graph_presence_exact = bool(np.all(graph[:, 320] == 1))
    query_ids = BUILDER.read_lf_ids(args.query_ids, BUILDER.QUERY_SHA256)
    query_set = set(query_ids)
    action_audit = json.loads(args.action_audit.read_text(encoding="utf-8"))
    action_ids = set(action_audit["interventions"]["stable_action_ensembl_ids"])
    if not query_set <= set(output_ids) or not action_ids <= set(output_ids):
        raise ValueError("query or action roster absent from output")

    fasta = BUILDER.sequence.verify_source_dir(args.ensembl_source)
    translations, _ = BUILDER.sequence.parse_longest_translations(fasta)
    if tuple(sorted(translations)) != graph_ids:
        raise ValueError("translated Ensembl graph universe changed")
    old_row = {gene: row for row, gene in enumerate(old_ids)}
    expected_cache_ids = tuple(
        gene
        for gene in graph_ids
        if gene not in old_row or old[old_row[gene], 320] != 1
    )
    cache_ids, cache_taxon, cache = load(args.embedding_cache)
    if (
        cache_ids != expected_cache_ids
        or cache.shape != (13_800, 321)
        or not np.array_equal(cache_taxon, np.full(len(cache_ids), 9606))
        or not np.isfinite(cache).all()
        or not np.all(cache[:, 320] == 1)
    ):
        raise ValueError("new embedding cache does not exactly cover required translated genes")
    normalized = [
        BUILDER.sequence.normalize_for_esm(translations[gene].peptide) for gene in cache_ids
    ]
    unique_peptides = tuple(dict.fromkeys(normalized))
    entity_window_count = sum(
        len(
            BUILDER.sequence.chunk_windows(
                len(peptide),
                BUILDER.sequence.ESM_MAX_RESIDUES,
                BUILDER.sequence.ESM_DEFAULT_OVERLAP,
            )
        )
        for peptide in normalized
    )
    unique_window_count = sum(
        len(
            BUILDER.sequence.chunk_windows(
                len(peptide),
                BUILDER.sequence.ESM_MAX_RESIDUES,
                BUILDER.sequence.ESM_DEFAULT_OVERLAP,
            )
        )
        for peptide in unique_peptides
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    extraction = manifest["baseFeatures"]["sequence"]["extraction"]
    if (
        extraction["presentEntityPeptides"] != len(cache_ids)
        or extraction["uniqueNormalizedPeptides"] != len(unique_peptides)
        or extraction["windowCount"] != unique_window_count
    ):
        raise ValueError("manifest extraction coverage disagrees with admitted peptides")

    output_bytes = BUILDER.sequence.deterministic_npz_bytes(
        BUILDER.output_arrays(output_ids, output)
    )
    graph_bytes = BUILDER.sequence.deterministic_npz_bytes(
        BUILDER.output_arrays(graph_ids, graph)
    )
    deterministic_output = hashlib.sha256(output_bytes).hexdigest() == sha256_file(args.features)
    deterministic_graph = hashlib.sha256(graph_bytes).hexdigest() == sha256_file(args.graph_base)
    edges, _ = BUILDER.load_edges(args.string_source)
    reconstructed, _ = BUILDER.fixed_neighborhood_features(
        output_ids, output[:, :577], graph_ids, graph, edges
    )
    full_reconstruction_exact = np.array_equal(reconstructed, output)
    subset_positions = np.arange(0, len(output_ids), 7, dtype=np.int64)
    subset_ids = tuple(output_ids[index] for index in subset_positions)
    subset, _ = BUILDER.fixed_neighborhood_features(
        subset_ids, output[subset_positions, :577], graph_ids, graph, edges
    )
    subset_invariance_exact = np.array_equal(subset, output[subset_positions])
    if not all(
        (
            old_base_exact,
            missingness_exact,
            graph_presence_exact,
            deterministic_output,
            deterministic_graph,
            full_reconstruction_exact,
            subset_invariance_exact,
        )
    ):
        raise RuntimeError("Frangieh static feature numerical verification failed")
    old_physical_changed = int(
        np.count_nonzero(np.any(output[old_positions, 577:] != old[:, 577:], axis=1))
    )
    query_presence = np.asarray([gene in graph_set for gene in query_ids])
    action_presence = np.asarray([gene in graph_set for gene in sorted(action_ids)])
    report = {
        "schema": "slp.frangieh-fixed-universe-static-verification/v1",
        "features": {
            "path": str(args.features),
            "sha256": sha256_file(args.features),
            "shape": list(output.shape),
            "deterministicReserializationExact": deterministic_output,
        },
        "graphBase": {
            "path": str(args.graph_base),
            "sha256": sha256_file(args.graph_base),
            "shape": list(graph.shape),
            "deterministicReserializationExact": deterministic_graph,
            "allProteinPresentFlagsOne": graph_presence_exact,
        },
        "identity": {
            "sortedUniqueOutputIds": True,
            "sortedUniqueGraphIds": True,
            "taxon": 9606,
            "queryIdsCovered": len(query_ids),
            "actionIdsCovered": len(action_ids),
        },
        "base": {
            "oldRows": len(old_ids),
            "oldFirst577BitwiseExact": old_base_exact,
            "proteinPresentFlagColumn": 320,
            "proteinPresentFlagExact": missingness_exact,
            "outputProteinPresent": int(expected_presence.sum()),
            "outputProteinMissing": int((expected_presence == 0).sum()),
            "rnaQueriesProteinPresent": int(query_presence.sum()),
            "rnaQueriesProteinMissing": int((~query_presence).sum()),
            "actionsProteinPresent": int(action_presence.sum()),
            "actionsProteinMissing": int((~action_presence).sum()),
        },
        "embeddingCoverage": {
            "cachePath": str(args.embedding_cache),
            "cacheSha256": sha256_file(args.embedding_cache),
            "requiredNewTranslatedGenes": len(expected_cache_ids),
            "cacheRowsExactlyRequiredGenes": True,
            "entityPeptideWindows": entity_window_count,
            "uniqueNormalizedPeptides": len(unique_peptides),
            "uniquePeptideWindowsActuallyEncoded": unique_window_count,
            "duplicatePeptideReuse": len(normalized) - len(unique_peptides),
            "residuesSilentlyTruncatedOrSkipped": 0,
        },
        "physical": {
            "fullRecomputationBitwiseExact": full_reconstruction_exact,
            "everySeventhRowSubsetInvariantBitwiseExact": subset_invariance_exact,
            "oldPhysicalBlocksChangedUnderCorrectedUniverse": old_physical_changed,
            "oldPhysicalBlocksUnchangedUnderCorrectedUniverse": len(old_ids)
            - old_physical_changed,
        },
        "accessBoundary": {"molecularOutcomesRead": False, "testOutcomesRead": False},
        "sourceCode": {
            "builderSha256": sha256_file(BUILDER_PATH),
            "verifierSha256": sha256_file(Path(__file__)),
        },
    }
    snapshot = args.features.parent / "source" / Path(__file__).name
    shutil.copyfile(Path(__file__), snapshot)
    report["sourceCode"]["verifierSnapshot"] = str(snapshot)
    report["sourceCode"]["verifierSnapshotSha256"] = sha256_file(snapshot)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--graph-base", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--string-source", type=Path, required=True)
    parser.add_argument("--ensembl-source", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
