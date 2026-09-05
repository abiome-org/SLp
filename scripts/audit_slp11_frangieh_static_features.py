#!/usr/bin/env python3
"""Audit metadata-only Frangieh coverage for a stable human feature universe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_slp11_human_sequence_features as sequence

GTF_BYTES = 141_121_632
GTF_SHA256 = "ed992f0eac7197d9627bda618f8f831ba355c95bd5d0796af785387d462828b6"
GTF_BSD_SUM = 49_151
GTF_BLOCKS = 137_815
QUERY_SHA256 = "87bac5ddbe3a1546d49896b3e1135efd087260a3db5704d6946d9de7a36fc14a"
ADT_SHA256 = "a23afdb6c0214fc79d429aac28a9b0b17599196f7768ab460d16a3ed34d1e3f8"
OLD_FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
STRING_HASHES = {
    "9606.protein.aliases.v12.0.txt.gz": "b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680",
    "9606.protein.physical.links.full.v12.0.txt.gz": "b28f494f58e1ace634ef1fe41734ada5be37f151e3168bb9658bc6ca1dd1a954",
}
GENE_RE = re.compile(r"^ENSG\d{11}$")
ATTRIBUTE_RE = re.compile(r'(\w+) "([^"]*)"')


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bsd_sum(path: Path) -> tuple[int, int]:
    checksum = 0
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            byte_count += len(chunk)
            for byte in chunk:
                checksum = (checksum >> 1) | ((checksum & 1) << 15)
                checksum = (checksum + byte) & 0xFFFF
    return checksum, (byte_count + 1023) // 1024


def read_lf_ids(path: Path, expected_sha: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("metadata roster hash mismatch")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("metadata roster must be LF-terminated")
    ids = tuple(payload.decode("ascii").splitlines())
    if len(ids) != len(set(ids)) or any(GENE_RE.fullmatch(item) is None for item in ids):
        raise ValueError("metadata roster identity mismatch")
    return ids


def parse_gtf_genes(path: Path) -> dict[str, str]:
    if path.stat().st_size != GTF_BYTES or sha256_file(path) != GTF_SHA256:
        raise ValueError("Ensembl 116 GTF hash or size mismatch")
    if bsd_sum(path) != (GTF_BSD_SUM, GTF_BLOCKS):
        raise ValueError("Ensembl 116 GTF upstream BSD checksum mismatch")
    genes: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = dict(ATTRIBUTE_RE.findall(fields[8]))
            gene = attributes.get("gene_id", "").split(".", 1)[0]
            biotype = attributes.get("gene_biotype")
            if GENE_RE.fullmatch(gene) is None or not biotype:
                raise ValueError("Ensembl 116 GTF gene identity/biotype missing")
            previous = genes.setdefault(gene, biotype)
            if previous != biotype:
                raise ValueError("Ensembl 116 stable gene has conflicting biotypes")
    return genes


def roster_coverage(
    ids: set[str], old: set[str], peptides: set[str], biotypes: dict[str, str]
) -> dict[str, object]:
    missing_peptide = ids - peptides
    current_missing = missing_peptide & set(biotypes)
    unresolved = missing_peptide - set(biotypes)
    breakdown = Counter(biotypes[item] for item in current_missing)
    return {
        "entities": len(ids),
        "existingPhysicalFeatureRows": len(ids & old),
        "missingExistingPhysicalFeatureRows": len(ids - old),
        "withEnsembl116SelectedPeptide": len(ids & peptides),
        "withoutEnsembl116SelectedPeptide": len(missing_peptide),
        "notPresentInEnsembl116Gtf": len(ids - set(biotypes)),
        "withoutPeptideAndNotPresentInEnsembl116Gtf": len(unresolved),
        "withoutPeptideButPresentInEnsembl116Gtf": len(current_missing),
        "lncRnaWithoutPeptide": sum(
            1 for item in current_missing if biotypes[item] == "lncRNA"
        ),
        "proteinCodingWithoutPeptide": sum(
            1 for item in current_missing if biotypes[item] == "protein_coding"
        ),
        "otherCurrentBiotypesWithoutPeptide": len(current_missing)
        - sum(1 for item in current_missing if biotypes[item] in {"lncRNA", "protein_coding"}),
        "missingPeptideBiotypeCounts": dict(sorted(breakdown.items())),
        "withoutPeptideAndNotPresentInEnsembl116GtfIds": sorted(unresolved),
    }


def physical_graph(
    source: Path, old: set[str], requested: set[str], peptides: set[str]
) -> dict[str, object]:
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
    edges: dict[tuple[str, str], int] = {}
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
            edges[pair] = max(edges.get(pair, 0), confidence)
    global_neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        global_neighbors[left].add(right)
        global_neighbors[right].add(left)

    def induced(universe: set[str]) -> tuple[int, int]:
        selected = [(left, right) for left, right in edges if left in universe and right in universe]
        nodes = {gene for edge in selected for gene in edge}
        return len(selected), len(nodes)

    old_edges, old_nodes = induced(old)
    requested_edges, requested_nodes = induced(requested)
    translated_edges, translated_nodes = induced(peptides)
    omitted = {
        gene for gene in old if any(neighbor not in old for neighbor in global_neighbors[gene])
    }
    return {
        "strongSourceRows": strong_rows,
        "uniqueExactGeneEdgesAllMappedStableIds": len(edges),
        "genesWithAnyMappedStrongEdge": len(global_neighbors),
        "oldRoster": {"edges": old_edges, "entitiesWithNeighbors": old_nodes},
        "requestedRosterInduced": {
            "edges": requested_edges,
            "entitiesWithNeighbors": requested_nodes,
        },
        "fixedEnsembl116TranslatedUniverse": {
            "genes": len(peptides),
            "edges": translated_edges,
            "entitiesWithNeighbors": translated_nodes,
        },
        "oldEntitiesWithStrongNeighborsOmittedByOldRoster": len(omitted),
        "oldPhysicalFeaturesRosterDependent": bool(omitted),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    queries = set(read_lf_ids(args.query_ids, QUERY_SHA256))
    audit = json.loads(args.metadata_audit.read_text(encoding="utf-8"))
    actions = set(audit["interventions"]["stable_action_ensembl_ids"])
    if len(actions) != 237 or any(GENE_RE.fullmatch(item) is None for item in actions):
        raise ValueError("Frangieh action metadata contract mismatch")
    if sha256_file(args.adt_roster) != ADT_SHA256:
        raise ValueError("ADT roster hash mismatch")
    adt = json.loads(args.adt_roster.read_text(encoding="utf-8"))["channels"]
    if len(adt) != 24 or len({item["channel_id"] for item in adt}) != 24:
        raise ValueError("ADT component roster contract mismatch")
    with np.load(args.old_features, allow_pickle=False) as archive:
        if sha256_file(args.old_features) != OLD_FEATURE_SHA256:
            raise ValueError("old physical feature pack drift")
        old_ids = set(archive["entity_id"].tolist())
        old_presence = {
            str(gene): bool(value)
            for gene, value in zip(
                archive["entity_id"], archive["feature_values"][:, 320], strict=True
            )
        }
    fasta = sequence.verify_source_dir(args.ensembl_peptide_source)
    translations, fasta_counts = sequence.parse_longest_translations(fasta)
    peptide_ids = set(translations)
    biotypes = parse_gtf_genes(args.ensembl_gtf)
    requested = queries | actions
    windows = sum(
        len(sequence.chunk_windows(len(translation.peptide)))
        for translation in translations.values()
    )
    missing_species_embeddings = peptide_ids - {
        gene for gene, present in old_presence.items() if present
    }
    report = {
        "schema": "slp.frangieh-static-feature-coverage-audit/v1",
        "accessBoundary": {
            "metadataOnly": True,
            "molecularOutcomeArraysRead": False,
            "testOutcomesRead": False,
        },
        "inputs": {
            "rnaQueryRosterSha256": QUERY_SHA256,
            "adtRosterSha256": ADT_SHA256,
            "oldPhysicalFeaturesSha256": OLD_FEATURE_SHA256,
            "ensembl116Gtf": {
                "bytes": GTF_BYTES,
                "sha256": GTF_SHA256,
                "upstreamBsdSum": GTF_BSD_SUM,
                "upstreamBlocks1024": GTF_BLOCKS,
            },
            "string": STRING_HASHES,
        },
        "coverage": {
            "rnaQueries": roster_coverage(queries, old_ids, peptide_ids, biotypes),
            "actions": roster_coverage(actions, old_ids, peptide_ids, biotypes),
            "requestedGeneUnion": roster_coverage(requested, old_ids, peptide_ids, biotypes),
            "actionGenesAlsoRnaQueries": len(actions & queries),
            "oldFeatureProteinPresentRows": sum(old_presence.values()),
        },
        "ensembl116": {
            "gtfStableGenes": len(biotypes),
            "gtfGeneBiotypeCounts": dict(sorted(Counter(biotypes.values()).items())),
            "selectedPeptideGenes": len(peptide_ids),
            "selectedPeptideWindows": windows,
            "selectedPeptideSourceCounts": fasta_counts,
            "specieswideNewEmbeddingsNeededBeyondOldPack": len(missing_species_embeddings),
            "specieswideNewEmbeddingWindows": sum(
                len(sequence.chunk_windows(len(translations[gene].peptide)))
                for gene in missing_species_embeddings
            ),
        },
        "physical": physical_graph(args.string_source, old_ids, requested, peptide_ids),
        "adt": {
            "stableAssayComponents": len(adt),
            "molecularTargetComponents": sum(item["role"] == "molecular-target" for item in adt),
            "isotypeQcComponents": sum(item["role"] == "isotype-qc" for item in adt),
            "identity": "TotalSeq-A barcode; no protein-label-to-ENSG join",
        },
        "proposal": {
            "geneBase": "Extend the exact ESM2-8M 321 and frozen 4414-term GO-basis 256 blocks to the requested ENSG union; copy all existing 577 values exactly.",
            "physical": "Create a new physical-feature version whose confidence-weighted neighbor summaries are computed against a pinned specieswide Ensembl116 translated-gene universe, then subset requested output rows. Do not append genes to the old induced graph.",
            "adt": "Keep 24 TotalSeq-A barcodes as a separate typed assay-component roster. Preserve role and matched-isotype metadata; do not fabricate ENSG identities from display labels.",
            "missingProtein": "Retain zero ESM vectors plus the explicit protein-present flag; report biotype and release-absence separately.",
            "futureNucleotide": "The downloaded Ensembl116 GTF supplies gene biotype. A separately pinned Ensembl116 ncRNA/DNA FASTA would be required for nucleotide features; no such sequence was consumed here.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--query-ids", type=Path, required=True)
    result.add_argument("--adt-roster", type=Path, required=True)
    result.add_argument("--metadata-audit", type=Path, required=True)
    result.add_argument("--old-features", type=Path, required=True)
    result.add_argument("--ensembl-peptide-source", type=Path, required=True)
    result.add_argument("--ensembl-gtf", type=Path, required=True)
    result.add_argument("--string-source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
