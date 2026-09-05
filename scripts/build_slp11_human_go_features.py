#!/usr/bin/env python3
"""Build deterministic human GO MF/CC features over the fixed ENSG universe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy
import sklearn
from build_slp11_sequence_features import deterministic_npz_bytes
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

SPECIES_TAXON = 9606
ENTITY_COUNT = 7_542
ENTITY_LIST_BYTES = 120_672
ENTITY_LIST_SHA256 = "c6836645dcfc24788f2c06110ddc08ee4949d97f710dd117db12db1949d9b33e"
QUERY_COUNT = 7_226
QUERY_LIST_SHA256 = "645b8d563b440a4b7ab6a3bb42450594b408c4e7cb84e4fe2789a6620174f12c"
ACTION_COUNT = 2_392
ACTION_LIST_SHA256 = "2884efd414949bfc3c7dc5f376aa69f0470080afdcab255b4a88f67cc53ac9ed"
ENTITY_RE = re.compile(r"^ENSG[0-9]+$")

GO_RELEASE = "2022-09-19"
GO_URL = "https://release.geneontology.org/2022-09-19/annotations/goa_human.gaf.gz"
GO_NAME = "goa_human_2022-09-19.gaf.gz"
GO_BYTES = 12_187_638
GO_SHA256 = "8b97980a895cb74255615f7cdbdd818f72a3999867b7d2a14f867874480693e1"
GO_DECOMPRESSED_BYTES = 114_549_201
GO_DECOMPRESSED_SHA256 = (
    "e543c1967514db1a2d4512f6f750d1b03bb227d22409df3883be740ca347e52c"
)

ENSEMBL_RELEASE = 108
ENSEMBL_MAPPING_URL = (
    "https://ftp.ensembl.org/pub/release-108/tsv/homo_sapiens/"
    "Homo_sapiens.GRCh38.108.uniprot.tsv.gz"
)
MAPPING_NAME = "Homo_sapiens.GRCh38.108.uniprot.tsv.gz"
MAPPING_BYTES = 1_835_509
MAPPING_SHA256 = "0d6fe982ce7023b2901171fd0e1419a2e9fbd7fbb9b3473a82ac6dee454f6e56"
MAPPING_BSD_SUM = 7_358
MAPPING_BLOCKS = 1_793
MAPPING_DECOMPRESSED_BYTES = 12_832_401
MAPPING_DECOMPRESSED_SHA256 = (
    "61e28ba0068690f02940cd29aff384d37dd1c537dfab2a6b88946c78216fbde6"
)
CHECKSUMS_SPEC = (
    516,
    "04dac451c23fb1184b4a1cf903fc6f146c21bfeddf2aaf10929cba19bb9a395c",
)
README_SPEC = (
    745,
    "6f11ca06a61d951b5bd2282857b9431c2ddd6c0d5df1146d12266bdd73d8bfc4",
)
MAPPING_HEADER = (
    "gene_stable_id\ttranscript_stable_id\tprotein_stable_id\txref\tdb_name\t"
    "info_type\tsource_identity\txref_identity\tlinkage_type"
)
UNIPROT_RE = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
TRANSCRIPT_RE = re.compile(r"^ENST[0-9]+$")
PROTEIN_RE = re.compile(r"^ENSP[0-9]+$")
GO_RE = re.compile(r"^GO:[0-9]{7}$")

DATE_CUTOFF = "20221231"
ALLOWED_ASPECTS = {"F": "molecular_function", "C": "cellular_component"}
EXCLUDED_EVIDENCE = frozenset({"IMP", "IGI", "IEP", "HMP", "HGI", "HEP"})
DEFAULT_COMPONENTS = 256
DEFAULT_SEED = 731
NPZ_NAME = "human-go-mf-cc-svd-features.npz"
MANIFEST_NAME = "human-go-mf-cc-svd-features.manifest.json"
OUTPUT_SCHEMA = "slp.human-go-direct-svd-feature-artifact/v1"


class HumanGoFeatureError(ValueError):
    """Raised when the fixed static GO feature contract is violated."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int, digest: str, label: str) -> bytes:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise HumanGoFeatureError(f"{label} does not match its pinned byte identity")
    return path.read_bytes()


def bsd_sum(payload: bytes) -> tuple[int, int]:
    checksum = 0
    for byte in payload:
        checksum = (checksum >> 1) | ((checksum & 1) << 15)
        checksum = (checksum + byte) & 0xFFFF
    return checksum, (len(payload) + 1023) // 1024


def load_entity_ids(path: Path) -> tuple[str, ...]:
    payload = require_file(path, ENTITY_LIST_BYTES, ENTITY_LIST_SHA256, "entity list")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise HumanGoFeatureError("entity list must be LF-terminated ASCII")
    try:
        identifiers = tuple(payload.decode("ascii").splitlines())
    except UnicodeDecodeError as exc:
        raise HumanGoFeatureError("entity list must be ASCII") from exc
    if (
        len(identifiers) != ENTITY_COUNT
        or list(identifiers) != sorted(set(identifiers))
        or any(ENTITY_RE.fullmatch(item) is None for item in identifiers)
    ):
        raise HumanGoFeatureError("entity list stable ENSG contract mismatch")
    return identifiers


def parse_mapping_bytes(
    payload: bytes, entity_ids: frozenset[str]
) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    try:
        decompressed = gzip.decompress(payload)
        lines = decompressed.decode("ascii").splitlines()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise HumanGoFeatureError(
            "Ensembl mapping must be a complete ASCII gzip"
        ) from exc
    if not lines or lines[0] != MAPPING_HEADER:
        raise HumanGoFeatureError("unexpected Ensembl UniProt TSV header")
    xref_to_genes: dict[str, set[str]] = {}
    total_rows = 0
    retained_rows = 0
    database_counts: Counter[str] = Counter()
    mapped_genes: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        columns = line.split("\t")
        if len(columns) != 9:
            raise HumanGoFeatureError(
                f"Ensembl mapping row {line_number} is not 9 columns"
            )
        gene, transcript, protein, xref, database = columns[:5]
        if (
            ENTITY_RE.fullmatch(gene) is None
            or TRANSCRIPT_RE.fullmatch(transcript) is None
            or PROTEIN_RE.fullmatch(protein) is None
            or UNIPROT_RE.fullmatch(xref) is None
            or not database.startswith("Uniprot")
        ):
            raise HumanGoFeatureError(
                f"invalid stable identity at mapping row {line_number}"
            )
        total_rows += 1
        database_counts[database] += 1
        if gene in entity_ids:
            xref_to_genes.setdefault(xref, set()).add(gene)
            mapped_genes.add(gene)
            retained_rows += 1
    frozen = {xref: frozenset(genes) for xref, genes in sorted(xref_to_genes.items())}
    stats: dict[str, object] = {
        "totalRows": total_rows,
        "retainedUniverseMappingRows": retained_rows,
        "mappedUniverseGenes": len(mapped_genes),
        "unmappedUniverseGenes": len(entity_ids - mapped_genes),
        "retainedUniqueUniProtXrefs": len(frozen),
        "databaseCounts": dict(sorted(database_counts.items())),
        "decompressedBytes": len(decompressed),
        "decompressedSha256": sha256_bytes(decompressed),
    }
    return frozen, stats


def parse_gaf_bytes(
    payload: bytes,
    xref_to_genes: Mapping[str, frozenset[str]],
    entity_ids: Sequence[str],
) -> tuple[list[frozenset[str]], dict[str, object]]:
    try:
        decompressed = gzip.decompress(payload)
        lines = decompressed.decode("utf-8").splitlines()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise HumanGoFeatureError(
            "GO source must be a complete UTF-8 gzip GAF"
        ) from exc
    headers = [line for line in lines if line.startswith("!")]
    if "!gaf-version: 2.2" not in headers:
        raise HumanGoFeatureError("GO source must declare GAF 2.2")
    terms_by_gene: dict[str, set[str]] = {entity: set() for entity in entity_ids}
    discarded: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    aspect_counts: Counter[str] = Counter()
    mapped_xrefs: set[str] = set()
    selected_xrefs: set[str] = set()
    total_rows = 0
    selected_before_mapping = 0
    mapped_rows = 0
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("!"):
            continue
        total_rows += 1
        columns = line.split("\t")
        if len(columns) != 17:
            raise HumanGoFeatureError(f"GAF row {line_number} is not 17 columns")
        database, xref = columns[0], columns[1]
        qualifier, term, evidence = columns[3], columns[4], columns[6]
        aspect, taxon, annotation_date = columns[8], columns[12], columns[13]
        if database != "UniProtKB" or UNIPROT_RE.fullmatch(xref) is None:
            raise HumanGoFeatureError(f"GAF row {line_number} lacks a UniProt identity")
        if (
            GO_RE.fullmatch(term) is None
            or re.fullmatch(r"taxon:9606(?:\|taxon:[0-9]+)?", taxon) is None
        ):
            raise HumanGoFeatureError(
                f"GAF row {line_number} has invalid term or taxon"
            )
        if re.fullmatch(r"[0-9]{8}", annotation_date) is None:
            raise HumanGoFeatureError(
                f"GAF row {line_number} has invalid annotation date"
            )
        if aspect not in ALLOWED_ASPECTS:
            discarded["biological-process-aspect"] += 1
            continue
        if annotation_date > DATE_CUTOFF:
            discarded["after-date-cutoff"] += 1
            continue
        if "NOT" in qualifier.split("|"):
            discarded["negated-qualifier"] += 1
            continue
        if evidence in EXCLUDED_EVIDENCE:
            discarded["perturbation-derived-evidence"] += 1
            continue
        selected_before_mapping += 1
        selected_xrefs.add(xref)
        genes = xref_to_genes.get(xref, frozenset())
        if not genes:
            discarded["no-exact-ensembl-universe-mapping"] += 1
            continue
        for gene in genes:
            terms_by_gene[gene].add(term)
        mapped_xrefs.add(xref)
        mapped_rows += 1
        evidence_counts[evidence] += 1
        aspect_counts[ALLOWED_ASPECTS[aspect]] += 1
    frozen = [frozenset(terms_by_gene[entity]) for entity in entity_ids]
    if not any(frozen):
        raise HumanGoFeatureError("no eligible GO annotation maps to the ENSG universe")
    stats: dict[str, object] = {
        "gafVersion": "2.2",
        "dateGeneratedHeaders": sorted(
            line.removeprefix("!date-generated: ")
            for line in headers
            if line.startswith("!date-generated: ")
        ),
        "totalAnnotationRows": total_rows,
        "selectedRowsBeforeMapping": selected_before_mapping,
        "mappedAnnotationRowsBeforeDeduplication": mapped_rows,
        "selectedUniqueUniProtXrefs": len(selected_xrefs),
        "mappedUniqueUniProtXrefs": len(mapped_xrefs),
        "unmappedSelectedUniProtXrefs": len(selected_xrefs - mapped_xrefs),
        "selectedEvidenceCounts": dict(sorted(evidence_counts.items())),
        "selectedAspectCounts": dict(sorted(aspect_counts.items())),
        "discardedRows": dict(sorted(discarded.items())),
        "decompressedBytes": len(decompressed),
        "decompressedSha256": sha256_bytes(decompressed),
    }
    return frozen, stats


def direct_matrix(
    entity_terms: Sequence[frozenset[str]],
) -> tuple[sparse.csr_matrix, tuple[str, ...], dict[str, int]]:
    terms = tuple(sorted({term for row in entity_terms for term in row}))
    if not terms:
        raise HumanGoFeatureError("mapped GO term vocabulary is empty")
    term_index = {term: index for index, term in enumerate(terms)}
    rows: list[int] = []
    columns: list[int] = []
    for row, direct in enumerate(entity_terms):
        for term in sorted(direct):
            rows.append(row)
            columns.append(term_index[term])
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(entity_terms), len(terms)),
        dtype=np.float32,
    )
    coverage = {
        "entities": len(entity_terms),
        "entitiesWithEligibleDirectTerms": sum(bool(row) for row in entity_terms),
        "zeroCoverageEntities": sum(not row for row in entity_terms),
        "directTermsMapped": len(terms),
        "binaryAssociations": int(matrix.nnz),
    }
    return matrix, terms, coverage


def fit_svd(
    matrix: sparse.csr_matrix,
    components: int = DEFAULT_COMPONENTS,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, TruncatedSVD]:
    if type(components) is not int or components < 1 or type(seed) is not int:
        raise HumanGoFeatureError("invalid SVD configuration")
    count = min(components, matrix.shape[1])
    model = TruncatedSVD(
        n_components=count,
        algorithm="randomized",
        n_iter=7,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=seed,
    )
    values = model.fit_transform(matrix).astype(np.dtype("<f4"), copy=False)
    if values.shape != (matrix.shape[0], count) or not np.isfinite(values).all():
        raise HumanGoFeatureError("SVD generated invalid feature values")
    return values, model


def build_artifact(
    entity_list: Path,
    source_dir: Path,
    output_dir: Path,
    *,
    components: int,
    seed: int,
) -> dict[str, object]:
    identifiers = load_entity_ids(entity_list)
    mapping_payload = require_file(
        source_dir / MAPPING_NAME, MAPPING_BYTES, MAPPING_SHA256, "Ensembl mapping"
    )
    go_payload = require_file(source_dir / GO_NAME, GO_BYTES, GO_SHA256, "GO GAF")
    require_file(source_dir / "ENSEMBL_CHECKSUMS", *CHECKSUMS_SPEC, "Ensembl CHECKSUMS")
    require_file(source_dir / "README_uniprot.tsv", *README_SPEC, "Ensembl README")
    if bsd_sum(mapping_payload) != (MAPPING_BSD_SUM, MAPPING_BLOCKS):
        raise HumanGoFeatureError("Ensembl mapping upstream BSD checksum mismatch")
    checksum_lines = (source_dir / "ENSEMBL_CHECKSUMS").read_text("ascii").splitlines()
    if f"{MAPPING_BSD_SUM:05d}  {MAPPING_BLOCKS} {MAPPING_NAME}" not in checksum_lines:
        raise HumanGoFeatureError("Ensembl mapping is absent from pinned CHECKSUMS")
    xref_to_genes, mapping_stats = parse_mapping_bytes(
        mapping_payload, frozenset(identifiers)
    )
    if (
        mapping_stats["decompressedBytes"] != MAPPING_DECOMPRESSED_BYTES
        or mapping_stats["decompressedSha256"] != MAPPING_DECOMPRESSED_SHA256
    ):
        raise HumanGoFeatureError("Ensembl mapping decompressed identity mismatch")
    entity_terms, gaf_stats = parse_gaf_bytes(go_payload, xref_to_genes, identifiers)
    if (
        gaf_stats["decompressedBytes"] != GO_DECOMPRESSED_BYTES
        or gaf_stats["decompressedSha256"] != GO_DECOMPRESSED_SHA256
    ):
        raise HumanGoFeatureError("GO GAF decompressed identity mismatch")
    matrix, terms, coverage = direct_matrix(entity_terms)
    values, svd = fit_svd(matrix, components, seed)
    arrays = {
        "feature_values": values,
        "entity_taxon": np.full(len(identifiers), SPECIES_TAXON, dtype=np.dtype("<i8")),
        "entity_id": np.asarray(identifiers, dtype="<U15"),
    }
    npz_payload = deterministic_npz_bytes(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / NPZ_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if npz_path.exists() or manifest_path.exists():
        raise HumanGoFeatureError(f"refusing to overwrite output in {output_dir}")
    npz_path.write_bytes(npz_payload)
    component_bytes = svd.components_.astype(np.dtype("<f4"), copy=False).tobytes("C")
    manifest: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "rows": len(identifiers),
            "ncbiTaxon": SPECIES_TAXON,
            "entityNamespace": "Ensembl-gene",
            "entityIdList": {
                "count": ENTITY_COUNT,
                "bytes": ENTITY_LIST_BYTES,
                "sha256": ENTITY_LIST_SHA256,
            },
            "queryIdList": {"count": QUERY_COUNT, "sha256": QUERY_LIST_SHA256},
            "actionIdList": {"count": ACTION_COUNT, "sha256": ACTION_LIST_SHA256},
        },
        "arrays": {
            "feature_values": {
                "dtype": "little-endian-float32",
                "shape": list(values.shape),
            },
            "entity_taxon": {
                "dtype": "little-endian-int64",
                "shape": list(arrays["entity_taxon"].shape),
            },
            "entity_id": {
                "dtype": str(arrays["entity_id"].dtype),
                "shape": list(arrays["entity_id"].shape),
                "pickleRequired": False,
            },
        },
        "sources": {
            "goAnnotations": {
                "publisher": "Gene Ontology Consortium",
                "release": GO_RELEASE,
                "url": GO_URL,
                "gzipBytes": GO_BYTES,
                "gzipSha256": GO_SHA256,
                "decompressedBytes": GO_DECOMPRESSED_BYTES,
                "decompressedSha256": GO_DECOMPRESSED_SHA256,
            },
            "ensemblMapping": {
                "publisher": "Ensembl, EMBL-EBI",
                "release": ENSEMBL_RELEASE,
                "assembly": "GRCh38",
                "url": ENSEMBL_MAPPING_URL,
                "gzipBytes": MAPPING_BYTES,
                "gzipSha256": MAPPING_SHA256,
                "upstreamBsdSum": MAPPING_BSD_SUM,
                "upstreamBlocks1024": MAPPING_BLOCKS,
                "decompressedBytes": MAPPING_DECOMPRESSED_BYTES,
                "decompressedSha256": MAPPING_DECOMPRESSED_SHA256,
            },
            "declaration": "sources/goa-human-ensembl-2022-09-19.yaml",
            "rights": "rights/goa-human-ensembl-2022-static-mapping.yaml",
        },
        "filter": {
            "aspects": dict(sorted(ALLOWED_ASPECTS.items())),
            "annotationDateMaximumInclusive": "2022-12-31",
            "excludedQualifier": "NOT",
            "excludedEvidence": sorted(EXCLUDED_EVIDENCE),
            "directTermsOnly": True,
            "ancestorPropagation": False,
            "termList": list(terms),
            "termListSha256": sha256_bytes(("\n".join(terms) + "\n").encode()),
            "mappingJoin": "exact GAF UniProtKB object ID to exact Ensembl xref, then stable ENSG",
            "symbolsUsedForIdentity": False,
        },
        "mappingStatistics": mapping_stats,
        "annotationStatistics": gaf_stats,
        "coverage": coverage,
        "compression": {
            "method": "sklearn.decomposition.TruncatedSVD",
            "algorithm": "randomized",
            "seed": seed,
            "iterations": 7,
            "oversamples": 10,
            "powerIterationNormalizer": "auto",
            "requestedComponents": components,
            "components": int(values.shape[1]),
            "fitRows": len(identifiers),
            "componentFloat32Sha256": sha256_bytes(component_bytes),
            "explainedVarianceRatioSum": float(svd.explained_variance_ratio_.sum()),
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "scikitLearnVersion": sklearn.__version__,
        },
        "accessBoundary": {
            "staticAnnotationsConsumed": True,
            "staticIdentifierMappingConsumed": True,
            "expressionTargetsConsumed": False,
            "quantitativeOutcomesConsumed": False,
            "partitionAssignmentsConsumed": False,
            "benchmarkDataConsumed": False,
            "perturbationDerivedEvidenceConsumed": False,
        },
        "artifact": {
            "path": NPZ_NAME,
            "bytes": len(npz_payload),
            "sha256": sha256_bytes(npz_payload),
            "compression": "zip-deflate-level-9-fixed-metadata",
        },
        "status": "exploratory-static-feature-artifact-not-omf-admitted",
        "limitations": [
            "Direct annotations omit ontology ancestors and may understate shared function.",
            "Zero vectors mean no eligible mapped direct annotation, not absence of biological function.",
            "The SVD basis is transductively fit over static annotation rows only and consumes no outcomes.",
            "The exact UniProt-to-ENSG mapping can omit or multiply map accessions; coverage is reported explicitly.",
            "This exploratory artifact is not an admitted OMF DatasetSnapshot.",
        ],
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--entity-list", type=Path, required=True)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = build_artifact(
            args.entity_list,
            args.source_dir,
            args.output_dir,
            components=args.components,
            seed=args.seed,
        )
    except (OSError, HumanGoFeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
