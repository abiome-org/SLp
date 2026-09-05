#!/usr/bin/env python3
"""Build a source-three-fit human GO biological-process descriptor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

TAXON = 9606
SEED = 731
COMPONENTS = 128
GO_NAME = "goa_human_2022-09-19.gaf.gz"
GO_BYTES = 12_187_638
GO_SHA256 = "8b97980a895cb74255615f7cdbdd818f72a3999867b7d2a14f867874480693e1"
GO_DECOMPRESSED_SHA256 = "e543c1967514db1a2d4512f6f750d1b03bb227d22409df3883be740ca347e52c"
MAPPING_NAME = "Homo_sapiens.GRCh38.108.uniprot.tsv.gz"
MAPPING_BYTES = 1_835_509
MAPPING_SHA256 = "0d6fe982ce7023b2901171fd0e1419a2e9fbd7fbb9b3473a82ac6dee454f6e56"
MAPPING_DECOMPRESSED_SHA256 = "61e28ba0068690f02940cd29aff384d37dd1c537dfab2a6b88946c78216fbde6"
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
GRAPH_ROSTER_SHA256 = "1e313149a2bd0e438ea37152f7ccf680c54aeef2527d869c0b2f6508c1cfdce0"
OUTPUT_ROSTER_SHA256 = "b0761109881c788c283bf5f426229c6fcde89c04752ea43d887d3e827e1a968a"
EXCLUDED_EVIDENCE = frozenset({"IMP", "IGI", "IEP", "HMP", "HGI", "HEP"})
DATE_CUTOFF = "20221231"
ENSG_RE = re.compile(r"^ENSG[0-9]+$")
UNIPROT_RE = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
GO_RE = re.compile(r"^GO:[0-9]{7}$")


class BpFeatureError(ValueError):
    """Raised when pinned BP feature construction violates its contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_file(path: Path, size: int, digest: str) -> bytes:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise BpFeatureError(f"pinned source mismatch: {path}")
    return path.read_bytes()


def read_roster(path: Path, expected_sha256: str | None = None) -> tuple[str, ...]:
    payload = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise BpFeatureError(f"roster hash mismatch: {path}")
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise BpFeatureError("roster must be LF-terminated")
    ids = tuple(payload.decode("ascii").splitlines())
    if tuple(sorted(set(ids))) != ids or any(ENSG_RE.fullmatch(gene) is None for gene in ids):
        raise BpFeatureError("roster must contain sorted unique stable ENSG IDs")
    return ids


def parse_mapping(payload: bytes, universe: frozenset[str]) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    decompressed = gzip.decompress(payload)
    lines = decompressed.decode("ascii").splitlines()
    expected_header = (
        "gene_stable_id\ttranscript_stable_id\tprotein_stable_id\txref\tdb_name\t"
        "info_type\tsource_identity\txref_identity\tlinkage_type"
    )
    if not lines or lines[0] != expected_header:
        raise BpFeatureError("unexpected Ensembl mapping header")
    mapped: dict[str, set[str]] = {}
    total = retained = 0
    mapped_genes: set[str] = set()
    for number, line in enumerate(lines[1:], 2):
        columns = line.split("\t")
        if len(columns) != 9:
            raise BpFeatureError(f"mapping row {number} is not 9 columns")
        gene, xref, database = columns[0], columns[3], columns[4]
        if ENSG_RE.fullmatch(gene) is None or UNIPROT_RE.fullmatch(xref) is None or not database.startswith("Uniprot"):
            raise BpFeatureError(f"invalid identity in mapping row {number}")
        total += 1
        if gene in universe:
            mapped.setdefault(xref, set()).add(gene)
            mapped_genes.add(gene)
            retained += 1
    return ({key: frozenset(value) for key, value in sorted(mapped.items())}, {
        "totalRows": total,
        "retainedRows": retained,
        "mappedUniverseGenes": len(mapped_genes),
        "unmappedUniverseGenes": len(universe - mapped_genes),
        "uniqueMappedUniProtXrefs": len(mapped),
        "decompressedSha256": sha256_bytes(decompressed),
    })


def parse_bp_gaf(
    payload: bytes,
    xref_to_genes: Mapping[str, frozenset[str]],
    universe: Sequence[str],
) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    decompressed = gzip.decompress(payload)
    lines = decompressed.decode("utf-8").splitlines()
    if "!gaf-version: 2.2" not in lines:
        raise BpFeatureError("GO archive does not declare GAF 2.2")
    terms: dict[str, set[str]] = {gene: set() for gene in universe}
    raw_bp_evidence: Counter[str] = Counter()
    eligible_evidence: Counter[str] = Counter()
    mapped_evidence: Counter[str] = Counter()
    excluded_evidence: Counter[str] = Counter()
    discarded: Counter[str] = Counter()
    total = bp_rows = eligible = mapped_rows = 0
    selected_xrefs: set[str] = set()
    mapped_xrefs: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("!"):
            continue
        total += 1
        columns = line.split("\t")
        if len(columns) != 17:
            raise BpFeatureError(f"GAF row {number} is not 17 columns")
        database, xref = columns[0], columns[1]
        qualifier, term, evidence = columns[3], columns[4], columns[6]
        aspect, taxon, date = columns[8], columns[12], columns[13]
        if database != "UniProtKB" or UNIPROT_RE.fullmatch(xref) is None:
            raise BpFeatureError(f"GAF row {number} lacks exact UniProt identity")
        if GO_RE.fullmatch(term) is None or re.fullmatch(r"taxon:9606(?:\|taxon:[0-9]+)?", taxon) is None:
            raise BpFeatureError(f"GAF row {number} has invalid GO/taxon identity")
        if re.fullmatch(r"[0-9]{8}", date) is None:
            raise BpFeatureError(f"GAF row {number} has invalid date")
        if aspect != "P":
            discarded["non-biological-process-aspect"] += 1
            continue
        bp_rows += 1
        raw_bp_evidence[evidence] += 1
        if date > DATE_CUTOFF:
            discarded["after-date-cutoff"] += 1
            continue
        if "NOT" in qualifier.split("|"):
            discarded["negated-qualifier"] += 1
            continue
        if evidence in EXCLUDED_EVIDENCE:
            discarded["perturbation-derived-evidence"] += 1
            excluded_evidence[evidence] += 1
            continue
        eligible += 1
        eligible_evidence[evidence] += 1
        selected_xrefs.add(xref)
        genes = xref_to_genes.get(xref, frozenset())
        if not genes:
            discarded["no-exact-ensembl-universe-mapping"] += 1
            continue
        for gene in genes:
            terms[gene].add(term)
        mapped_rows += 1
        mapped_evidence[evidence] += 1
        mapped_xrefs.add(xref)
    return ({gene: frozenset(value) for gene, value in terms.items()}, {
        "totalAnnotationRows": total,
        "biologicalProcessRows": bp_rows,
        "eligibleRowsBeforeMapping": eligible,
        "mappedEligibleRowsBeforeDeduplication": mapped_rows,
        "rawBiologicalProcessEvidenceCounts": dict(sorted(raw_bp_evidence.items())),
        "eligibleEvidenceCountsBeforeMapping": dict(sorted(eligible_evidence.items())),
        "mappedEligibleEvidenceCounts": dict(sorted(mapped_evidence.items())),
        "excludedPerturbationEvidenceCounts": dict(sorted(excluded_evidence.items())),
        "discardedRows": dict(sorted(discarded.items())),
        "selectedUniqueUniProtXrefs": len(selected_xrefs),
        "mappedUniqueUniProtXrefs": len(mapped_xrefs),
        "decompressedSha256": sha256_bytes(decompressed),
    })


def matrices_from_fit_terms(
    terms_by_gene: Mapping[str, frozenset[str]],
    fit_ids: Sequence[str],
    output_ids: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, tuple[str, ...], dict[str, object]]:
    vocabulary = tuple(sorted({term for gene in fit_ids for term in terms_by_gene[gene]}))
    if len(vocabulary) < COMPONENTS:
        raise BpFeatureError("fitting BP vocabulary is too small for 128 components")
    term_index = {term: index for index, term in enumerate(vocabulary)}

    def matrix(ids: Sequence[str]) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        for row, gene in enumerate(ids):
            for term in sorted(terms_by_gene[gene]):
                if term in term_index:
                    rows.append(row)
                    columns.append(term_index[term])
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(ids), len(vocabulary)),
            dtype=np.float32,
        )

    fit_matrix = matrix(fit_ids)
    output_matrix = matrix(output_ids)
    all_projection_terms = {term for gene in output_ids for term in terms_by_gene[gene]}
    return fit_matrix, output_matrix, vocabulary, {
        "fitVocabularyTerms": len(vocabulary),
        "projectionTermsOutsideFitVocabulary": len(all_projection_terms - set(vocabulary)),
        "fitEntitiesWithEligibleDirectTerms": int(np.count_nonzero(np.diff(fit_matrix.indptr))),
        "fitEntitiesWithoutEligibleDirectTerms": int(len(fit_ids) - np.count_nonzero(np.diff(fit_matrix.indptr))),
        "fitBinaryAssociations": int(fit_matrix.nnz),
        "outputEntitiesWithFitVocabularyTerms": int(np.count_nonzero(np.diff(output_matrix.indptr))),
        "outputEntitiesWithoutFitVocabularyTerms": int(len(output_ids) - np.count_nonzero(np.diff(output_matrix.indptr))),
        "outputBinaryAssociationsInFitVocabulary": int(output_matrix.nnz),
    }


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            stream = io.BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, stream.getvalue(), compresslevel=9)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"))
    parser.add_argument("--source", type=Path, default=Path("data/derived/slp11-human-go/source"))
    parser.add_argument("--graph-roster", type=Path, default=Path("data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/graph-universe-entity-ids.txt"))
    parser.add_argument("--current-output-roster", type=Path, default=Path("data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/entity-ids.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if sha256_file(args.data) != DATA_SHA256:
        raise BpFeatureError("development data hash mismatch")
    graph_ids = read_roster(args.graph_roster, GRAPH_ROSTER_SHA256)
    current_output_ids = read_roster(args.current_output_roster, OUTPUT_ROSTER_SHA256)
    with np.load(args.data, allow_pickle=False) as data:
        if len(data["split_test"]) != 0:
            raise BpFeatureError("development snapshot unexpectedly contains test rows")
        fit_ids = tuple(sorted(set(data["action_ids"][data["split_train"]].astype(str))))
    if any(ENSG_RE.fullmatch(gene) is None for gene in fit_ids):
        raise BpFeatureError("fitting roster contains non-ENSG identity")
    output_ids = tuple(sorted(set(graph_ids) | set(current_output_ids)))
    universe = tuple(sorted(set(output_ids) | set(fit_ids)))
    mapping_payload = require_file(args.source / MAPPING_NAME, MAPPING_BYTES, MAPPING_SHA256)
    go_payload = require_file(args.source / GO_NAME, GO_BYTES, GO_SHA256)
    mapping, mapping_stats = parse_mapping(mapping_payload, frozenset(universe))
    if mapping_stats["decompressedSha256"] != MAPPING_DECOMPRESSED_SHA256:
        raise BpFeatureError("decompressed mapping hash mismatch")
    terms_by_gene, annotation_stats = parse_bp_gaf(go_payload, mapping, universe)
    if annotation_stats["decompressedSha256"] != GO_DECOMPRESSED_SHA256:
        raise BpFeatureError("decompressed GO hash mismatch")
    fit_matrix, output_matrix, terms, coverage = matrices_from_fit_terms(terms_by_gene, fit_ids, output_ids)
    with threadpool_limits(limits=2):
        svd = TruncatedSVD(
            n_components=COMPONENTS,
            algorithm="randomized",
            n_iter=7,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=SEED,
        )
        svd.fit(fit_matrix)
        values = svd.transform(output_matrix).astype("<f4")
    present = (np.diff(output_matrix.indptr) > 0).astype(np.uint8)
    if not np.isfinite(values).all():
        raise BpFeatureError("nonfinite BP feature generated")
    args.output.mkdir(parents=True, exist_ok=False)
    fit_payload = "".join(f"{gene}\n" for gene in fit_ids).encode("ascii")
    output_payload = "".join(f"{gene}\n" for gene in output_ids).encode("ascii")
    term_payload = "".join(f"{term}\n" for term in terms).encode("ascii")
    (args.output / "fit-entity-ids.txt").write_bytes(fit_payload)
    (args.output / "entity-ids.txt").write_bytes(output_payload)
    (args.output / "fit-term-ids.txt").write_bytes(term_payload)
    feature_path = args.output / "human-go-bp-source3-fit-svd128-features.npz"
    feature_path.write_bytes(deterministic_npz({
        "feature_values": values,
        "annotation_present": present,
        "entity_taxon": np.full(len(output_ids), TAXON, dtype="<i8"),
        "entity_id": np.asarray(output_ids, dtype="<U15"),
    }))
    components = svd.components_.astype("<f4")
    basis_path = args.output / "human-go-bp-source3-fit-svd128-basis.npz"
    basis_path.write_bytes(deterministic_npz({
        "components": components,
        "singular_values": svd.singular_values_.astype("<f4"),
        "explained_variance_ratio": svd.explained_variance_ratio_.astype("<f4"),
        "term_id": np.asarray(terms, dtype="<U10"),
    }))
    source_dir = args.output / "source"
    source_dir.mkdir()
    shutil.copyfile(Path(__file__), source_dir / Path(__file__).name)
    manifest = {
        "schema": "slp.human-go-bp-source3-fit-svd-feature-artifact/v1",
        "status": "exploratory-static-feature-candidate-not-model-fitted",
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "taxon": TAXON,
            "namespace": "Ensembl-gene",
            "rows": len(output_ids),
            "ordering": "ascending stable unversioned ENSG",
            "outputRule": "union of fixed Ensembl116 translated graph roster and current specieswide feature output roster",
            "graphRows": len(graph_ids),
            "currentOutputRows": len(current_output_ids),
        },
        "sources": {
            "goaHuman": {"release": "2022-09-19", "sha256": GO_SHA256, "path": str(args.source / GO_NAME)},
            "ensemblMapping": {"release": 108, "sha256": MAPPING_SHA256, "path": str(args.source / MAPPING_NAME)},
            "developmentIdentityOnly": {"sha256": DATA_SHA256, "path": str(args.data), "arraysRead": ["action_ids", "split_train", "split_test"]},
            "graphRoster": {"sha256": GRAPH_ROSTER_SHA256, "path": str(args.graph_roster)},
            "currentOutputRoster": {"sha256": OUTPUT_ROSTER_SHA256, "path": str(args.current_output_roster)},
            "declaration": "sources/goa-human-ensembl-2022-09-19-bp.yaml",
            "rights": "rights/goa-human-ensembl-2022-static-bp-mapping.yaml",
        },
        "filter": {
            "aspect": {"P": "biological_process"},
            "annotationDateMaximumInclusive": "2022-12-31",
            "excludedQualifier": "NOT",
            "excludedEvidence": sorted(EXCLUDED_EVIDENCE),
            "directTermsOnly": True,
            "ancestorPropagation": False,
            "identityJoin": "exact GAF UniProtKB accession to exact Ensembl108 xref to stable ENSG",
            "symbolsUsed": False,
            "freeTextUsed": False,
        },
        "fit": {
            "rows": len(fit_ids),
            "rosterSha256": sha256_bytes(fit_payload),
            "rosterSource": "unique source-three split_train intervention genes; no molecular values read",
            "components": COMPONENTS,
            "algorithm": "sklearn TruncatedSVD randomized",
            "seed": SEED,
            "iterations": 7,
            "oversamples": 10,
            "termVocabularyFitRosterOnly": True,
            "termCount": len(terms),
            "termListSha256": sha256_bytes(term_payload),
            "componentFloat32Sha256": sha256_bytes(components.tobytes("C")),
            "explainedVarianceRatioSum": float(svd.explained_variance_ratio_.sum()),
        },
        "mappingStatistics": mapping_stats,
        "annotationStatistics": annotation_stats,
        "coverage": coverage,
        "outputs": {
            "features": {"path": feature_path.name, "sha256": sha256_file(feature_path), "shape": list(values.shape)},
            "basis": {"path": basis_path.name, "sha256": sha256_file(basis_path), "shape": list(components.shape)},
            "entityRosterSha256": sha256_bytes(output_payload),
        },
        "runtime": {
            "elapsedSeconds": time.monotonic() - started,
            "cpuThreads": 2,
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "scikitLearnVersion": sklearn.__version__,
        },
        "accessBoundary": {
            "quantitativeMolecularValuesRead": False,
            "testOutcomesRead": False,
            "benchmarkLabelsRead": False,
            "staticAnnotationsOnly": True,
        },
        "limitations": [
            "Direct annotations omit ontology ancestors.",
            "Terms absent from source-three fitting genes are excluded from the fitted vocabulary.",
            "Zero vectors mean no eligible mapped direct term in the fitting-derived vocabulary, not absence of biological process.",
            "GO annotations are associative static evidence and do not establish a causal intervention mechanism.",
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "features": manifest["outputs"]["features"], "coverage": coverage, "elapsedSeconds": manifest["runtime"]["elapsedSeconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
