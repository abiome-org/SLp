#!/usr/bin/env python3
"""Build deterministic, static GO features for the SLp-1.1 yeast entity set.

Only direct Molecular Function and Cellular Component annotations from the
immutable 2022-09-19 GO Consortium SGD release are eligible.  The parser drops
negated annotations and perturbation-derived evidence before constructing the
term vocabulary.  Entity projection uses only exact ``sourceSequenceIds`` from
the frozen sequence-statistics provenance, including typed SGD-to-UniProt
relations.  No quantitative outcome, split, interaction, phenotype, or
synthetic-lethality artifact is accepted.
"""

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
from build_slp11_sequence_features import (
    EXPECTED_ENTITY_KEY_SET_SHA256,
    EXPECTED_ENTITY_ROWS_SHA256,
    EXPECTED_FEATURE_BLOCK_SHA256,
    EXPECTED_SEQUENCE_PROVENANCE_SHA256,
    SPECIES_TAXON,
    STATIC_UNIVERSE_RESOURCE,
    deterministic_npz_bytes,
    load_pinned_feature_provenance,
)
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

SOURCE_URL = "https://release.geneontology.org/2022-09-19/annotations/sgd.gaf.gz"
SOURCE_RELEASE = "2022-09-19"
EXPECTED_GAF_BYTES = 3_742_380
EXPECTED_GAF_SHA256 = "cb991189e29e847e1a0437baf5d786a8cff0cab2bc7d58f97251f5e5df11a535"
EXPECTED_GAF_DECOMPRESSED_BYTES = 26_993_070
EXPECTED_GAF_DECOMPRESSED_SHA256 = (
    "fc526528dd492c76be75121506e56b9b2b1edc904712a9028f3a94af77bfdea5"
)
ANNOTATION_DATE_CUTOFF = "20221231"
ALLOWED_ASPECTS = {"F": "molecular_function", "C": "cellular_component"}
EXCLUDED_EVIDENCE = frozenset({"IMP", "IGI", "IEP", "HMP", "HGI", "HEP"})
SGD_ID = re.compile(r"S[0-9]{9}")
GO_ID = re.compile(r"GO:[0-9]{7}")
OUTPUT_SCHEMA = "slp.go-direct-svd-feature-artifact/v1"
NPZ_NAME = "go-direct-svd-features.npz"
MANIFEST_NAME = "go-direct-svd-features.manifest.json"
DEFAULT_COMPONENTS = 256
DEFAULT_SEED = 731


class GoFeatureError(ValueError):
    """Raised when the static GO feature contract cannot be reproduced."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def parse_gaf_bytes(payload: bytes) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    """Parse and filter gzip-compressed GAF 2.2 bytes before vocabulary creation."""

    try:
        decompressed = gzip.decompress(payload)
        text = decompressed.decode("utf-8")
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as error:
        raise GoFeatureError("GO source must be a complete UTF-8 gzip GAF") from error
    lines = text.splitlines()
    headers = [line for line in lines if line.startswith("!")]
    if "!gaf-version: 2.2" not in headers:
        raise GoFeatureError("GO source must declare GAF version 2.2")

    annotations: dict[str, set[str]] = {}
    discarded: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    aspect_counts: Counter[str] = Counter()
    total_rows = 0
    selected_rows = 0
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("!"):
            continue
        total_rows += 1
        columns = line.split("\t")
        if len(columns) != 17:
            raise GoFeatureError(f"GAF row {line_number} does not have exactly 17 columns")
        database, raw_id = columns[0], columns[1]
        qualifier, term, evidence = columns[3], columns[4], columns[6]
        aspect, taxon, annotation_date = columns[8], columns[12], columns[13]
        if database != "SGD" or SGD_ID.fullmatch(raw_id) is None:
            raise GoFeatureError(f"GAF row {line_number} lacks a stable SGD identifier")
        if GO_ID.fullmatch(term) is None:
            raise GoFeatureError(f"GAF row {line_number} has an invalid GO term")
        if taxon != "taxon:559292":
            raise GoFeatureError(f"GAF row {line_number} has unexpected organism taxon")
        if re.fullmatch(r"[0-9]{8}", annotation_date) is None:
            raise GoFeatureError(f"GAF row {line_number} has an invalid annotation date")
        if aspect not in ALLOWED_ASPECTS:
            discarded["biological-process-aspect"] += 1
            continue
        if annotation_date > ANNOTATION_DATE_CUTOFF:
            discarded["after-date-cutoff"] += 1
            continue
        if "NOT" in qualifier.split("|"):
            discarded["negated-qualifier"] += 1
            continue
        if evidence in EXCLUDED_EVIDENCE:
            discarded["perturbation-derived-evidence"] += 1
            continue
        stable_id = "SGD:" + raw_id
        annotations.setdefault(stable_id, set()).add(term)
        selected_rows += 1
        evidence_counts[evidence] += 1
        aspect_counts[ALLOWED_ASPECTS[aspect]] += 1

    if not annotations:
        raise GoFeatureError("GO filters selected no stable SGD annotations")
    frozen = {key: frozenset(values) for key, values in sorted(annotations.items())}
    date_headers = sorted(
        line.removeprefix("!date-generated: ")
        for line in headers
        if line.startswith("!date-generated: ")
    )
    statistics: dict[str, object] = {
        "gafVersion": "2.2",
        "dateGeneratedHeaders": date_headers,
        "decompressedBytes": len(decompressed),
        "decompressedSha256": sha256_bytes(decompressed),
        "totalAnnotationRows": total_rows,
        "selectedRowsBeforeDeduplication": selected_rows,
        "selectedDirectAssociations": sum(len(terms) for terms in frozen.values()),
        "selectedSgdGenes": len(frozen),
        "selectedEvidenceCounts": dict(sorted(evidence_counts.items())),
        "selectedAspectCounts": dict(sorted(aspect_counts.items())),
        "discardedRows": dict(sorted(discarded.items())),
    }
    return frozen, statistics


def project_direct_terms(
    annotations: Mapping[str, frozenset[str]],
    entities: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
) -> tuple[sparse.csr_matrix, tuple[str, ...], dict[str, int]]:
    """Project SGD direct terms through exact frozen sequence-source relations."""

    if len(entities) == 0 or len(entities) != len(provenance):
        raise GoFeatureError("entity and sequence-provenance rows must be non-empty and aligned")
    entity_terms: list[set[str]] = []
    covered_source_ids: set[str] = set()
    keys: list[tuple[int, str]] = []
    for index, (entity, source) in enumerate(zip(entities, provenance, strict=True)):
        taxon, entity_id = entity.get("ncbiTaxon"), entity.get("entityId")
        if (
            entity.get("rowIndex") != index
            or taxon != SPECIES_TAXON
            or not isinstance(entity_id, str)
            or source.get("rowIndex") != index
            or source.get("ncbiTaxon") != taxon
            or source.get("entityId") != entity_id
        ):
            raise GoFeatureError(f"entity/provenance identity mismatch at row {index}")
        source_ids = source.get("sourceSequenceIds")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or source_ids != sorted(set(source_ids))
            or any(
                not isinstance(item, str)
                or not item.startswith("SGD:")
                or SGD_ID.fullmatch(item[4:]) is None
                for item in source_ids
            )
        ):
            raise GoFeatureError(f"invalid exact sourceSequenceIds at row {index}")
        direct: set[str] = set()
        for source_id in source_ids:
            terms = annotations.get(source_id, frozenset())
            if terms:
                covered_source_ids.add(source_id)
                direct.update(terms)
        keys.append((taxon, entity_id))
        entity_terms.append(direct)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise GoFeatureError("composite entity keys must be uniquely sorted")

    terms = tuple(sorted({term for row in entity_terms for term in row}))
    if not terms:
        raise GoFeatureError("no selected GO term maps to the frozen static entity set")
    term_index = {term: index for index, term in enumerate(terms)}
    rows: list[int] = []
    columns: list[int] = []
    for row, direct in enumerate(entity_terms):
        for term in sorted(direct):
            rows.append(row)
            columns.append(term_index[term])
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(entities), len(terms)), dtype=np.float32
    )
    coverage = {
        "staticEntities": len(entities),
        "entitiesWithDirectTerms": sum(bool(direct) for direct in entity_terms),
        "zeroCoverageEntities": sum(not direct for direct in entity_terms),
        "coveredSgdSourceIds": len(covered_source_ids),
        "directTermsMapped": len(terms),
        "binaryAssociations": int(matrix.nnz),
    }
    return matrix, terms, coverage


def fit_truncated_svd(
    matrix: sparse.csr_matrix,
    max_components: int = DEFAULT_COMPONENTS,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Fit deterministic randomized TruncatedSVD over static entity rows only."""

    if type(max_components) is not int or max_components < 1:
        raise GoFeatureError("max_components must be a positive integer")
    if type(seed) is not int:
        raise GoFeatureError("seed must be an integer")
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise GoFeatureError("direct-term matrix must be non-empty and two-dimensional")
    components = min(max_components, matrix.shape[1])
    model = TruncatedSVD(
        n_components=components,
        algorithm="randomized",
        n_iter=7,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=seed,
    )
    values = model.fit_transform(matrix).astype(np.dtype("<f4"), copy=False)
    if values.shape != (matrix.shape[0], components) or not np.isfinite(values).all():
        raise GoFeatureError("TruncatedSVD generated invalid feature values")
    return values, model


def build_artifact(
    gaf_path: Path,
    feature_block_path: Path,
    output_dir: Path,
    *,
    max_components: int = DEFAULT_COMPONENTS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Build the feature NPZ and complete exploratory provenance manifest."""

    payload = gaf_path.read_bytes()
    if len(payload) != EXPECTED_GAF_BYTES or sha256_bytes(payload) != EXPECTED_GAF_SHA256:
        raise GoFeatureError("GO GAF gzip does not match the frozen 2022-09-19 source")
    annotations, annotation_statistics = parse_gaf_bytes(payload)
    if (
        annotation_statistics["decompressedBytes"] != EXPECTED_GAF_DECOMPRESSED_BYTES
        or annotation_statistics["decompressedSha256"] != EXPECTED_GAF_DECOMPRESSED_SHA256
    ):
        raise GoFeatureError("decompressed GO GAF content hash mismatch")
    entities, provenance, source_manifest = load_pinned_feature_provenance(
        feature_block_path
    )
    matrix, terms, coverage = project_direct_terms(annotations, entities, provenance)
    values, svd = fit_truncated_svd(matrix, max_components=max_components, seed=seed)

    max_id_chars = max(len(str(entity["entityId"])) for entity in entities)
    arrays = {
        "feature_values": values,
        "entity_taxon": np.asarray(
            [int(entity["ncbiTaxon"]) for entity in entities], dtype=np.dtype("<i8")
        ),
        "entity_id": np.asarray(
            [str(entity["entityId"]) for entity in entities], dtype=f"<U{max_id_chars}"
        ),
    }
    npz_payload = deterministic_npz_bytes(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / NPZ_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if npz_path.exists() or manifest_path.exists():
        raise GoFeatureError(f"refusing to overwrite existing output in {output_dir}")
    npz_path.write_bytes(npz_payload)

    component_bytes = svd.components_.astype(np.dtype("<f4"), copy=False).tobytes(order="C")
    manifest: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "entityKeySetSha256": EXPECTED_ENTITY_KEY_SET_SHA256,
            "rows": len(entities),
            "speciesTaxon": SPECIES_TAXON,
        },
        "arrays": {
            "feature_values": {"dtype": "little-endian-float32", "shape": list(values.shape)},
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
        "source": {
            "publisher": "Gene Ontology Consortium; annotations supplied by SGD",
            "release": SOURCE_RELEASE,
            "exactDownloadUrl": SOURCE_URL,
            "retrievedAt": "2026-09-04",
            "lastModified": "2022-09-21T23:32:07Z",
            "gzipBytes": EXPECTED_GAF_BYTES,
            "gzipSha256": EXPECTED_GAF_SHA256,
            "decompressedBytes": EXPECTED_GAF_DECOMPRESSED_BYTES,
            "decompressedSha256": EXPECTED_GAF_DECOMPRESSED_SHA256,
            "rightsDeclaration": "rights/go-sgd-2022-09-19-cc-by-4.0.yaml",
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
            "geneticInteractionAnnotationsUsed": False,
            "syntheticLethalityAnnotationsUsed": False,
        },
        "annotationStatistics": annotation_statistics,
        "coverage": coverage,
        "compression": {
            "method": "sklearn.decomposition.TruncatedSVD",
            "algorithm": "randomized",
            "seed": seed,
            "iterations": 7,
            "oversamples": 10,
            "powerIterationNormalizer": "auto",
            "requestedMaximumComponents": max_components,
            "components": int(values.shape[1]),
            "fitRows": len(entities),
            "fitPopulation": "all frozen static entities only",
            "componentFloat32Sha256": sha256_bytes(component_bytes),
            "explainedVarianceRatioSum": float(svd.explained_variance_ratio_.sum()),
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "scikitLearnVersion": sklearn.__version__,
        },
        "inputs": {
            "sequenceStatisticsFeatureBlock": {
                "archiveSha256": EXPECTED_FEATURE_BLOCK_SHA256,
                "schema": source_manifest["schema"],
                "entityRowsSha256": EXPECTED_ENTITY_ROWS_SHA256,
                "sequenceProvenanceSha256": EXPECTED_SEQUENCE_PROVENANCE_SHA256,
                "staticEntityUniverseResource": STATIC_UNIVERSE_RESOURCE,
                "relationField": "sourceSequenceIds",
            }
        },
        "accessBoundary": {
            "staticAnnotationsConsumed": True,
            "quantitativeOutcomesConsumed": False,
            "heldRosterConsumed": False,
            "partitionAssignmentsConsumed": False,
            "benchmarkDataConsumed": False,
            "geneticInteractionDataConsumed": False,
            "syntheticLethalityDataConsumed": False,
            "fittedOnSlpOutcomes": False,
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
            "Zero vectors mean no eligible direct annotation mapped through the frozen sequence relation; they do not imply absence of biological function.",
            "The randomized SVD basis is transductively fit over static entity annotations only and consumes no quantitative outcomes.",
            "This artifact and its source have not been admitted as OMF DatasetSnapshot resources.",
        ],
    }
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaf", type=Path, required=True)
    parser.add_argument("--feature-block", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_artifact(
            args.gaf,
            args.feature_block,
            args.output_dir,
            max_components=args.components,
            seed=args.seed,
        )
    except (OSError, GoFeatureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
