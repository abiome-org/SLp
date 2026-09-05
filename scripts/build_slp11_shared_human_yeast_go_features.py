"""Build one shared 2022 MF/CC GO coordinate system for human and yeast."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy
import sklearn
from build_slp11_go_features import (
    EXPECTED_GAF_BYTES as YEAST_GAF_BYTES,
)
from build_slp11_go_features import (
    EXPECTED_GAF_SHA256 as YEAST_GAF_SHA256,
)
from build_slp11_go_features import parse_gaf_bytes as parse_yeast_gaf
from build_slp11_human_go_features import (
    GO_BYTES,
    GO_NAME,
    GO_SHA256,
    MAPPING_BYTES,
    MAPPING_NAME,
    MAPPING_SHA256,
    parse_mapping_bytes,
    require_file,
)
from build_slp11_human_go_features import (
    parse_gaf_bytes as parse_human_gaf,
)
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

HUMAN_TAXON = 9606
YEAST_TAXON = 4932
SEED = 731
COMPONENTS = 256
SGD_RE = re.compile(r"S[0-9]{9}")
HUMAN_UNIVERSE_SHA256 = "f4bbfe62b73cf6362170996fcf34200cea68da106d687d3c9e994e709e951f40"
HUMAN_UNIVERSE_ROWS = 23_879
YEAST_GAF_NAME = "sgd-2022-09-19.gaf.gz"
OUTPUT_NPZ = "human-yeast-shared-go-mf-cc-svd256-features.npz"
BASIS_NPZ = "human-yeast-shared-go-mf-cc-svd256-basis.npz"
MANIFEST = "manifest.json"


class SharedGoError(ValueError):
    """Raised when the frozen shared GO contract is violated."""


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.save(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    return output.getvalue()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_yeast_gene_ids(payload: bytes) -> tuple[str, ...]:
    try:
        lines = gzip.decompress(payload).decode("utf-8").splitlines()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise SharedGoError("yeast GAF is not a complete UTF-8 gzip") from exc
    identifiers: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line or line.startswith("!"):
            continue
        columns = line.split("\t")
        if len(columns) != 17 or columns[0] != "SGD" or SGD_RE.fullmatch(columns[1]) is None:
            raise SharedGoError(f"invalid yeast identity at GAF row {number}")
        identifiers.add("SGD:" + columns[1])
    if not identifiers:
        raise SharedGoError("yeast GAF has no stable SGD identities")
    return tuple(sorted(identifiers))


def load_human_universe(path: Path) -> tuple[str, ...]:
    if sha256_file(path) != HUMAN_UNIVERSE_SHA256:
        raise SharedGoError("human translated-gene universe hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"feature_values", "entity_taxon", "entity_id"}:
            raise SharedGoError("human universe schema mismatch")
        identifiers = tuple(source["entity_id"].astype(str).tolist())
        taxa = source["entity_taxon"]
    if (
        len(identifiers) != HUMAN_UNIVERSE_ROWS
        or identifiers != tuple(sorted(set(identifiers)))
        or not np.all(taxa == HUMAN_TAXON)
    ):
        raise SharedGoError("human universe identity contract mismatch")
    return identifiers


def binary_matrix(
    identifiers: Sequence[str], terms_by_id: Mapping[str, frozenset[str]], vocabulary: Sequence[str]
) -> sparse.csr_matrix:
    term_index = {term: index for index, term in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    for row, identifier in enumerate(identifiers):
        for term in sorted(terms_by_id.get(identifier, frozenset())):
            rows.append(row)
            columns.append(term_index[term])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(identifiers), len(vocabulary)),
        dtype=np.float32,
    )


def fit_shared_svd(
    yeast: sparse.csr_matrix,
    human: sparse.csr_matrix,
    components: int = COMPONENTS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    if yeast.shape[1] != human.shape[1] or yeast.shape[1] < 1:
        raise SharedGoError("species matrices must share a nonempty term axis")
    if yeast.shape[0] < 1 or human.shape[0] < 1:
        raise SharedGoError("both species require at least one row")
    if components < 1 or components > yeast.shape[1]:
        raise SharedGoError("invalid shared SVD dimension")
    yeast_weight = np.float32(1.0 / np.sqrt(yeast.shape[0]))
    human_weight = np.float32(1.0 / np.sqrt(human.shape[0]))
    weighted = sparse.vstack(
        [yeast * yeast_weight, human * human_weight], format="csr", dtype=np.float32
    )
    model = TruncatedSVD(
        n_components=components,
        algorithm="randomized",
        n_iter=7,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=seed,
    )
    model.fit(weighted)
    component_matrix = model.components_.astype(np.dtype("<f4"), copy=False)
    yeast_values = (yeast @ component_matrix.T).astype(np.dtype("<f4"), copy=False)
    human_values = (human @ component_matrix.T).astype(np.dtype("<f4"), copy=False)
    if not np.isfinite(yeast_values).all() or not np.isfinite(human_values).all():
        raise SharedGoError("nonfinite shared GO feature generated")
    return yeast_values, human_values, model


def identical_cross_species_audit(
    yeast: sparse.csr_matrix,
    human: sparse.csr_matrix,
    yeast_values: np.ndarray,
    human_values: np.ndarray,
) -> dict[str, int | bool]:
    fingerprints: defaultdict[bytes, dict[str, list[int]]] = defaultdict(
        lambda: {"yeast": [], "human": []}
    )
    for species, matrix in (("yeast", yeast), ("human", human)):
        for row in range(matrix.shape[0]):
            left, right = matrix.indptr[row : row + 2]
            fingerprint = matrix.indices[left:right].astype(np.dtype("<i4"), copy=False).tobytes()
            fingerprints[fingerprint][species].append(row)
    shared = [group for group in fingerprints.values() if group["yeast"] and group["human"]]
    checked_pairs = 0
    for group in shared:
        reference = yeast_values[group["yeast"][0]]
        for index in group["yeast"]:
            if not np.array_equal(reference, yeast_values[index]):
                raise SharedGoError("identical yeast annotation rows produced unequal vectors")
            checked_pairs += 1
        for index in group["human"]:
            if not np.array_equal(reference, human_values[index]):
                raise SharedGoError("identical cross-species annotation rows produced unequal vectors")
            checked_pairs += 1
    return {
        "sharedExactAnnotationPatterns": len(shared),
        "rowsChecked": checked_pairs,
        "exactVectorEquality": True,
    }


def build(
    human_universe: Path,
    human_source: Path,
    yeast_gaf_path: Path,
    output_dir: Path,
) -> dict:
    started = time.monotonic()
    human_ids = load_human_universe(human_universe)
    mapping_payload = require_file(
        human_source / MAPPING_NAME, MAPPING_BYTES, MAPPING_SHA256, "Ensembl mapping"
    )
    human_gaf_payload = require_file(
        human_source / GO_NAME, GO_BYTES, GO_SHA256, "human GO GAF"
    )
    if (
        not yeast_gaf_path.is_file()
        or yeast_gaf_path.stat().st_size != YEAST_GAF_BYTES
        or sha256_file(yeast_gaf_path) != YEAST_GAF_SHA256
    ):
        raise SharedGoError("yeast GO GAF hash mismatch")
    yeast_gaf_payload = yeast_gaf_path.read_bytes()

    yeast_ids = raw_yeast_gene_ids(yeast_gaf_payload)
    yeast_annotations, yeast_stats = parse_yeast_gaf(yeast_gaf_payload)
    xrefs, mapping_stats = parse_mapping_bytes(mapping_payload, frozenset(human_ids))
    human_rows, human_stats = parse_human_gaf(human_gaf_payload, xrefs, human_ids)
    human_annotations = dict(zip(human_ids, human_rows, strict=True))
    vocabulary = tuple(
        sorted(
            {term for terms in yeast_annotations.values() for term in terms}
            | {term for terms in human_rows for term in terms}
        )
    )
    if len(vocabulary) < COMPONENTS:
        raise SharedGoError("common GO vocabulary is too small for 256 components")
    yeast_matrix = binary_matrix(yeast_ids, yeast_annotations, vocabulary)
    human_matrix = binary_matrix(human_ids, human_annotations, vocabulary)
    yeast_values, human_values, model = fit_shared_svd(yeast_matrix, human_matrix)
    equality = identical_cross_species_audit(
        yeast_matrix, human_matrix, yeast_values, human_values
    )

    entity_ids = np.asarray(yeast_ids + human_ids, dtype="<U15")
    entity_taxon = np.concatenate(
        [
            np.full(len(yeast_ids), YEAST_TAXON, dtype=np.dtype("<i8")),
            np.full(len(human_ids), HUMAN_TAXON, dtype=np.dtype("<i8")),
        ]
    )
    values = np.vstack([yeast_values, human_values]).astype(np.dtype("<f4"), copy=False)
    present = np.concatenate(
        [yeast_matrix.getnnz(axis=1) > 0, human_matrix.getnnz(axis=1) > 0]
    ).astype(np.bool_)
    keys = list(zip(entity_taxon.tolist(), entity_ids.tolist(), strict=True))
    if keys != sorted(set(keys)):
        raise SharedGoError("composite output keys are not uniquely sorted")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / OUTPUT_NPZ
    basis_path = output_dir / BASIS_NPZ
    manifest_path = output_dir / MANIFEST
    if feature_path.exists() or basis_path.exists() or manifest_path.exists():
        raise SharedGoError("refusing to overwrite shared GO artifact")
    feature_payload = deterministic_npz_bytes(
        {
            "feature_values": values,
            "entity_taxon": entity_taxon,
            "entity_id": entity_ids,
            "direct_annotation_present": present,
        }
    )
    component_matrix = model.components_.astype(np.dtype("<f4"), copy=False)
    basis_payload = deterministic_npz_bytes(
        {"components": component_matrix, "term_id": np.asarray(vocabulary, dtype="<U10")}
    )
    feature_path.write_bytes(feature_payload)
    basis_path.write_bytes(basis_payload)
    manifest = {
        "schema": "slp.shared-human-yeast-go-mf-cc-svd/v1",
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending taxon then codepoint entity ID",
            "rows": len(keys),
            "human": {
                "taxon": HUMAN_TAXON,
                "namespace": "Ensembl-gene",
                "rows": len(human_ids),
                "universe": "all Ensembl-116 stable genes with a selected translation",
            },
            "yeast": {
                "taxon": YEAST_TAXON,
                "namespace": "SGD",
                "rows": len(yeast_ids),
                "universe": "every exact SGD DB object ID present in the pinned 2022-09-19 GAF",
            },
        },
        "featureDefinition": {
            "dimensions": COMPONENTS,
            "directTermsOnly": True,
            "aspects": ["molecular_function", "cellular_component"],
            "ancestorPropagation": False,
            "excludedQualifier": "NOT",
            "excludedEvidence": ["HEP", "HGI", "HMP", "IEP", "IGI", "IMP"],
            "dateMaximumInclusive": "2022-12-31",
            "termVocabularySize": len(vocabulary),
            "termVocabularySha256": hashlib.sha256(
                ("\n".join(vocabulary) + "\n").encode()
            ).hexdigest(),
            "projection": "unweighted binary direct-term row @ shared components.T",
        },
        "compression": {
            "method": "sklearn TruncatedSVD randomized",
            "components": COMPONENTS,
            "seed": SEED,
            "iterations": 7,
            "oversamples": 10,
            "fitMatrix": "pooled species rows with row weights 1/sqrt(species row count)",
            "humanTotalSquaredRowWeight": 1.0,
            "yeastTotalSquaredRowWeight": 1.0,
            "componentsFloat32Sha256": hashlib.sha256(component_matrix.tobytes("C")).hexdigest(),
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "scikitLearnVersion": sklearn.__version__,
        },
        "coverage": {
            "human": {
                "rows": len(human_ids),
                "withEligibleDirectTerms": int(np.count_nonzero(human_matrix.getnnz(axis=1))),
                "binaryAssociations": int(human_matrix.nnz),
            },
            "yeast": {
                "rows": len(yeast_ids),
                "withEligibleDirectTerms": int(np.count_nonzero(yeast_matrix.getnnz(axis=1))),
                "binaryAssociations": int(yeast_matrix.nnz),
            },
            "identicalRowInvariant": equality,
        },
        "sources": {
            "humanGafSha256": GO_SHA256,
            "humanEnsemblMappingSha256": MAPPING_SHA256,
            "yeastGafSha256": YEAST_GAF_SHA256,
            "humanUniverseSha256": HUMAN_UNIVERSE_SHA256,
            "humanRights": "rights/goa-human-ensembl-2022-static-mapping.yaml",
            "yeastRights": "rights/go-sgd-2022-09-19-cc-by-4.0.yaml",
        },
        "sourceStatistics": {
            "humanMapping": mapping_stats,
            "humanAnnotations": human_stats,
            "yeastAnnotations": yeast_stats,
        },
        "artifacts": {
            "features": {
                "path": OUTPUT_NPZ,
                "sha256": hashlib.sha256(feature_payload).hexdigest(),
                "shape": list(values.shape),
            },
            "basis": {
                "path": BASIS_NPZ,
                "sha256": hashlib.sha256(basis_payload).hexdigest(),
                "shape": list(component_matrix.shape),
            },
        },
        "accessBoundary": {
            "staticAnnotationsOnly": True,
            "quantitativeOutcomesRead": False,
            "splitAssignmentsRead": False,
            "benchmarkDataRead": False,
        },
        "runtimeSeconds": time.monotonic() - started,
        "limitations": [
            "The basis is transductively fit to static annotations across both species, with no molecular outcomes.",
            "Direct annotations omit ancestors and may understate shared biology.",
            "Zero rows mean no eligible mapped direct annotation, not absence of biological function.",
            "The yeast universe is GAF-defined and does not claim to enumerate every current SGD gene.",
            "This feature artifact is exploratory and is not an admitted OMF DatasetSnapshot.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-universe", required=True, type=Path)
    parser.add_argument("--human-source", required=True, type=Path)
    parser.add_argument("--yeast-gaf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.human_universe, args.human_source, args.yeast_gaf, args.output_dir),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
