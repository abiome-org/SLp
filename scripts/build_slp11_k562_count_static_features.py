#!/usr/bin/env python3
"""Build static features for the raw K562-essential count experiment.

Only stable identifier rosters and frozen static resources are read.  No count
matrix or quantitative perturbation outcome is accessed by this builder.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TAXON = 9606
ENTITY = re.compile(r"ENSG[0-9]+")
ROUTING_SHA256 = "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad"
STATIC_SHA256 = "20313e37d70d52253fa7b4b9b569b0fd504686a35be46b0607db1ab1c7484e54"
STATIC_MANIFEST_SHA256 = "857e34d73c45f94ef078cc1e2e271f91ec223d51e217f87d5411869457b3fe3c"
BASIS_SHA256 = "718764f4ebb6ab9ac31dba65d7d6453525e04a98b999aa7dcfeb4c3a1ab62abd"
MAPPING_SHA256 = "0d6fe982ce7023b2901171fd0e1419a2e9fbd7fbb9b3473a82ac6dee454f6e56"
GAF_SHA256 = "8b97980a895cb74255615f7cdbdd818f72a3999867b7d2a14f867874480693e1"
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
QUERY_ROSTER_SHA256 = "9182efe0304204a30418c55d364de3178557d6b0813748436d9fa81b54da4d79"
OUTPUT_NAME = "k562-essential-count-static577.npz"
ROSTER_NAME = "roster-index.npz"


class CountStaticError(ValueError):
    """Raised when a frozen static-feature contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise CountStaticError(f"SHA-256 mismatch for {path}: {actual}")


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def exact_mappings(
    payload: bytes, requested: frozenset[str]
) -> tuple[dict[str, frozenset[str]], set[str]]:
    """Return exact UniProt-to-Ensembl mappings for requested stable genes."""
    try:
        lines = gzip.decompress(payload).decode("ascii").splitlines()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise CountStaticError("invalid pinned Ensembl mapping gzip") from exc
    expected_header = (
        "gene_stable_id\ttranscript_stable_id\tprotein_stable_id\txref\tdb_name\t"
        "info_type\tsource_identity\txref_identity\tlinkage_type"
    )
    if not lines or lines[0] != expected_header:
        raise CountStaticError("unexpected Ensembl mapping header")
    result: dict[str, set[str]] = {}
    mapped: set[str] = set()
    for line in lines[1:]:
        columns = line.split("\t")
        if len(columns) != 9:
            raise CountStaticError("unexpected Ensembl mapping row width")
        if columns[0] in requested and columns[3] and columns[4].startswith("Uniprot"):
            result.setdefault(columns[3], set()).add(columns[0])
            mapped.add(columns[0])
    return ({key: frozenset(value) for key, value in sorted(result.items())}, mapped)


def project_missing_go(
    identifiers: list[str],
    xref_to_genes: dict[str, frozenset[str]],
    gaf_payload: bytes,
    basis: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], set[str], dict[str, int]]:
    """Project eligible archived direct MF/CC terms through a frozen basis."""
    if not identifiers:
        return {}, set(), {"rowsWithEligibleTerms": 0, "retainedAssociations": 0}
    components = basis.get("components")
    terms = basis.get("term_id")
    if (
        components is None
        or terms is None
        or components.shape != (256, 6876)
        or components.dtype != np.float32
        or terms.shape != (6876,)
    ):
        raise CountStaticError("shared GO basis contract mismatch")
    try:
        lines = gzip.decompress(gaf_payload).decode("utf-8").splitlines()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise CountStaticError("invalid pinned GO GAF gzip") from exc
    if "!gaf-version: 2.2" not in lines:
        raise CountStaticError("GO source must declare GAF 2.2")
    by_gene: dict[str, set[str]] = {item: set() for item in identifiers}
    excluded_evidence = {"IMP", "IGI", "IEP", "HMP", "HGI", "HEP"}
    for line in lines:
        if not line or line.startswith("!"):
            continue
        columns = line.split("\t")
        if len(columns) != 17:
            raise CountStaticError("unexpected GO GAF row width")
        if columns[0] != "UniProtKB" or columns[8] not in {"F", "C"}:
            continue
        if columns[13] > "20221231" or "NOT" in columns[3].split("|"):
            continue
        if columns[6] in excluded_evidence:
            continue
        for gene in xref_to_genes.get(columns[1], ()):  # exact stable-ID bridge
            if gene in by_gene:
                by_gene[gene].add(columns[4])
    vocabulary = {str(term): index for index, term in enumerate(terms)}
    projected: dict[str, np.ndarray] = {}
    direct: set[str] = set()
    associations = 0
    for gene in identifiers:
        retained = sorted(term for term in by_gene[gene] if term in vocabulary)
        vector = np.zeros(256, dtype=np.float32)
        if retained:
            columns = np.asarray([vocabulary[term] for term in retained], dtype=np.int64)
            vector = components[:, columns].sum(axis=1, dtype=np.float32)
            direct.add(gene)
            associations += len(retained)
        projected[gene] = vector
    return projected, direct, {
        "rowsWithEligibleTerms": len(direct),
        "retainedAssociations": associations,
    }


def normalization(
    values: np.ndarray, fitting_indices: np.ndarray, floor: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one affine feature transform on unique fitting action rows."""
    if values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all():
        raise CountStaticError("features must be finite float32 [E,F]")
    indices = np.asarray(fitting_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())) or not len(indices):
        raise CountStaticError("fitting indices must be nonempty and unique")
    if indices.min() < 0 or indices.max() >= len(values):
        raise CountStaticError("fitting index out of range")
    selected = values[indices].astype(np.float64)
    mean64 = selected.mean(axis=0)
    sd64 = selected.std(axis=0, ddof=0)
    scale64 = np.where(sd64 <= floor, 1.0, sd64)
    normalized = ((values.astype(np.float64) - mean64) / scale64).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise CountStaticError("normalization produced nonfinite features")
    return (
        normalized,
        mean64.astype(np.float32),
        sd64.astype(np.float32),
        scale64.astype(np.float32),
    )


def duplicate_audit(values: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, object]:
    groups: dict[bytes, list[int]] = {}
    contiguous = np.ascontiguousarray(values)
    for index, row in enumerate(contiguous):
        groups.setdefault(row.tobytes(), []).append(index)
    duplicate = [indices for indices in groups.values() if len(indices) > 1]
    all_zero = np.all(values == np.float32(0.0), axis=1)
    return {
        "distinctFeatureRows": len(groups),
        "duplicateEquivalenceGroups": len(duplicate),
        "rowsBeyondFirstInDuplicateGroups": sum(len(group) - 1 for group in duplicate),
        "largestEquivalenceGroup": max(map(len, groups.values()), default=0),
        "allZeroRows": int(all_zero.sum()),
        "allZeroByRole": {
            label: int((all_zero & np.asarray(mask, dtype=np.bool_)).sum())
            for label, mask in masks.items()
        },
    }


def assemble(
    routing: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    exact_mapping_ids: set[str],
    missing_go: dict[str, np.ndarray],
    missing_go_direct: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    query_ids = routing["query_ids"].astype(str)
    cell_actions = routing["action_ids"].astype(str)
    cell_roles = routing["intervention_role"].astype(str)
    if (
        len(query_ids) != 8563
        or len(set(query_ids.tolist())) != len(query_ids)
        or any(ENTITY.fullmatch(item) is None for item in query_ids)
    ):
        raise CountStaticError("query roster is not 8,563 unique stable ENSGs")
    if not np.all(routing["query_taxon"] == TAXON) or not routing[
        "query_in_matrix"
    ].all():
        raise CountStaticError("query taxonomy/source-observation contract mismatch")
    actions = np.asarray(sorted(set(cell_actions.tolist()) - {""}))
    if len(actions) != 2057 or any(ENTITY.fullmatch(item) is None for item in actions):
        raise CountStaticError("action roster is not 2,057 unique stable ENSGs")

    role_by_action: dict[str, str] = {}
    allowed_roles = {"train", "validation", "test-excluded"}
    for action, role in zip(cell_actions, cell_roles, strict=True):
        if not action:
            continue
        if role not in allowed_roles:
            raise CountStaticError("target action has invalid intervention role")
        previous = role_by_action.setdefault(action, role)
        if previous != role:
            raise CountStaticError("one action occurs in multiple intervention roles")
    role_counts = {role: sum(value == role for value in role_by_action.values()) for role in sorted(allowed_roles)}
    if role_counts != {"test-excluded": 309, "train": 1443, "validation": 305}:
        raise CountStaticError(f"unexpected action routing counts: {role_counts}")

    source_ids = source["entity_id"].astype(str)
    source_taxon = source["entity_taxon"]
    source_values = source["feature_values"]
    if (
        source_values.shape != (24031, 577)
        or source_values.dtype != np.float32
        or source_taxon.shape != source_ids.shape
        or not np.all(source_taxon == TAXON)
        or list(source_ids) != sorted(set(source_ids.tolist()))
        or not np.isfinite(source_values).all()
    ):
        raise CountStaticError("frozen human static pack contract mismatch")
    source_index = {entity_id: index for index, entity_id in enumerate(source_ids)}
    ids = np.asarray(sorted(set(query_ids.tolist()) | set(actions.tolist())))
    index = {entity_id: row for row, entity_id in enumerate(ids)}
    values = np.zeros((len(ids), 577), dtype=np.float32)
    source_present = np.zeros(len(ids), dtype=np.bool_)
    esm_present = np.zeros(len(ids), dtype=np.bool_)
    translated = np.zeros(len(ids), dtype=np.bool_)
    go_direct = np.zeros(len(ids), dtype=np.bool_)
    go_mapping = np.isin(ids, np.asarray(sorted(exact_mapping_ids)))
    overlap = []
    for row, entity_id in enumerate(ids):
        old = source_index.get(entity_id)
        if old is None:
            values[row, 321:] = missing_go[entity_id]
            go_direct[row] = entity_id in missing_go_direct
            continue
        values[row] = source_values[old]
        source_present[row] = True
        esm_present[row] = bool(source["esm_present"][old])
        translated[row] = bool(source["is_ensembl116_translated_gene"][old])
        go_direct[row] = bool(source["go_direct_annotation_present"][old])
        overlap.append((row, old))

    query_flag = np.isin(ids, query_ids)
    action_flag = np.isin(ids, actions)
    fitting_ids = np.asarray(sorted(key for key, value in role_by_action.items() if value == "train"))
    validation_ids = np.asarray(sorted(key for key, value in role_by_action.items() if value == "validation"))
    test_ids = np.asarray(sorted(key for key, value in role_by_action.items() if value == "test-excluded"))
    fitting_indices = np.asarray([index[item] for item in fitting_ids], dtype=np.int64)
    normalized, mean, sd, scale = normalization(values, fitting_indices)
    arrays = {
        "feature_values": values,
        "normalized_feature_values": normalized,
        "feature_mean": mean,
        "feature_sd": sd,
        "feature_scale": scale,
        "entity_taxon": np.full(len(ids), TAXON, dtype=np.int64),
        "entity_id": ids,
        "source_static_row_present": source_present,
        "esm_present": esm_present,
        "go_direct_annotation_present": go_direct,
        "go_exact_uniprot_mapping_present": go_mapping,
        "is_ensembl116_translated_gene": translated,
        "is_query": query_flag,
        "is_action": action_flag,
        "is_fitting_action": np.isin(ids, fitting_ids),
        "is_validation_action": np.isin(ids, validation_ids),
        "is_test_excluded_action": np.isin(ids, test_ids),
    }
    roster = {
        "query_ids": query_ids,
        "query_entity_index": np.asarray([index[item] for item in query_ids], dtype=np.int64),
        "action_ids": actions,
        "action_entity_index": np.asarray([index[item] for item in actions], dtype=np.int64),
        "action_role": np.asarray([role_by_action[item] for item in actions]),
        "fitting_action_ids": fitting_ids,
        "fitting_action_entity_index": fitting_indices,
    }
    masks = {
        "query": query_flag,
        "action": action_flag,
        "fittingAction": arrays["is_fitting_action"],
        "validationAction": arrays["is_validation_action"],
        "testExcludedAction": arrays["is_test_excluded_action"],
    }
    audit = {
        "entities": len(ids),
        "queries": len(query_ids),
        "actions": len(actions),
        "sourceRowsCopied": int(source_present.sum()),
        "sourceRowsMissing": int((~source_present).sum()),
        "sourceOverlapBitExact": all(
            np.array_equal(values[row], source_values[old]) for row, old in overlap
        ),
        "esmPresent": int(esm_present.sum()),
        "goDirectAnnotationPresent": int(go_direct.sum()),
        "goExactUniprotMappingPresent": int(go_mapping.sum()),
        "translated": int(translated.sum()),
        "roleCounts": role_counts,
        "constantOrNearConstantColumns": int((sd <= np.float32(1e-5)).sum()),
        "duplicates": duplicate_audit(values, masks),
        "coverageByRole": {
            label: {
                "rows": int(mask.sum()),
                "esmPresent": int((mask & esm_present).sum()),
                "goDirectAnnotationPresent": int((mask & go_direct).sum()),
                "goExactUniprotMappingPresent": int((mask & go_mapping).sum()),
                "sourceStaticRowPresent": int((mask & source_present).sum()),
            }
            for label, mask in masks.items()
        },
    }
    return arrays, roster, audit


def build(args: argparse.Namespace) -> dict[str, object]:
    pins = {
        "routing": (args.routing, ROUTING_SHA256),
        "static": (args.static, STATIC_SHA256),
        "staticManifest": (args.static_manifest, STATIC_MANIFEST_SHA256),
        "sharedGoBasis": (args.shared_go_basis, BASIS_SHA256),
        "ensemblMapping": (args.mapping, MAPPING_SHA256),
        "humanGoGaf": (args.gaf, GAF_SHA256),
    }
    for path, digest in pins.values():
        require_hash(path, digest)
    with np.load(args.routing, allow_pickle=False) as archive:
        routing = {name: archive[name].copy() for name in archive.files}
    if str(routing["schema"].item()) != "slp.replogle-k562-essential-cell-routing/v1":
        raise CountStaticError("routing schema mismatch")
    if str(routing["source_sha256"].item()) != SOURCE_SHA256:
        raise CountStaticError("routing source identity mismatch")
    query_payload = "".join(f"{item}\n" for item in routing["query_ids"].astype(str)).encode("ascii")
    if hashlib.sha256(query_payload).hexdigest() != QUERY_ROSTER_SHA256:
        raise CountStaticError("ordered query roster drift")
    with np.load(args.static, allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    all_ids = set(routing["query_ids"].astype(str).tolist()) | set(
        routing["action_ids"].astype(str).tolist()
    )
    all_ids.discard("")
    source_ids = set(source["entity_id"].astype(str).tolist())
    missing = frozenset(all_ids - source_ids)
    xref_to_genes, exact_mapping_ids = exact_mappings(
        args.mapping.read_bytes(), frozenset(all_ids)
    )
    missing_xrefs = {
        xref: frozenset(genes & missing)
        for xref, genes in xref_to_genes.items()
        if genes & missing
    }
    with np.load(args.shared_go_basis, allow_pickle=False) as archive:
        basis = {name: archive[name].copy() for name in archive.files}
    missing_go, missing_go_direct, projection_audit = project_missing_go(
        sorted(missing), missing_xrefs, args.gaf.read_bytes(), basis
    )
    arrays, roster, audit = assemble(
        routing,
        source,
        exact_mapping_ids,
        missing_go,
        missing_go_direct,
    )
    if audit["sourceRowsMissing"] != 215 or not audit["sourceOverlapBitExact"]:
        raise CountStaticError(f"unexpected source overlap audit: {audit}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = args.output_dir / OUTPUT_NAME
    roster_path = args.output_dir / ROSTER_NAME
    feature_path.write_bytes(deterministic_npz(arrays))
    roster_path.write_bytes(deterministic_npz(roster))
    missing_ids = arrays["entity_id"][~arrays["source_static_row_present"]]
    missing_payload = "".join(f"{item}\n" for item in missing_ids).encode("ascii")
    missing_path = args.output_dir / "source-static-missing-ids.txt"
    missing_path.write_bytes(missing_payload)
    action_payload = "".join(f"{item}\n" for item in roster["action_ids"]).encode("ascii")
    action_path = args.output_dir / "action-ids.txt"
    action_path.write_bytes(action_payload)
    fitting_payload = "".join(f"{item}\n" for item in roster["fitting_action_ids"]).encode("ascii")
    fitting_path = args.output_dir / "fitting-action-ids.txt"
    fitting_path.write_bytes(fitting_payload)

    manifest = {
        "schema": "slp.k562-essential-count-static577/v1",
        "artifacts": {
            "features": {
                "path": OUTPUT_NAME,
                "sha256": sha256_file(feature_path),
                "bytes": feature_path.stat().st_size,
                "rawShape": [audit["entities"], 577],
                "normalizedShape": [audit["entities"], 577],
            },
            "rosterIndex": {
                "path": ROSTER_NAME,
                "sha256": sha256_file(roster_path),
                "bytes": roster_path.stat().st_size,
            },
            "sourceStaticMissingRoster": {
                "path": missing_path.name,
                "rows": len(missing_ids),
                "sha256": hashlib.sha256(missing_payload).hexdigest(),
            },
            "actionRoster": {
                "path": action_path.name,
                "rows": len(roster["action_ids"]),
                "sha256": hashlib.sha256(action_payload).hexdigest(),
            },
            "fittingActionRoster": {
                "path": fitting_path.name,
                "rows": len(roster["fitting_action_ids"]),
                "sha256": hashlib.sha256(fitting_payload).hexdigest(),
            },
        },
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "taxon": TAXON,
            "namespace": "stable unversioned Ensembl gene",
            "entityOrdering": "ascending codepoint stable ENSG",
            "queryOrdering": "exact source var/gene_id order from routing sidecar",
            "actionOrdering": "ascending codepoint stable ENSG",
        },
        "featureDefinition": {
            "dimension": 577,
            "columns": {
                "0:320": "frozen ESM-2 t6 8M full-protein vector; zero if unavailable",
                "320": "exact Ensembl-116 selected-translation presence",
                "321:577": "eligible direct MF/CC annotations projected through frozen shared human/yeast GO basis",
            },
            "rawArray": "feature_values",
            "normalizedArray": "normalized_feature_values",
            "normalization": "column population mean and SD on the 1,443 unique split-train action genes, including unavailable zero rows; scale=1 where SD<=1e-5; same transform for query and action rows",
            "availability": [
                "source_static_row_present",
                "esm_present",
                "go_direct_annotation_present",
                "go_exact_uniprot_mapping_present",
                "is_ensembl116_translated_gene",
            ],
            "learnedGeneIdentity": False,
            "sharedGoBasisSha256": BASIS_SHA256,
        },
        "coverage": audit,
        "missingProjection": {
            "rowsOutsideFrozenHumanStaticPack": len(missing),
            "exactEnsembl108UniProtMappingRows": len(exact_mapping_ids & missing),
            "eligibleFrozenBasisProjectionRows": projection_audit[
                "rowsWithEligibleTerms"
            ],
            "retainedAssociations": projection_audit["retainedAssociations"],
            "semantics": "Source-missing rows retain zero ESM/presence; eligible direct MF/CC terms are projected through the existing basis when an exact pinned Ensembl-UniProt mapping exists.",
        },
        "inputs": {
            label: {"path": path.as_posix(), "sha256": digest}
            for label, (path, digest) in pins.items()
        },
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "source": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "accessBoundary": {
            "routingMetadataRead": True,
            "staticFeatureAndAnnotationSourcesRead": True,
            "countMatrixValuesRead": False,
            "quantitativePerturbationOutcomesRead": False,
            "testOutcomeValuesRead": False,
        },
        "limitations": [
            "The 215 measured RNA queries outside the frozen static universe are retained but share an all-zero raw static vector because no Ensembl-116 translation or exact pinned Ensembl-108 UniProt mapping is available.",
            "Exact duplicate static rows cannot be distinguished by this feature-only model.",
            "Static annotations may cover held genes but provide no molecular performance evidence.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--routing",
        type=Path,
        default=ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz",
    )
    result.add_argument(
        "--static",
        type=Path,
        default=ROOT / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz",
    )
    result.add_argument(
        "--static-manifest",
        type=Path,
        default=ROOT / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/manifest.json",
    )
    result.add_argument(
        "--shared-go-basis",
        type=Path,
        default=ROOT / "data/derived/slp11-shared-human-yeast-go/goa-2022-09-19-mf-cc-svd256-v1/human-yeast-shared-go-mf-cc-svd256-basis.npz",
    )
    result.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "data/derived/slp11-human-go/source/Homo_sapiens.GRCh38.108.uniprot.tsv.gz",
    )
    result.add_argument(
        "--gaf",
        type=Path,
        default=ROOT / "data/derived/slp11-human-go/source/goa_human_2022-09-19.gaf.gz",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
