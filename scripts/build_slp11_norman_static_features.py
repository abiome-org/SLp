#!/usr/bin/env python3
"""Extend the frozen human static feature rows to Norman 2019 action genes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import build_slp11_human_go_features as go
import build_slp11_human_sequence_features as seq
import numpy as np
from scipy import sparse

TAXON = 9606
BASE_COUNT = 7_542
EXTENDED_COUNT = 7_605
NORMAN_ACTION_COUNT = 105
NEW_ACTION_COUNT = 63
BASE_ENTITY_SHA256 = "c6836645dcfc24788f2c06110ddc08ee4949d97f710dd117db12db1949d9b33e"
ACTION_SHA256 = "99fb1e24574b4bd6c76ea329008ba87fa2352bf6fa97242d033ad79abd78eb1e"
BASE_SPECS = {
    "sequence": (
        "9c0ade1b580f46f26938e5eab6e0222b9e543e44bc2c7d5113336c80459bfb52",
        321,
    ),
    "go": (
        "208be756b81229b3881af8229e18ba2f5e806f5be85180b6f5560c3f2d07c0ea",
        256,
    ),
    "fusion": (
        "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0",
        577,
    ),
}
GO_COMPONENT_SHA256 = "44dc50187681703238b66a905750cfd25decbba8e9adb457a8f77bb69a2f5f2d"
NPZ_NAMES = {
    "sequence": "norman-extended-esm2-features.npz",
    "go": "norman-extended-go-mf-cc-features.npz",
    "fusion": "norman-extended-static-esm-go-features.npz",
}


class NormanStaticFeatureError(ValueError):
    """A frozen-row extension contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path, *, count: int, digest: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise NormanStaticFeatureError(f"ID roster SHA-256 mismatch: {path}")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise NormanStaticFeatureError("ID roster must be LF-terminated")
    try:
        values = tuple(payload.decode("ascii").splitlines())
    except UnicodeDecodeError as exc:
        raise NormanStaticFeatureError("ID roster must be ASCII") from exc
    if (
        len(values) != count
        or list(values) != sorted(set(values))
        or any(seq.ENTITY_RE.fullmatch(item) is None for item in values)
    ):
        raise NormanStaticFeatureError("ID roster identity or ordering mismatch")
    return values


def extend_ids(
    base_ids: Sequence[str], action_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    extended = tuple(sorted(set(base_ids) | set(action_ids)))
    added = tuple(item for item in extended if item not in set(base_ids))
    if len(extended) != EXTENDED_COUNT or len(added) != NEW_ACTION_COUNT:
        raise NormanStaticFeatureError("unexpected Norman static-universe extension")
    return extended, added


def load_base_feature(
    path: Path, ids: Sequence[str], *, label: str
) -> np.ndarray:
    expected_hash, dimensions = BASE_SPECS[label]
    if sha256_file(path) != expected_hash:
        raise NormanStaticFeatureError(f"frozen {label} artifact SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("feature_values", "entity_taxon", "entity_id"):
            raise NormanStaticFeatureError(f"frozen {label} NPZ schema mismatch")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        entity_ids = archive["entity_id"]
    if (
        values.shape != (BASE_COUNT, dimensions)
        or values.dtype != np.float32
        or taxa.dtype != np.int64
        or not np.array_equal(taxa, np.full(BASE_COUNT, TAXON, dtype=np.int64))
        or entity_ids.tolist() != list(ids)
    ):
        raise NormanStaticFeatureError(f"frozen {label} array contract mismatch")
    return values


def copy_frozen_rows(
    old_values: np.ndarray,
    base_ids: Sequence[str],
    extended_ids: Sequence[str],
) -> tuple[np.ndarray, Mapping[str, int]]:
    row_by_id = {item: index for index, item in enumerate(extended_ids)}
    result = np.zeros((len(extended_ids), old_values.shape[1]), dtype=np.float32)
    positions = np.asarray([row_by_id[item] for item in base_ids], dtype=np.int64)
    result[positions] = old_values
    if not np.array_equal(result[positions], old_values):
        raise NormanStaticFeatureError("frozen feature rows changed during extension")
    return result, row_by_id


def fixed_term_matrix(
    entity_terms: Sequence[frozenset[str]], terms: Sequence[str]
) -> tuple[sparse.csr_matrix, int]:
    term_index = {term: index for index, term in enumerate(terms)}
    rows: list[int] = []
    columns: list[int] = []
    omitted: set[str] = set()
    for row, direct in enumerate(entity_terms):
        for term in sorted(direct):
            column = term_index.get(term)
            if column is None:
                omitted.add(term)
            else:
                rows.append(row)
                columns.append(column)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(entity_terms), len(terms)),
        dtype=np.float32,
    )
    return matrix, len(omitted)


def output_arrays(ids: Sequence[str], values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "feature_values": values.astype(np.dtype("<f4"), copy=False),
        "entity_taxon": np.full(len(ids), TAXON, dtype=np.dtype("<i8")),
        "entity_id": np.asarray(ids, dtype="<U15"),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    base_ids = load_ids(
        args.base_entity_ids, count=BASE_COUNT, digest=BASE_ENTITY_SHA256
    )
    action_ids = load_ids(
        args.norman_action_ids, count=NORMAN_ACTION_COUNT, digest=ACTION_SHA256
    )
    extended_ids, added_ids = extend_ids(base_ids, action_ids)
    extended_payload = "".join(f"{item}\n" for item in extended_ids).encode("ascii")

    old_sequence = load_base_feature(
        args.base_sequence_npz, base_ids, label="sequence"
    )
    old_go = load_base_feature(args.base_go_npz, base_ids, label="go")
    old_fusion = load_base_feature(args.base_fusion_npz, base_ids, label="fusion")
    if not np.array_equal(old_fusion[:, :321], old_sequence) or not np.array_equal(
        old_fusion[:, 321:], old_go
    ):
        raise NormanStaticFeatureError("frozen fusion blocks differ from source packs")

    sequence_values, row_by_id = copy_frozen_rows(
        old_sequence, base_ids, extended_ids
    )
    fasta = seq.verify_source_dir(args.sequence_source_dir)
    model_files = seq.verify_esm_model_dir(args.esm_model_dir)
    translations, sequence_source_counts = seq.parse_longest_translations(fasta)
    present_added = tuple(item for item in added_ids if item in translations)
    missing_added = tuple(item for item in added_ids if item not in translations)
    normalized = [seq.normalize_for_esm(translations[item].peptide) for item in present_added]
    embeddings, esm_stats = seq.extract_esm(
        normalized,
        args.esm_model_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        max_residues=seq.ESM_MAX_RESIDUES,
        overlap=seq.ESM_DEFAULT_OVERLAP,
    )
    for item, embedding in zip(present_added, embeddings, strict=True):
        sequence_values[row_by_id[item], :320] = embedding
        sequence_values[row_by_id[item], 320] = np.float32(1.0)

    source = args.go_source_dir
    mapping_payload = go.require_file(
        source / go.MAPPING_NAME, go.MAPPING_BYTES, go.MAPPING_SHA256, "Ensembl mapping"
    )
    gaf_payload = go.require_file(source / go.GO_NAME, go.GO_BYTES, go.GO_SHA256, "GO GAF")
    base_xrefs, _ = go.parse_mapping_bytes(mapping_payload, frozenset(base_ids))
    base_entity_terms, _ = go.parse_gaf_bytes(gaf_payload, base_xrefs, base_ids)
    base_matrix, terms, _ = go.direct_matrix(base_entity_terms)
    reconstructed, svd = go.fit_svd(base_matrix, 256, 731)
    component_hash = hashlib.sha256(
        svd.components_.astype(np.dtype("<f4"), copy=False).tobytes("C")
    ).hexdigest()
    if component_hash != GO_COMPONENT_SHA256 or not np.array_equal(
        reconstructed, old_go
    ):
        raise NormanStaticFeatureError("existing GO SVD basis did not reproduce exactly")
    added_xrefs, _ = go.parse_mapping_bytes(mapping_payload, frozenset(added_ids))
    added_terms, _ = go.parse_gaf_bytes(gaf_payload, added_xrefs, added_ids)
    added_matrix, omitted_terms = fixed_term_matrix(added_terms, terms)
    added_basis_coverage = np.asarray(added_matrix.getnnz(axis=1)).reshape(-1) > 0
    added_go_values = svd.transform(added_matrix).astype(np.float32, copy=False)
    go_values, go_row_by_id = copy_frozen_rows(old_go, base_ids, extended_ids)
    for item, values in zip(added_ids, added_go_values, strict=True):
        go_values[go_row_by_id[item]] = values

    fusion_values, fusion_row_by_id = copy_frozen_rows(
        old_fusion, base_ids, extended_ids
    )
    for item in added_ids:
        fusion_values[fusion_row_by_id[item], :321] = sequence_values[row_by_id[item]]
        fusion_values[fusion_row_by_id[item], 321:] = go_values[go_row_by_id[item]]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for label, values in (
        ("sequence", sequence_values),
        ("go", go_values),
        ("fusion", fusion_values),
    ):
        path = output_dir / NPZ_NAMES[label]
        payload = seq.deterministic_npz_bytes(output_arrays(extended_ids, values))
        path.write_bytes(payload)
        outputs[label] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dimensions": values.shape[1],
        }
    entity_path = output_dir / "entity-ids.txt"
    entity_path.write_bytes(extended_payload)
    provenance_rows = []
    for item in present_added:
        selected = translations[item]
        provenance_rows.append(
            {
                "entityId": item,
                "ncbiTaxon": TAXON,
                "selectedTranscriptId": f"{selected.transcript_id}.{selected.transcript_version}",
                "selectedProteinId": f"{selected.protein_id}.{selected.protein_version}",
                "peptideLength": len(selected.peptide),
                "sourcePeptideSha256": seq.sha256_bytes(selected.peptide),
                "selectionRule": "longest-then-stable-transcript-then-stable-protein",
            }
        )
    provenance_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        for row in provenance_rows
    )
    provenance_path = output_dir / "added-protein-provenance.jsonl"
    provenance_path.write_bytes(provenance_payload)
    old_positions = np.asarray([row_by_id[item] for item in base_ids], dtype=np.int64)
    action_positions = np.asarray(
        [row_by_id[item] for item in action_ids], dtype=np.int64
    )
    action_go_present = np.any(go_values[action_positions] != 0, axis=1)
    manifest = {
        "schema": "slp.norman-extended-human-static-features/v1",
        "identity": {
            "ncbiTaxon": TAXON,
            "namespace": "Ensembl-gene",
            "rows": len(extended_ids),
            "baseRows": len(base_ids),
            "addedNormanActionRows": len(added_ids),
            "entityList": {
                "path": entity_path.name,
                "bytes": len(extended_payload),
                "sha256": hashlib.sha256(extended_payload).hexdigest(),
            },
            "normanActionRosterSha256": ACTION_SHA256,
        },
        "frozenRows": {
            "copiedWithoutRecomputation": True,
            "sequenceArrayEqual": bool(np.array_equal(sequence_values[old_positions], old_sequence)),
            "goArrayEqual": bool(np.array_equal(go_values[old_positions], old_go)),
            "fusionArrayEqual": bool(np.array_equal(fusion_values[old_positions], old_fusion)),
            "baseArtifactSha256": {
                label: BASE_SPECS[label][0] for label in ("sequence", "go", "fusion")
            },
        },
        "normanActionCoverage": {
            "actions": len(action_ids),
            "withProteinEmbedding": int(
                np.count_nonzero(sequence_values[action_positions, 320] == 1)
            ),
            "withNonzeroGoProjection": int(np.count_nonzero(action_go_present)),
            "withoutNonzeroGoProjectionIds": [
                item
                for item, present in zip(action_ids, action_go_present, strict=True)
                if not present
            ],
        },
        "sequence": {
            "addedWithProtein": len(present_added),
            "addedWithoutProtein": len(missing_added),
            "missingProteinEntityIds": list(missing_added),
            "proteinPresentColumn": 320,
            "esm": {
                "repository": seq.ESM_REPOSITORY,
                "revision": seq.ESM_REVISION,
                "files": model_files,
                "pooling": "inverse-overlap-weighted full-residue mean; no truncation",
                "maximumResiduesPerWindow": seq.ESM_MAX_RESIDUES,
                "overlapResidues": seq.ESM_DEFAULT_OVERLAP,
            },
            "extraction": esm_stats,
            "sourceCounts": sequence_source_counts,
            "addedProteinProvenance": {
                "path": provenance_path.name,
                "records": len(provenance_rows),
                "bytes": len(provenance_payload),
                "sha256": hashlib.sha256(provenance_payload).hexdigest(),
            },
        },
        "go": {
            "basisFitRows": BASE_COUNT,
            "basisRefitOnExtendedUniverse": False,
            "basisReproducedExactly": True,
            "componentFloat32Sha256": component_hash,
            "termCount": len(terms),
            "addedRowsWithEligibleTerms": sum(bool(item) for item in added_terms),
            "addedRowsWithoutEligibleTerms": sum(not item for item in added_terms),
            "addedRowsRepresentedInFrozenBasis": int(
                np.count_nonzero(added_basis_coverage)
            ),
            "addedRowsWithoutFrozenBasisTermsIds": [
                item
                for item, present in zip(added_ids, added_basis_coverage, strict=True)
                if not present
            ],
            "newOnlyTermsOmittedFromFrozenBasis": omitted_terms,
            "zeroVectorMeaning": "no eligible term represented in the frozen term set",
        },
        "outputs": outputs,
        "accessBoundary": {
            "staticSequenceConsumed": True,
            "staticAnnotationConsumed": True,
            "quantitativeMolecularOutcomesConsumed": False,
            "benchmarkDataConsumed": False,
        },
        "status": "exploratory-static-feature-extension-not-omf-admitted",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(seq.canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-entity-ids", type=Path, required=True)
    result.add_argument("--norman-action-ids", type=Path, required=True)
    result.add_argument("--base-sequence-npz", type=Path, required=True)
    result.add_argument("--base-go-npz", type=Path, required=True)
    result.add_argument("--base-fusion-npz", type=Path, required=True)
    result.add_argument("--sequence-source-dir", type=Path, required=True)
    result.add_argument("--esm-model-dir", type=Path, required=True)
    result.add_argument("--go-source-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=16)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = build(args)
    except (OSError, NormanStaticFeatureError, go.HumanGoFeatureError, seq.HumanSequenceFeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
