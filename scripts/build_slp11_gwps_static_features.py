#!/usr/bin/env python3
"""Extend frozen human static features to every K562 GWPS intervention gene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import build_slp11_human_go_features as go
import build_slp11_human_sequence_features as seq
import build_slp11_norman_static_features as frozen
import numpy as np

TAXON = 9606
BASE_COUNT = 7_605
ACTION_COUNT = 9_852
ADDED_COUNT = 2_626
EXTENDED_COUNT = 10_231
BASE_ENTITY_SHA256 = "6f282a37e7aa303e23b9f6c3bf61127c83c850438a7e16740c53e6cf85a5944e"
ACTION_SHA256 = "cb89e8110aaf63e1fcb9f21b04b10bef2626e5d02e435bf379b24858bea8b9b8"
BASE_FEATURE_SPECS = {
    "sequence": (
        "bcdc643b0e84966ee83eea069a9cbf93c1c2bfb7a47cf68201547d838f28eb19",
        321,
    ),
    "go": (
        "1b35550da7b518458ee5f213581cf411485f2e4d9c74455d1029d5e286e7cc21",
        256,
    ),
    "fusion": (
        "7b3d78af66f013e2d1df3a3f98924707ed111bc795757753e82a5e8f495408b5",
        577,
    ),
}
ORIGINAL_GO_SHA256 = "208be756b81229b3881af8229e18ba2f5e806f5be85180b6f5560c3f2d07c0ea"
GO_COMPONENT_SHA256 = "44dc50187681703238b66a905750cfd25decbba8e9adb457a8f77bb69a2f5f2d"
OUTPUT_NAMES = {
    "sequence": "gwps-extended-esm2-features.npz",
    "go": "gwps-extended-go-mf-cc-features.npz",
    "fusion": "gwps-extended-static-esm-go-features.npz",
}


class GwpsStaticFeatureError(ValueError):
    """A GWPS static-feature extension contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path, count: int, digest: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise GwpsStaticFeatureError(f"ID roster SHA-256 mismatch: {path}")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise GwpsStaticFeatureError("ID roster must be LF-terminated ASCII")
    values = tuple(payload.decode("ascii").splitlines())
    if (
        len(values) != count
        or list(values) != sorted(set(values))
        or any(seq.ENTITY_RE.fullmatch(item) is None for item in values)
    ):
        raise GwpsStaticFeatureError("ID roster identity or ordering mismatch")
    return values


def merge_ids(
    base_ids: Sequence[str], action_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base = set(base_ids)
    extended = tuple(sorted(base | set(action_ids)))
    added = tuple(item for item in extended if item not in base)
    return extended, added


def load_feature(
    path: Path, ids: Sequence[str], label: str
) -> np.ndarray:
    digest, dimensions = BASE_FEATURE_SPECS[label]
    if sha256_file(path) != digest:
        raise GwpsStaticFeatureError(f"base {label} feature SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("feature_values", "entity_taxon", "entity_id"):
            raise GwpsStaticFeatureError(f"base {label} NPZ schema mismatch")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        entity_ids = archive["entity_id"]
    if (
        values.shape != (BASE_COUNT, dimensions)
        or values.dtype != np.float32
        or entity_ids.tolist() != list(ids)
        or not np.array_equal(taxa, np.full(BASE_COUNT, TAXON, dtype=np.int64))
    ):
        raise GwpsStaticFeatureError(f"base {label} feature array mismatch")
    return values


def output_arrays(ids: Sequence[str], values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "feature_values": values.astype(np.dtype("<f4"), copy=False),
        "entity_taxon": np.full(len(ids), TAXON, dtype=np.dtype("<i8")),
        "entity_id": np.asarray(ids, dtype="<U15"),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    base_ids = load_ids(args.base_entity_ids, BASE_COUNT, BASE_ENTITY_SHA256)
    action_ids = load_ids(args.gwps_action_ids, ACTION_COUNT, ACTION_SHA256)
    extended_ids, added_ids = merge_ids(base_ids, action_ids)
    if len(extended_ids) != EXTENDED_COUNT or len(added_ids) != ADDED_COUNT:
        raise GwpsStaticFeatureError("unexpected GWPS feature-universe size")
    old_sequence = load_feature(args.base_sequence_npz, base_ids, "sequence")
    old_go = load_feature(args.base_go_npz, base_ids, "go")
    old_fusion = load_feature(args.base_fusion_npz, base_ids, "fusion")
    if not np.array_equal(old_fusion[:, :321], old_sequence) or not np.array_equal(
        old_fusion[:, 321:], old_go
    ):
        raise GwpsStaticFeatureError("base fusion blocks differ from source packs")
    sequence_values, row_by_id = frozen.copy_frozen_rows(
        old_sequence, base_ids, extended_ids
    )
    fasta = seq.verify_source_dir(args.sequence_source_dir)
    model_files = seq.verify_esm_model_dir(args.esm_model_dir)
    translations, source_counts = seq.parse_longest_translations(fasta)
    present_added = tuple(item for item in added_ids if item in translations)
    missing_added = tuple(item for item in added_ids if item not in translations)
    embeddings, esm_stats = seq.extract_esm(
        [seq.normalize_for_esm(translations[item].peptide) for item in present_added],
        args.esm_model_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        max_residues=seq.ESM_MAX_RESIDUES,
        overlap=seq.ESM_DEFAULT_OVERLAP,
    )
    for item, embedding in zip(present_added, embeddings, strict=True):
        sequence_values[row_by_id[item], :320] = embedding
        sequence_values[row_by_id[item], 320] = np.float32(1.0)

    if sha256_file(args.original_go_npz) != ORIGINAL_GO_SHA256:
        raise GwpsStaticFeatureError("original frozen GO artifact SHA-256 mismatch")
    original_ids = seq.load_entity_ids(args.original_entity_ids)
    with np.load(args.original_go_npz, allow_pickle=False) as archive:
        original_go = archive["feature_values"]
    source = args.go_source_dir
    mapping_payload = go.require_file(
        source / go.MAPPING_NAME, go.MAPPING_BYTES, go.MAPPING_SHA256, "Ensembl mapping"
    )
    gaf_payload = go.require_file(source / go.GO_NAME, go.GO_BYTES, go.GO_SHA256, "GO GAF")
    original_xrefs, _ = go.parse_mapping_bytes(mapping_payload, frozenset(original_ids))
    original_terms, _ = go.parse_gaf_bytes(gaf_payload, original_xrefs, original_ids)
    original_matrix, terms, _ = go.direct_matrix(original_terms)
    reconstructed, svd = go.fit_svd(original_matrix, 256, 731)
    component_hash = hashlib.sha256(
        svd.components_.astype(np.dtype("<f4"), copy=False).tobytes("C")
    ).hexdigest()
    if component_hash != GO_COMPONENT_SHA256 or not np.array_equal(
        reconstructed, original_go
    ):
        raise GwpsStaticFeatureError("original GO basis did not reconstruct exactly")
    added_xrefs, _ = go.parse_mapping_bytes(mapping_payload, frozenset(added_ids))
    added_terms, _ = go.parse_gaf_bytes(gaf_payload, added_xrefs, added_ids)
    added_matrix, omitted_terms = frozen.fixed_term_matrix(added_terms, terms)
    added_basis_present = np.asarray(added_matrix.getnnz(axis=1)).reshape(-1) > 0
    added_go_values = svd.transform(added_matrix).astype(np.float32, copy=False)
    go_values, go_row_by_id = frozen.copy_frozen_rows(old_go, base_ids, extended_ids)
    for item, values in zip(added_ids, added_go_values, strict=True):
        go_values[go_row_by_id[item]] = values

    fusion_values, fusion_row_by_id = frozen.copy_frozen_rows(
        old_fusion, base_ids, extended_ids
    )
    for item in added_ids:
        row = fusion_row_by_id[item]
        fusion_values[row, :321] = sequence_values[row_by_id[item]]
        fusion_values[row, 321:] = go_values[go_row_by_id[item]]
    base_positions = np.asarray([row_by_id[item] for item in base_ids], dtype=np.int64)
    if (
        sequence_values[base_positions].tobytes() != old_sequence.tobytes()
        or go_values[base_positions].tobytes() != old_go.tobytes()
        or fusion_values[base_positions].tobytes() != old_fusion.tobytes()
    ):
        raise GwpsStaticFeatureError("an existing feature row changed at byte level")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for label, values in (
        ("sequence", sequence_values),
        ("go", go_values),
        ("fusion", fusion_values),
    ):
        payload = seq.deterministic_npz_bytes(output_arrays(extended_ids, values))
        path = output / OUTPUT_NAMES[label]
        if path.exists():
            raise GwpsStaticFeatureError(f"refusing to overwrite {path}")
        path.write_bytes(payload)
        outputs[label] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dimensions": values.shape[1],
        }
    entity_payload = "".join(f"{item}\n" for item in extended_ids).encode("ascii")
    entity_path = output / "entity-ids.txt"
    entity_path.write_bytes(entity_payload)
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
    provenance_path = output / "added-protein-provenance.jsonl"
    provenance_path.write_bytes(provenance_payload)
    action_positions = np.asarray([row_by_id[item] for item in action_ids], dtype=np.int64)
    action_go_present = np.any(go_values[action_positions] != 0, axis=1)
    manifest = {
        "schema": "slp.gwps-extended-human-static-features/v1",
        "identity": {
            "ncbiTaxon": TAXON,
            "namespace": "Ensembl-gene",
            "rows": len(extended_ids),
            "baseRows": len(base_ids),
            "addedGwpsActionRows": len(added_ids),
            "entityList": {
                "path": entity_path.name,
                "bytes": len(entity_payload),
                "sha256": hashlib.sha256(entity_payload).hexdigest(),
            },
            "gwpsActionRosterSha256": ACTION_SHA256,
        },
        "frozenRows": {
            "copiedWithoutRecomputation": True,
            "byteEqualityVerified": True,
            "baseArtifactSha256": {
                label: BASE_FEATURE_SPECS[label][0]
                for label in ("sequence", "go", "fusion")
            },
        },
        "sequence": {
            "addedWithProtein": len(present_added),
            "addedWithoutProtein": len(missing_added),
            "missingAddedProteinEntityIds": list(missing_added),
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
            "sourceCounts": source_counts,
            "addedProteinProvenance": {
                "path": provenance_path.name,
                "records": len(provenance_rows),
                "bytes": len(provenance_payload),
                "sha256": hashlib.sha256(provenance_payload).hexdigest(),
            },
        },
        "go": {
            "basisFitRows": len(original_ids),
            "basisRefitOnExtendedUniverse": False,
            "basisReproducedExactly": True,
            "componentFloat32Sha256": component_hash,
            "termCount": len(terms),
            "addedRowsWithEligibleTerms": sum(bool(item) for item in added_terms),
            "addedRowsRepresentedInFrozenBasis": int(np.count_nonzero(added_basis_present)),
            "addedRowsWithoutFrozenBasisTermsIds": [
                item
                for item, present in zip(added_ids, added_basis_present, strict=True)
                if not present
            ],
            "newOnlyTermsOmittedFromFrozenBasis": omitted_terms,
        },
        "gwpsActionCoverage": {
            "actions": len(action_ids),
            "withProteinEmbedding": int(
                np.count_nonzero(sequence_values[action_positions, 320] == 1)
            ),
            "withoutProteinEmbeddingIds": [
                item
                for item, present in zip(
                    action_ids,
                    sequence_values[action_positions, 320] == 1,
                    strict=True,
                )
                if not present
            ],
            "withNonzeroGoProjection": int(np.count_nonzero(action_go_present)),
            "withoutNonzeroGoProjectionIds": [
                item
                for item, present in zip(action_ids, action_go_present, strict=True)
                if not present
            ],
        },
        "outputs": outputs,
        "accessBoundary": {
            "staticActionRosterConsumed": True,
            "staticSequenceConsumed": True,
            "staticAnnotationConsumed": True,
            "quantitativeMolecularOutcomesConsumed": False,
            "testOutcomesConsumed": False,
            "benchmarkDataConsumed": False,
        },
        "status": "exploratory-static-feature-extension-not-omf-admitted",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(seq.canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-entity-ids", type=Path, required=True)
    result.add_argument("--gwps-action-ids", type=Path, required=True)
    result.add_argument("--base-sequence-npz", type=Path, required=True)
    result.add_argument("--base-go-npz", type=Path, required=True)
    result.add_argument("--base-fusion-npz", type=Path, required=True)
    result.add_argument("--original-entity-ids", type=Path, required=True)
    result.add_argument("--original-go-npz", type=Path, required=True)
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
    except (
        OSError,
        GwpsStaticFeatureError,
        frozen.NormanStaticFeatureError,
        go.HumanGoFeatureError,
        seq.HumanSequenceFeatureError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
