#!/usr/bin/env python3
"""Build human ESM-2 plus shared-coordinate MF/CC GO static features.

The output covers the complete Ensembl-116 translated-gene universe plus the
three-source K562/RPE1/GWPS query/action union.  It reads static vectors and
identifier rosters only; molecular values and held-context data are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

TAXON = 9606
OUTPUT_NAME = "human-static-esm8m-shared-go-mf-cc-features.npz"
ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_MODULE_NAME = "build_slp11_sequence_features"
SEQUENCE_SPEC = importlib.util.spec_from_file_location(
    SEQUENCE_MODULE_NAME,
    ROOT / "scripts" / "build_slp11_sequence_features.py",
)
if SEQUENCE_SPEC is None or SEQUENCE_SPEC.loader is None:
    raise RuntimeError("cannot load human GO parser dependency")
SEQUENCE_MODULE = importlib.util.module_from_spec(SEQUENCE_SPEC)
PREVIOUS_SEQUENCE_MODULE = sys.modules.get(SEQUENCE_MODULE_NAME)
sys.modules[SEQUENCE_MODULE_NAME] = SEQUENCE_MODULE
SEQUENCE_SPEC.loader.exec_module(SEQUENCE_MODULE)
GO_SPEC = importlib.util.spec_from_file_location(
    "_slp11_human_go_projection_parser",
    ROOT / "scripts" / "build_slp11_human_go_features.py",
)
if GO_SPEC is None or GO_SPEC.loader is None:
    raise RuntimeError("cannot load pinned human GO parser")
GO_PARSER = importlib.util.module_from_spec(GO_SPEC)
sys.modules[GO_SPEC.name] = GO_PARSER
GO_SPEC.loader.exec_module(GO_PARSER)
if PREVIOUS_SEQUENCE_MODULE is None:
    del sys.modules[SEQUENCE_MODULE_NAME]
else:
    sys.modules[SEQUENCE_MODULE_NAME] = PREVIOUS_SEQUENCE_MODULE
EXPECTED = {
    "translated": "f4bbfe62b73cf6362170996fcf34200cea68da106d687d3c9e994e709e951f40",
    "translated_manifest": "5fc95466a547b76b43f4a7223066c119c37abd3fef69a5bf5460ea3cfe245e9c",
    "source3_esm": "0b05b14ac496352e98569b2a0d033199925255f6fffefada47028bbc5e576de9",
    "source3_manifest": "3c6b23e9e42a927b528eb7f946743487bc13f68de93378bf9bed77776bdc0927",
    "shared_go": "fb673cf6053bb7bfe88c6b454cedb662646f7256f094abf9a6df1d2865f873f6",
    "shared_go_manifest": "5a95bbd75e4e14666f12b39440303cbd9d68e7aaf87bbdeeb2da480697689624",
    "shared_go_basis": "718764f4ebb6ab9ac31dba65d7d6453525e04a98b999aa7dcfeb4c3a1ab62abd",
    "human_go_gaf": "8b97980a895cb74255615f7cdbdd818f72a3999867b7d2a14f867874480693e1",
    "human_go_mapping": "0d6fe982ce7023b2901171fd0e1419a2e9fbd7fbb9b3473a82ac6dee454f6e56",
    "query_roster": "645b8d563b440a4b7ab6a3bb42450594b408c4e7cb84e4fe2789a6620174f12c",
    "essential_action_roster": "2884efd414949bfc3c7dc5f376aa69f0470080afdcab255b4a88f67cc53ac9ed",
    "gwps_action_roster": "cb89e8110aaf63e1fcb9f21b04b10bef2626e5d02e435bf379b24858bea8b9b8",
}


class HumanStaticError(ValueError):
    """Raised when a frozen human static-feature contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise HumanStaticError(f"SHA-256 mismatch for {path}: {actual}")


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def load_roster(path: Path, expected_sha256: str) -> list[str]:
    require_hash(path, expected_sha256)
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise HumanStaticError("roster must be LF terminated")
    try:
        values = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise HumanStaticError("roster is not ASCII") from exc
    if values != sorted(set(values)) or any(
        len(value) != 15 or not value.startswith("ENSG") or not value[4:].isdigit()
        for value in values
    ):
        raise HumanStaticError("roster must contain sorted unique stable ENSG IDs")
    return values


def rows(
    arrays: dict[str, np.ndarray], dimension: int
) -> dict[tuple[int, str], np.ndarray]:
    values = arrays.get("feature_values")
    taxon = arrays.get("entity_taxon")
    ids = arrays.get("entity_id")
    if (
        values is None
        or taxon is None
        or ids is None
        or values.shape != (len(ids), dimension)
        or taxon.shape != ids.shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise HumanStaticError("invalid static feature arrays")
    result: dict[tuple[int, str], np.ndarray] = {}
    previous: tuple[int, str] | None = None
    for index, (raw_taxon, raw_id) in enumerate(zip(taxon, ids, strict=True)):
        key = (int(raw_taxon), str(raw_id))
        if previous is not None and key <= previous:
            raise HumanStaticError("feature keys must be unique and sorted")
        previous = key
        result[key] = values[index]
    return result


def validate_manifests(
    translated_manifest_path: Path,
    source3_manifest_path: Path,
    shared_go_manifest_path: Path,
) -> dict[str, dict[str, object]]:
    translated = json.loads(translated_manifest_path.read_text(encoding="utf-8"))
    source3 = json.loads(source3_manifest_path.read_text(encoding="utf-8"))
    shared = json.loads(shared_go_manifest_path.read_text(encoding="utf-8"))
    if (
        translated.get("schema")
        != "slp.frangieh-fixed-universe-static-features/v1"
        or translated.get("identity", {}).get("taxon") != TAXON
        or translated.get("identity", {}).get("graphUniverseRows") != 23879
        or translated.get("baseFeatures", {}).get("sequence", {}).get(
            "dimensions"
        )
        != 321
    ):
        raise HumanStaticError("translated-universe source contract mismatch")
    source_sequence = source3.get("sequence", {})
    if (
        source3.get("schema") != "slp.gwps-extended-human-static-features/v1"
        or source3.get("identity", {}).get("ncbiTaxon") != TAXON
        or source_sequence.get("proteinPresentColumn") != 320
        or source_sequence.get("esm", {}).get("revision")
        != "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
    ):
        raise HumanStaticError("source-three ESM contract mismatch")
    feature = shared.get("featureDefinition", {})
    invariant = shared.get("coverage", {}).get("identicalRowInvariant", {})
    if (
        shared.get("schema") != "slp.shared-human-yeast-go-mf-cc-svd/v1"
        or feature.get("dimensions") != 256
        or feature.get("projection")
        != "unweighted binary direct-term row @ shared components.T"
        or invariant.get("exactVectorEquality") is not True
    ):
        raise HumanStaticError("shared GO coordinate contract mismatch")
    return {"translated": translated, "source3": source3, "sharedGo": shared}


def assemble(
    translated: dict[tuple[int, str], np.ndarray],
    source3: dict[tuple[int, str], np.ndarray],
    shared_go: dict[tuple[int, str], np.ndarray],
    shared_go_direct: dict[tuple[int, str], bool],
    queries: set[str],
    essential_actions: set[str],
    gwps_actions: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    translated_ids = {key[1] for key in translated if key[0] == TAXON}
    source3_ids = queries | essential_actions | gwps_actions
    ids = sorted(translated_ids | source3_ids)
    overlap = sorted(translated_ids & set(source3_ids))
    conflicts = [
        entity_id
        for entity_id in overlap
        if not np.array_equal(
            translated[(TAXON, entity_id)][:321], source3[(TAXON, entity_id)]
        )
    ]
    if conflicts:
        raise HumanStaticError(
            f"translated/source-three ESM conflict for {len(conflicts)} rows"
        )
    if {key[1] for key in shared_go if key[0] == TAXON} != translated_ids | source3_ids:
        raise HumanStaticError("shared GO projection does not cover the output universe")

    n = len(ids)
    values = np.zeros((n, 577), dtype=np.float32)
    translated_flag = np.zeros(n, dtype=np.bool_)
    esm_present = np.zeros(n, dtype=np.bool_)
    go_identity = np.zeros(n, dtype=np.bool_)
    go_direct = np.zeros(n, dtype=np.bool_)
    query_flag = np.zeros(n, dtype=np.bool_)
    essential_action_flag = np.zeros(n, dtype=np.bool_)
    gwps_action_flag = np.zeros(n, dtype=np.bool_)
    for index, entity_id in enumerate(ids):
        key = (TAXON, entity_id)
        sequence = translated.get(key, source3.get(key))
        if sequence is None or sequence.shape[0] < 321:
            raise HumanStaticError(f"missing sequence row for {entity_id}")
        values[index, :321] = sequence[:321]
        esm_present[index] = bool(sequence[320] == np.float32(1.0))
        if sequence[320] not in (np.float32(0.0), np.float32(1.0)):
            raise HumanStaticError("protein presence column is not binary")
        translated_flag[index] = entity_id in translated_ids
        go = shared_go.get(key)
        if go is not None:
            values[index, 321:] = go
            go_identity[index] = True
            go_direct[index] = shared_go_direct[key]
        query_flag[index] = entity_id in queries
        essential_action_flag[index] = entity_id in essential_actions
        gwps_action_flag[index] = entity_id in gwps_actions
    if not np.isfinite(values).all():
        raise HumanStaticError("nonfinite feature value")
    arrays = {
        "feature_values": values,
        "entity_taxon": np.full(n, TAXON, dtype=np.int64),
        "entity_id": np.asarray(ids),
        "esm_present": esm_present,
        "go_identity_present": go_identity,
        "go_direct_annotation_present": go_direct,
        "is_ensembl116_translated_gene": translated_flag,
        "is_source3_query": query_flag,
        "is_source3_essential_action": essential_action_flag,
        "is_source3_gwps_action": gwps_action_flag,
    }
    audit = {
        "translatedRows": len(translated_ids),
        "source3Rows": len(source3_ids),
        "source3TranslatedRows": len(source3_ids & translated_ids),
        "source3ProteinMissingRows": len(source3_ids - translated_ids),
        "unionRows": len(ids),
        "overlapRowsCompared": len(overlap),
        "sequenceConflicts": len(conflicts),
    }
    return arrays, audit


def project_additional_go_rows(
    shared_arrays: dict[str, np.ndarray],
    basis_arrays: dict[str, np.ndarray],
    identifiers: list[str],
    mapping_path: Path,
    gaf_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Project extra stable ENSGs through the existing shared basis."""

    components = basis_arrays.get("components")
    term_ids = basis_arrays.get("term_id")
    if (
        components is None
        or term_ids is None
        or components.shape != (256, 6876)
        or components.dtype != np.float32
        or term_ids.shape != (6876,)
    ):
        raise HumanStaticError("shared GO basis arrays mismatch")
    vocabulary = [str(term) for term in term_ids]
    vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
    xrefs, mapping_stats = GO_PARSER.parse_mapping_bytes(
        mapping_path.read_bytes(), frozenset(identifiers)
    )
    term_rows, gaf_stats = GO_PARSER.parse_gaf_bytes(
        gaf_path.read_bytes(), xrefs, identifiers
    )
    values = np.zeros((len(identifiers), 256), dtype=np.float32)
    direct = np.zeros(len(identifiers), dtype=np.bool_)
    omitted_terms: set[str] = set()
    retained_associations = 0
    for row, terms in enumerate(term_rows):
        retained = sorted(term for term in terms if term in vocabulary_index)
        omitted_terms.update(term for term in terms if term not in vocabulary_index)
        if retained:
            indices = np.asarray(
                [vocabulary_index[term] for term in retained], dtype=np.int64
            )
            values[row] = components[:, indices].sum(axis=1, dtype=np.float32)
            direct[row] = True
            retained_associations += len(retained)
    combined = {
        "feature_values": np.vstack(
            [shared_arrays["feature_values"], values]
        ).astype(np.float32),
        "entity_taxon": np.concatenate(
            [
                shared_arrays["entity_taxon"],
                np.full(len(identifiers), TAXON, dtype=np.int64),
            ]
        ),
        "entity_id": np.concatenate(
            [shared_arrays["entity_id"], np.asarray(identifiers)]
        ),
        "direct_annotation_present": np.concatenate(
            [shared_arrays["direct_annotation_present"], direct]
        ),
    }
    order = np.lexsort((combined["entity_id"], combined["entity_taxon"]))
    combined = {name: value[order] for name, value in combined.items()}
    return combined, {
        "projectedRows": len(identifiers),
        "rowsWithSharedVocabularyTerms": int(direct.sum()),
        "retainedAssociations": retained_associations,
        "eligibleTermsOutsideFrozenSharedVocabulary": len(omitted_terms),
        "mapping": mapping_stats,
        "annotations": gaf_stats,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "translated": args.translated,
        "translated_manifest": args.translated_manifest,
        "source3_esm": args.source3_esm,
        "source3_manifest": args.source3_manifest,
        "shared_go": args.shared_go,
        "shared_go_manifest": args.shared_go_manifest,
        "shared_go_basis": args.shared_go_basis,
        "human_go_gaf": args.human_go_source / GO_PARSER.GO_NAME,
        "human_go_mapping": args.human_go_source / GO_PARSER.MAPPING_NAME,
        "query_roster": args.query_roster,
        "essential_action_roster": args.essential_action_roster,
        "gwps_action_roster": args.gwps_action_roster,
    }
    for name, path in paths.items():
        require_hash(path, EXPECTED[name])
    manifests = validate_manifests(
        args.translated_manifest, args.source3_manifest, args.shared_go_manifest
    )
    translated_arrays = load_npz(args.translated)
    source3_arrays = load_npz(args.source3_esm)
    shared_arrays = load_npz(args.shared_go)
    basis_arrays = load_npz(args.shared_go_basis)
    translated = rows(translated_arrays, 577)
    source3 = rows(source3_arrays, 321)
    queries = set(load_roster(args.query_roster, EXPECTED["query_roster"]))
    essential_actions = set(
        load_roster(
            args.essential_action_roster, EXPECTED["essential_action_roster"]
        )
    )
    gwps_actions = set(
        load_roster(args.gwps_action_roster, EXPECTED["gwps_action_roster"])
    )
    expected_source3 = queries | essential_actions | gwps_actions
    missing_source_rows = sorted(
        entity_id
        for entity_id in expected_source3
        if (TAXON, entity_id) not in source3
    )
    if missing_source_rows:
        raise HumanStaticError(
            f"source-three ESM artifact misses {len(missing_source_rows)} roster IDs"
        )
    translated_ids = {key[1] for key in translated if key[0] == TAXON}
    additional_go_ids = sorted(expected_source3 - translated_ids)
    shared_arrays, additional_go_audit = project_additional_go_rows(
        shared_arrays,
        basis_arrays,
        additional_go_ids,
        args.human_go_source / GO_PARSER.MAPPING_NAME,
        args.human_go_source / GO_PARSER.GO_NAME,
    )
    shared_go = rows(shared_arrays, 256)
    shared_direct = {
        (int(taxon), str(entity_id)): bool(present)
        for taxon, entity_id, present in zip(
            shared_arrays["entity_taxon"],
            shared_arrays["entity_id"],
            shared_arrays["direct_annotation_present"],
            strict=True,
        )
    }
    arrays, audit = assemble(
        translated,
        source3,
        shared_go,
        shared_direct,
        queries,
        essential_actions,
        gwps_actions,
    )
    if audit != {
        "translatedRows": 23879,
        "source3Rows": 10213,
        "source3TranslatedRows": 10061,
        "source3ProteinMissingRows": 152,
        "unionRows": 24031,
        "overlapRowsCompared": 10061,
        "sequenceConflicts": 0,
    }:
        raise HumanStaticError(f"unexpected universe audit: {audit}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_NAME
    manifest_path = args.output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise HumanStaticError("immutable output already exists")
    output_path.write_bytes(deterministic_npz(arrays))
    missing = [
        str(entity_id)
        for entity_id, present in zip(
            arrays["entity_id"], arrays["esm_present"], strict=True
        )
        if not present
    ]
    missing_payload = "".join(f"{entity_id}\n" for entity_id in missing).encode(
        "ascii"
    )
    (args.output_dir / "protein-missing-source3-ids.txt").write_bytes(missing_payload)

    def role_coverage(mask: np.ndarray) -> dict[str, int]:
        return {
            "rows": int(mask.sum()),
            "esmPresent": int((mask & arrays["esm_present"]).sum()),
            "goIdentityPresent": int((mask & arrays["go_identity_present"]).sum()),
            "goDirectAnnotationPresent": int(
                (mask & arrays["go_direct_annotation_present"]).sum()
            ),
        }

    source3_mask = (
        arrays["is_source3_query"]
        | arrays["is_source3_essential_action"]
        | arrays["is_source3_gwps_action"]
    )
    shared_manifest = manifests["sharedGo"]
    manifest = {
        "schema": "slp.human-esm8m-shared-go-mf-cc-static-features/v1",
        "artifact": {
            "path": OUTPUT_NAME,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "shape": [24031, 577],
            "dtype": "float32",
        },
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "taxon": TAXON,
            "namespace": "stable unversioned Ensembl gene",
            "ordering": "ascending taxon then codepoint entity ID",
            **audit,
        },
        "featureDefinition": {
            "dimension": 577,
            "columns": {
                "0:320": "frozen ESM-2 t6 8M full-protein vector, zero when absent",
                "320": "exact Ensembl-116 selected-translation presence flag",
                "321:577": "unweighted direct MF/CC row projected in the frozen shared human/yeast GO basis",
            },
            "normalization": "none; downstream standardization must be fit on the applicable fitting partition",
            "learnedGeneIdentity": False,
        },
        "coverage": {
            "union": role_coverage(np.ones(len(arrays["entity_id"]), dtype=np.bool_)),
            "translatedUniverse": role_coverage(
                arrays["is_ensembl116_translated_gene"]
            ),
            "source3Union": role_coverage(source3_mask),
            "source3Queries": role_coverage(arrays["is_source3_query"]),
            "source3EssentialActions": role_coverage(
                arrays["is_source3_essential_action"]
            ),
            "source3GwpsActions": role_coverage(arrays["is_source3_gwps_action"]),
        },
        "sequenceCompatibilityAudit": {
            "translatedVsSource3OverlapRows": audit["overlapRowsCompared"],
            "bitExactRows": audit["overlapRowsCompared"],
            "conflictingRows": audit["sequenceConflicts"],
            "sameModelRevision": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
            "sameEnsemblRelease": 116,
            "sameLongestTranslationSelection": True,
            "sameFullProteinOverlapPooling": True,
        },
        "sharedGoCoordinateContract": {
            "basisSha256": shared_manifest["artifacts"]["basis"]["sha256"],
            "componentsFloat32Sha256": shared_manifest["compression"][
                "componentsFloat32Sha256"
            ],
            "projection": shared_manifest["featureDefinition"]["projection"],
            "crossSpeciesIdenticalAnnotationRowsVectorEqual": shared_manifest[
                "coverage"
            ]["identicalRowInvariant"]["exactVectorEquality"],
            "additionalSource3Projection": additional_go_audit,
        },
        "proteinMissingRoster": {
            "path": "protein-missing-source3-ids.txt",
            "rows": len(missing),
            "sha256": hashlib.sha256(missing_payload).hexdigest(),
        },
        "inputs": {
            name: {"path": str(path).replace("\\", "/"), "sha256": EXPECTED[name]}
            for name, path in paths.items()
        },
        "accessBoundary": {
            "staticFeatureVectorsRead": True,
            "identifierRostersRead": True,
            "quantitativeMolecularOutcomesRead": False,
            "heldContextDataRead": False,
            "benchmarkDataRead": False,
        },
        "limitations": [
            "The 152 source-three IDs without an Ensembl-116 selected translation retain zero ESM blocks; their eligible GO annotations are projected without refitting through the shared basis.",
            "The GO basis is transductively fit to static annotations across both species and contains no quantitative molecular outcomes.",
            "Coordinate compatibility does not itself establish cross-species molecular transfer.",
        ],
    }
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--translated", type=Path, required=True)
    result.add_argument("--translated-manifest", type=Path, required=True)
    result.add_argument("--source3-esm", type=Path, required=True)
    result.add_argument("--source3-manifest", type=Path, required=True)
    result.add_argument("--shared-go", type=Path, required=True)
    result.add_argument("--shared-go-manifest", type=Path, required=True)
    result.add_argument("--shared-go-basis", type=Path, required=True)
    result.add_argument("--human-go-source", type=Path, required=True)
    result.add_argument("--query-roster", type=Path, required=True)
    result.add_argument("--essential-action-roster", type=Path, required=True)
    result.add_argument("--gwps-action-roster", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
