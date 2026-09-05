#!/usr/bin/env python3
"""Fuse frozen yeast ESM-2 and shared-coordinate MF/CC GO features.

The builder reads only identifier rosters and static feature artifacts.  It
never reads molecular target arrays or partition assignments.  Missing static
modalities are represented by zero feature blocks and explicit Boolean arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

TAXON = 4932
STRICT_MAPPING_CLASSES = frozenset(
    {
        "current-orf-systematic",
        "current-orf-standard",
        "current-feature-systematic",
        "current-feature-standard",
    }
)
EXPECTED = {
    "rna_mapping": "8e40587551a329b94e73989fe284240116645986e00cbd05fc2f1bd52bc01643",
    "development": "42f754425637bdf0413dbac6c36206737b5e402e04ba9732aa329cf2f1e702d5",
    "esm": "96f5e1b81036e0d42238ed6ac797f9fd399006f4d5f8227e96d9ee11358318ca",
    "sequence421": "f0a2a439a9ca17c066c2fedaf089ab5cb70ed2ea8d26fa83ed886d581ebfcad4",
    "shared_go": "fb673cf6053bb7bfe88c6b454cedb662646f7256f094abf9a6df1d2865f873f6",
    "shared_go_manifest": "5a95bbd75e4e14666f12b39440303cbd9d68e7aaf87bbdeeb2da480697689624",
    "current_orfs": "df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d",
}
EXPECTED_COUNTS = {
    "strict_queries": 6683,
    "actions": 1732,
    "current_orfs": 6613,
    "union": 6735,
}
OUTPUT_NAME = "yeast-static-esm8m-shared-go-mf-cc-features.npz"


class FeatureBuildError(ValueError):
    """Raised when a frozen static feature contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FeatureBuildError(f"SHA-256 mismatch for {path}: {actual}")


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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureBuildError(f"invalid JSONL row {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise FeatureBuildError(f"non-object JSONL row {line_number}: {path}")
            rows.append(row)
    return rows


def read_action_ids_without_targets(path: Path) -> list[str]:
    """Read only action_ids.npy from the response archive.

    Direct zip-member access prevents loading target, observed, or split arrays.
    """

    with zipfile.ZipFile(path, "r") as archive:
        if "action_ids.npy" not in archive.namelist():
            raise FeatureBuildError("development artifact lacks action_ids.npy")
        with archive.open("action_ids.npy") as stream:
            values = np.lib.format.read_array(stream, allow_pickle=False)
    if values.ndim != 1 or values.dtype.kind != "U":
        raise FeatureBuildError("action_ids must be a one-dimensional Unicode array")
    ids = values.tolist()
    if any(not value.startswith("SGD:S") for value in ids):
        raise FeatureBuildError("action roster contains a non-SGD stable identifier")
    return sorted(set(ids))


def strict_query_ids(rows: list[dict[str, object]]) -> list[str]:
    ids = [
        row.get("canonicalSgdCurie")
        for row in rows
        if row.get("mappingClass") in STRICT_MAPPING_CLASSES
    ]
    if any(not isinstance(value, str) for value in ids):
        raise FeatureBuildError("strict mapping row lacks canonical SGD CURIE")
    result = sorted(ids)  # type: ignore[arg-type]
    if len(result) != len(set(result)):
        raise FeatureBuildError("strict query mapping contains duplicate stable IDs")
    return result


def current_orf_ids(rows: list[dict[str, object]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        value = row.get("canonicalSgdCurie")
        if (
            row.get("schema") != "slp.sgd-current-orf/v1"
            or row.get("ncbiTaxon") != TAXON
            or not isinstance(value, str)
        ):
            raise FeatureBuildError("invalid current SGD ORF mapping row")
        if value in ids:
            raise FeatureBuildError("duplicate current SGD ORF")
        ids.add(value)
    return ids


def load_static_rows(
    path: Path,
    dimension: int,
    presence_name: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"entity_taxon", "entity_id", "feature_values"}
        if not required.issubset(data.files):
            raise FeatureBuildError(f"missing arrays in {path}")
        taxon = data["entity_taxon"]
        ids = data["entity_id"]
        values = data["feature_values"]
        if values.shape != (ids.size, dimension) or taxon.shape != ids.shape:
            raise FeatureBuildError(f"invalid static feature shape in {path}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise FeatureBuildError(f"invalid static feature values in {path}")
        if presence_name is None:
            present = np.ones(ids.size, dtype=np.bool_)
        else:
            if presence_name not in data.files:
                raise FeatureBuildError(f"missing {presence_name} in {path}")
            present = data[presence_name]
            if present.shape != ids.shape or present.dtype != np.bool_:
                raise FeatureBuildError(f"invalid {presence_name} in {path}")
        result: dict[str, np.ndarray] = {}
        flags: dict[str, bool] = {}
        previous: tuple[int, str] | None = None
        for index, (raw_taxon, raw_id) in enumerate(zip(taxon, ids, strict=True)):
            key = (int(raw_taxon), str(raw_id))
            if previous is not None and key <= previous:
                raise FeatureBuildError(f"static keys are not unique and sorted: {path}")
            previous = key
            if key[0] == TAXON and key[1].startswith("SGD:"):
                result[key[1]] = values[index].copy()
                flags[key[1]] = bool(present[index])
        return result, flags


def validate_shared_go_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    feature = manifest.get("featureDefinition", {})
    coverage = manifest.get("coverage", {})
    invariant = coverage.get("identicalRowInvariant", {}) if isinstance(coverage, dict) else {}
    compression = manifest.get("compression", {})
    if (
        manifest.get("schema") != "slp.shared-human-yeast-go-mf-cc-svd/v1"
        or feature.get("dimensions") != 256
        or feature.get("projection")
        != "unweighted binary direct-term row @ shared components.T"
        or compression.get("humanTotalSquaredRowWeight") != 1.0
        or compression.get("yeastTotalSquaredRowWeight") != 1.0
        or invariant.get("exactVectorEquality") is not True
    ):
        raise FeatureBuildError("shared GO semantic-coordinate contract mismatch")
    return manifest


def assemble_features(
    entity_ids: list[str],
    esm: dict[str, np.ndarray],
    go: dict[str, np.ndarray],
    go_direct: dict[str, bool],
    sequence421_ids: set[str],
    source_sequence_ids: set[str],
    query_ids: set[str],
    action_ids: set[str],
) -> dict[str, np.ndarray]:
    if entity_ids != sorted(set(entity_ids)):
        raise FeatureBuildError("output entity IDs must be unique and sorted")
    n = len(entity_ids)
    values = np.zeros((n, 577), dtype=np.float32)
    esm_present = np.zeros(n, dtype=np.bool_)
    go_identity_present = np.zeros(n, dtype=np.bool_)
    go_direct_present = np.zeros(n, dtype=np.bool_)
    sequence421_present = np.zeros(n, dtype=np.bool_)
    source_sequence_available = np.zeros(n, dtype=np.bool_)
    is_query = np.zeros(n, dtype=np.bool_)
    is_action = np.zeros(n, dtype=np.bool_)
    for index, entity_id in enumerate(entity_ids):
        if entity_id in esm:
            vector = esm[entity_id]
            if vector.shape != (320,):
                raise FeatureBuildError("invalid ESM vector dimension")
            values[index, :320] = vector
            values[index, 320] = np.float32(1.0)
            esm_present[index] = True
        if entity_id in go:
            vector = go[entity_id]
            if vector.shape != (256,):
                raise FeatureBuildError("invalid GO vector dimension")
            values[index, 321:] = vector
            go_identity_present[index] = True
            go_direct_present[index] = go_direct[entity_id]
        sequence421_present[index] = entity_id in sequence421_ids
        source_sequence_available[index] = entity_id in source_sequence_ids
        is_query[index] = entity_id in query_ids
        is_action[index] = entity_id in action_ids
    if not np.isfinite(values).all():
        raise FeatureBuildError("assembled features contain nonfinite values")
    return {
        "feature_values": values,
        "entity_taxon": np.full(n, TAXON, dtype=np.int64),
        "entity_id": np.asarray(entity_ids),
        "esm_present": esm_present,
        "go_identity_present": go_identity_present,
        "go_direct_annotation_present": go_direct_present,
        "sequence421_present": sequence421_present,
        "pinned_source_sequence_available": source_sequence_available,
        "is_strict_rna_query": is_query,
        "is_development_action": is_action,
    }


def lf_roster(ids: list[str]) -> bytes:
    return ("".join(f"{value}\n" for value in ids)).encode("ascii")


def build(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "rna_mapping": args.rna_mapping,
        "development": args.development,
        "esm": args.esm,
        "sequence421": args.sequence421,
        "shared_go": args.shared_go,
        "shared_go_manifest": args.shared_go_manifest,
        "current_orfs": args.current_orfs,
    }
    for key, path in paths.items():
        require_hash(path, EXPECTED[key])

    go_manifest = validate_shared_go_manifest(args.shared_go_manifest)
    queries = strict_query_ids(read_jsonl(args.rna_mapping))
    actions = read_action_ids_without_targets(args.development)
    orfs = current_orf_ids(read_jsonl(args.current_orfs))
    for key, actual in (
        ("strict_queries", len(queries)),
        ("actions", len(actions)),
        ("current_orfs", len(orfs)),
    ):
        if actual != EXPECTED_COUNTS[key]:
            raise FeatureBuildError(f"unexpected {key} count: {actual}")
    entities = sorted(set(queries) | set(actions))
    if len(entities) != EXPECTED_COUNTS["union"]:
        raise FeatureBuildError(f"unexpected union count: {len(entities)}")

    esm, _ = load_static_rows(args.esm, 320)
    sequence421, _ = load_static_rows(args.sequence421, 421)
    if set(esm) != set(sequence421):
        raise FeatureBuildError("ESM and sequence421 exact composite-key coverage differs")
    go, go_direct = load_static_rows(
        args.shared_go, 256, "direct_annotation_present"
    )
    arrays = assemble_features(
        entities,
        esm,
        go,
        go_direct,
        set(sequence421),
        orfs,
        set(queries),
        set(actions),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_NAME
    if output_path.exists() or (args.output_dir / "manifest.json").exists():
        raise FeatureBuildError("immutable output already exists")
    output_path.write_bytes(deterministic_npz(arrays))

    rosters: dict[str, list[str]] = {
        "strict-rna-query-ids.txt": queries,
        "development-action-ids.txt": actions,
        "missing-esm-ids.txt": [x for x in entities if x not in esm],
        "missing-go-identity-ids.txt": [x for x in entities if x not in go],
        "go-no-eligible-direct-annotation-ids.txt": [
            x for x in entities if x in go and not go_direct[x]
        ],
        "esm-missing-source-sequence-available-ids.txt": [
            x for x in entities if x not in esm and x in orfs
        ],
        "pinned-source-sequence-unavailable-ids.txt": [
            x for x in entities if x not in orfs
        ],
    }
    roster_manifest: dict[str, object] = {}
    for name, ids in rosters.items():
        payload = lf_roster(ids)
        (args.output_dir / name).write_bytes(payload)
        roster_manifest[name] = {
            "rows": len(ids),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def coverage(role_mask: np.ndarray) -> dict[str, int]:
        return {
            "rows": int(role_mask.sum()),
            "esmPresent": int((arrays["esm_present"] & role_mask).sum()),
            "esmMissing": int((~arrays["esm_present"] & role_mask).sum()),
            "sequence421Present": int(
                (arrays["sequence421_present"] & role_mask).sum()
            ),
            "pinnedSourceSequenceAvailable": int(
                (arrays["pinned_source_sequence_available"] & role_mask).sum()
            ),
            "esmMissingButPinnedSourceSequenceAvailable": int(
                (
                    ~arrays["esm_present"]
                    & arrays["pinned_source_sequence_available"]
                    & role_mask
                ).sum()
            ),
            "goIdentityPresent": int(
                (arrays["go_identity_present"] & role_mask).sum()
            ),
            "goDirectAnnotationPresent": int(
                (arrays["go_direct_annotation_present"] & role_mask).sum()
            ),
        }

    output_hash = sha256_file(output_path)
    manifest: dict[str, object] = {
        "schema": "slp.yeast-esm8m-shared-go-mf-cc-static-features/v1",
        "artifact": {
            "path": OUTPUT_NAME,
            "sha256": output_hash,
            "bytes": output_path.stat().st_size,
            "shape": [len(entities), 577],
            "dtype": "float32",
        },
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "taxon": TAXON,
            "namespace": "SGD CURIE",
            "ordering": "ascending taxon then codepoint entity ID",
            "rows": len(entities),
            "strictRnaQueries": len(queries),
            "developmentActions": len(actions),
            "actionOnlyRows": len(set(actions) - set(queries)),
        },
        "featureDefinition": {
            "dimension": 577,
            "columns": {
                "0:320": "frozen ESM-2 t6 8M full-protein vector, zero when absent",
                "320": "ESM exact-identity presence flag",
                "321:577": "unweighted direct MF/CC row projected in the frozen shared human/yeast GO basis, zero when identity or eligible annotation is absent",
            },
            "explicitAvailabilityArrays": [
                "esm_present",
                "go_identity_present",
                "go_direct_annotation_present",
                "sequence421_present",
                "pinned_source_sequence_available",
            ],
            "learnedGeneIdentity": False,
        },
        "coverage": {
            "union": coverage(np.ones(len(entities), dtype=np.bool_)),
            "strictRnaQueries": coverage(arrays["is_strict_rna_query"]),
            "developmentActions": coverage(arrays["is_development_action"]),
            "mappingClassCounts": dict(
                sorted(
                    Counter(
                        row["mappingClass"]
                        for row in read_jsonl(args.rna_mapping)
                        if row.get("mappingClass") in STRICT_MAPPING_CLASSES
                    ).items()
                )
            ),
        },
        "sharedGoCoordinateContract": {
            "basisSha256": go_manifest["artifacts"]["basis"]["sha256"],
            "componentsFloat32Sha256": go_manifest["compression"][
                "componentsFloat32Sha256"
            ],
            "projection": go_manifest["featureDefinition"]["projection"],
            "equalSpeciesTotalSquaredRowWeight": True,
            "crossSpeciesIdenticalAnnotationRowsVectorEqual": go_manifest["coverage"][
                "identicalRowInvariant"
            ]["exactVectorEquality"],
        },
        "inputs": {
            key: {"path": str(path).replace("\\", "/"), "sha256": EXPECTED[key]}
            for key, path in paths.items()
        },
        "rosters": roster_manifest,
        "accessBoundary": {
            "quantitativeOutcomesRead": False,
            "splitAssignmentsRead": False,
            "developmentArchiveMembersRead": ["action_ids.npy"],
            "staticFeaturesOnly": True,
        },
        "limitations": [
            "An ESM zero block denotes unavailable precomputed embedding; the pinned source may already contain a peptide that would require separately authorized inference.",
            "A zero shared-GO block may mean absent GO identity or no eligible direct MF/CC annotation; explicit arrays distinguish these states.",
            "The 421-dimensional deterministic sequence artifact is audited for coverage but is not concatenated into this 577-dimensional pack.",
            "Static annotation coverage provides no evidence of molecular forecast performance.",
        ],
    }
    (args.output_dir / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--rna-mapping", type=Path, required=True)
    result.add_argument("--development", type=Path, required=True)
    result.add_argument("--esm", type=Path, required=True)
    result.add_argument("--sequence421", type=Path, required=True)
    result.add_argument("--shared-go", type=Path, required=True)
    result.add_argument("--shared-go-manifest", type=Path, required=True)
    result.add_argument("--current-orfs", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
