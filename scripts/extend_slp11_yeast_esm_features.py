#!/usr/bin/env python3
"""Extend frozen yeast ESM features to current ORFs in a metadata roster.

Existing ESM vectors are copied bit-for-bit.  Only current ORFs with peptides
in the exact admitted SGD R64.5.1 FASTA are inferred.  The companion static
pack retains the shared-coordinate GO block and explicit missingness arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = importlib.util.spec_from_file_location(
    "_slp11_frozen_yeast_esm_recipe",
    ROOT / "scripts" / "build_slp11_sequence_features.py",
)
if SOURCE_SPEC is None or SOURCE_SPEC.loader is None:
    raise RuntimeError("cannot load frozen yeast ESM recipe")
SOURCE = importlib.util.module_from_spec(SOURCE_SPEC)
sys.modules[SOURCE_SPEC.name] = SOURCE
SOURCE_SPEC.loader.exec_module(SOURCE)

TAXON = 4932
EXPECTED = {
    "fasta": "17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b",
    "original_esm": "96f5e1b81036e0d42238ed6ac797f9fd399006f4d5f8227e96d9ee11358318ca",
    "base_static": "08570e20bd6c8839c1b17aa5c205ea8db9635a6515a44cb30e60cf7c79666d91",
    "base_manifest": "0425a90944e37cc0dc7e2d69b8952f2deb64e6d6f88f39aba2c597f300ee0438",
    "shared_go": "fb673cf6053bb7bfe88c6b454cedb662646f7256f094abf9a6df1d2865f873f6",
    "current_orfs": "df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d",
}
EXTENDED_ESM_NAME = "yeast-esm2-t6-8m-extended-features.npz"
STATIC_NAME = "yeast-static-esm8m-shared-go-mf-cc-features.npz"


class ExtensionError(ValueError):
    """Raised when an immutable extension contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ExtensionError(f"SHA-256 mismatch for {path}: {actual}")


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


def load_current_orfs(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        entity_id = row.get("canonicalSgdCurie")
        if (
            row.get("schema") != "slp.sgd-current-orf/v1"
            or row.get("ncbiTaxon") != TAXON
            or not isinstance(entity_id, str)
            or entity_id in result
        ):
            raise ExtensionError("invalid current ORF mapping")
        result.add(entity_id)
    if len(result) != 6613:
        raise ExtensionError(f"unexpected current ORF count: {len(result)}")
    return result


def load_lf_roster(path: Path, expected_sha256: str) -> list[str]:
    require_hash(path, expected_sha256)
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExtensionError("action roster is not ASCII") from exc
    if raw and not raw.endswith(b"\n"):
        raise ExtensionError("action roster must be LF terminated")
    ids = text.splitlines()
    if ids != sorted(set(ids)) or any(not item.startswith("SGD:S") for item in ids):
        raise ExtensionError("action roster must be sorted unique SGD CURIEs")
    return ids


def feature_map(
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
        raise ExtensionError("invalid feature artifact arrays")
    result: dict[tuple[int, str], np.ndarray] = {}
    previous: tuple[int, str] | None = None
    for index, (raw_taxon, raw_id) in enumerate(zip(taxon, ids, strict=True)):
        key = (int(raw_taxon), str(raw_id))
        if previous is not None and key <= previous:
            raise ExtensionError("feature keys are not uniquely sorted")
        previous = key
        result[key] = values[index]
    return result


def select_profile_ids(ids: list[str], peptides: dict[str, bytes], count: int) -> list[str]:
    if count < 1:
        raise ExtensionError("profile count must be positive")
    ordered = sorted(ids, key=lambda item: (len(peptides[item]), item))
    if count >= len(ordered):
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return [ordered[int(index)] for index in indices]


def extract(
    ids: list[str],
    peptide_by_id: dict[str, bytes],
    model_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    keys = [(TAXON, entity_id) for entity_id in ids]
    peptides = [peptide_by_id[entity_id] for entity_id in ids]
    return SOURCE._esm_feature_arrays(
        keys,
        peptides,
        model_dir,
        device_name=device,
        batch_size=batch_size,
        max_residues=SOURCE.ESM_MAX_RESIDUES,
        overlap=SOURCE.ESM_DEFAULT_OVERLAP,
    )


def prepare(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "fasta": args.fasta,
        "original_esm": args.original_esm,
        "base_static": args.base_static,
        "base_manifest": args.base_manifest,
        "shared_go": args.shared_go,
        "current_orfs": args.current_orfs,
    }
    for name, path in paths.items():
        require_hash(path, EXPECTED[name])
    SOURCE.verify_esm_model_dir(args.model_dir)
    sequences = SOURCE.parse_pinned_fasta(args.fasta)
    orfs = load_current_orfs(args.current_orfs)
    missing_fasta = sorted(orfs - set(sequences))
    if missing_fasta:
        raise ExtensionError(
            f"pinned FASTA is missing {len(missing_fasta)} current ORFs"
        )
    peptide_by_id = {entity_id: sequences[entity_id][:-1] for entity_id in orfs}

    base = load_npz(args.base_static)
    original = load_npz(args.original_esm)
    original_map = feature_map(original, 320)
    base_ids = [str(item) for item in base["entity_id"]]
    if base["feature_values"].shape != (len(base_ids), 577):
        raise ExtensionError("base static dimension mismatch")
    if not np.all(base["entity_taxon"] == TAXON):
        raise ExtensionError("base static pack is not yeast-only")
    actions = load_lf_roster(args.action_roster, args.action_roster_sha256)
    target_ids = sorted(set(base_ids) | set(actions))
    target_orfs = sorted(set(target_ids) & orfs)
    new_ids = sorted(
        entity_id for entity_id in target_orfs if (TAXON, entity_id) not in original_map
    )
    existing_concordance_candidates = sorted(
        entity_id for entity_id in target_orfs if (TAXON, entity_id) in original_map
    )
    return {
        "paths": paths,
        "sequences": sequences,
        "orfs": orfs,
        "peptide_by_id": peptide_by_id,
        "base": base,
        "original": original,
        "original_map": original_map,
        "actions": actions,
        "target_ids": target_ids,
        "target_orfs": target_orfs,
        "new_ids": new_ids,
        "concordance_candidates": existing_concordance_candidates,
    }


def run_profile(args: argparse.Namespace, prepared: dict[str, object]) -> dict[str, object]:
    new_ids = prepared["new_ids"]
    candidates = prepared["concordance_candidates"]
    peptides = prepared["peptide_by_id"]
    original_map = prepared["original_map"]
    assert isinstance(new_ids, list) and isinstance(candidates, list)
    assert isinstance(peptides, dict) and isinstance(original_map, dict)
    profile_ids = select_profile_ids(new_ids, peptides, args.profile_count)
    concordance_ids = select_profile_ids(candidates, peptides, args.concordance_count)
    selected = sorted(set(profile_ids) | set(concordance_ids))
    started = time.monotonic()
    extracted, statistics = extract(
        selected, peptides, args.model_dir, args.device, args.batch_size
    )
    elapsed = time.monotonic() - started
    extracted_map = feature_map(extracted, 320)
    differences = np.stack(
        [
            extracted_map[(TAXON, entity_id)]
            - original_map[(TAXON, entity_id)]
            for entity_id in concordance_ids
        ]
    )
    max_abs = float(np.abs(differences).max())
    rms = float(np.sqrt(np.mean(np.square(differences.astype(np.float64)))))
    if max_abs > args.concordance_atol:
        raise ExtensionError(
            f"existing-vector concordance failed: max_abs={max_abs:.9g}"
        )
    total_windows = sum(
        len(SOURCE.chunk_windows(len(peptides[entity_id]))) for entity_id in new_ids
    )
    sample_windows = sum(
        len(SOURCE.chunk_windows(len(peptides[entity_id]))) for entity_id in profile_ids
    )
    projected_seconds = elapsed * total_windows / max(sample_windows, 1)
    report = {
        "schema": "slp.yeast-esm2-t6-extension-profile/v1",
        "newEntities": len(new_ids),
        "profileEntities": len(profile_ids),
        "concordanceEntities": len(concordance_ids),
        "totalWindows": total_windows,
        "profileWindows": sample_windows,
        "elapsedSeconds": elapsed,
        "projectedFullSecondsByWindowCount": projected_seconds,
        "concordance": {
            "maximumAbsoluteDifference": max_abs,
            "rmsDifference": rms,
            "absoluteTolerance": args.concordance_atol,
        },
        "extractorStatistics": statistics,
        "frozenRecipe": {
            "modelRevision": SOURCE.ESM_REVISION,
            "finalLayer": 6,
            "hiddenSize": 320,
            "maxResiduesPerWindow": SOURCE.ESM_MAX_RESIDUES,
            "overlapResidues": SOURCE.ESM_DEFAULT_OVERLAP,
            "pooling": "inverse-overlap-corrected residue representations, then uniform full-protein mean",
            "truncation": False,
            "batchSize": args.batch_size,
            "float32ModelInference": True,
            "float64OverlapAccumulation": True,
        },
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_bytes(canonical_json(report))
    return report


def extend_esm(
    original: dict[str, np.ndarray], extracted: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    old_map = feature_map(original, 320)
    new_map = feature_map(extracted, 320)
    overlap = set(old_map) & set(new_map)
    if overlap:
        raise ExtensionError("new ESM extraction overlaps original identities")
    keys = sorted(set(old_map) | set(new_map))
    values = np.stack(
        [old_map.get(key, new_map.get(key)) for key in keys]
    ).astype(np.float32, copy=False)
    return {
        "feature_values": values,
        "entity_taxon": np.asarray([key[0] for key in keys], dtype=np.int64),
        "entity_id": np.asarray([key[1] for key in keys]),
    }


def build_static(
    base: dict[str, np.ndarray],
    target_ids: list[str],
    actions: set[str],
    extended_esm: dict[str, np.ndarray],
    go_path: Path,
    source_orfs: set[str],
) -> dict[str, np.ndarray]:
    esm_map = feature_map(extended_esm, 320)
    go_arrays = load_npz(go_path)
    go_map = feature_map(go_arrays, 256)
    go_direct_map = {
        (int(taxon), str(entity_id)): bool(present)
        for taxon, entity_id, present in zip(
            go_arrays["entity_taxon"],
            go_arrays["entity_id"],
            go_arrays["direct_annotation_present"],
            strict=True,
        )
    }
    base_index = {str(entity_id): index for index, entity_id in enumerate(base["entity_id"])}
    n = len(target_ids)
    values = np.zeros((n, 577), dtype=np.float32)
    esm_present = np.zeros(n, dtype=np.bool_)
    go_identity = np.zeros(n, dtype=np.bool_)
    go_direct = np.zeros(n, dtype=np.bool_)
    source_present = np.zeros(n, dtype=np.bool_)
    is_query = np.zeros(n, dtype=np.bool_)
    is_old_action = np.zeros(n, dtype=np.bool_)
    is_full_action = np.zeros(n, dtype=np.bool_)
    for out_index, entity_id in enumerate(target_ids):
        old_index = base_index.get(entity_id)
        if old_index is not None:
            values[out_index] = base["feature_values"][old_index]
            is_query[out_index] = base["is_strict_rna_query"][old_index]
            is_old_action[out_index] = base["is_development_action"][old_index]
        esm_vector = esm_map.get((TAXON, entity_id))
        if esm_vector is not None:
            values[out_index, :320] = esm_vector
            values[out_index, 320] = np.float32(1.0)
            esm_present[out_index] = True
        go_vector = go_map.get((TAXON, entity_id))
        if go_vector is not None:
            values[out_index, 321:] = go_vector
            go_identity[out_index] = True
            go_direct[out_index] = go_direct_map[(TAXON, entity_id)]
        source_present[out_index] = entity_id in source_orfs
        is_full_action[out_index] = entity_id in actions
    return {
        "feature_values": values,
        "entity_taxon": np.full(n, TAXON, dtype=np.int64),
        "entity_id": np.asarray(target_ids),
        "esm_present": esm_present,
        "go_identity_present": go_identity,
        "go_direct_annotation_present": go_direct,
        "pinned_source_sequence_available": source_present,
        "is_strict_rna_query": is_query,
        "is_original_fc_development_action": is_old_action,
        "is_full_raw_development_action": is_full_action,
    }


def run_build(args: argparse.Namespace, prepared: dict[str, object]) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in (EXTENDED_ESM_NAME, STATIC_NAME, "manifest.json")):
        raise ExtensionError("immutable output already exists")
    new_ids = prepared["new_ids"]
    peptides = prepared["peptide_by_id"]
    assert isinstance(new_ids, list) and isinstance(peptides, dict)
    started = time.monotonic()
    extracted, statistics = extract(
        new_ids, peptides, args.model_dir, args.device, args.batch_size
    )
    inference_seconds = time.monotonic() - started
    extended = extend_esm(prepared["original"], extracted)
    static = build_static(
        prepared["base"],
        prepared["target_ids"],
        set(prepared["actions"]),
        extended,
        args.shared_go,
        prepared["orfs"],
    )
    esm_path = output_dir / EXTENDED_ESM_NAME
    static_path = output_dir / STATIC_NAME
    esm_path.write_bytes(deterministic_npz(extended))
    static_path.write_bytes(deterministic_npz(static))

    old_map = feature_map(prepared["original"], 320)
    extended_map = feature_map(extended, 320)
    if any(
        not np.array_equal(vector, extended_map[key])
        for key, vector in old_map.items()
    ):
        raise ExtensionError("an original ESM vector changed")
    base = prepared["base"]
    base_ids = [str(item) for item in base["entity_id"]]
    static_index = {str(item): i for i, item in enumerate(static["entity_id"])}
    old_present_preserved = all(
        np.array_equal(
            base["feature_values"][i, :320],
            static["feature_values"][static_index[entity_id], :320],
        )
        for i, entity_id in enumerate(base_ids)
        if base["esm_present"][i]
    )
    old_go_preserved = all(
        np.array_equal(
            base["feature_values"][i, 321:],
            static["feature_values"][static_index[entity_id], 321:],
        )
        for i, entity_id in enumerate(base_ids)
    )
    if not old_present_preserved or not old_go_preserved:
        raise ExtensionError("an existing static feature block changed")

    missing_ids = [
        str(entity_id)
        for entity_id, present in zip(
            static["entity_id"], static["esm_present"], strict=True
        )
        if not present
    ]
    missing_payload = "".join(f"{item}\n" for item in missing_ids).encode("ascii")
    (output_dir / "missing-protein-feature-ids.txt").write_bytes(missing_payload)
    manifest = {
        "schema": "slp.yeast-esm2-t6-shared-go-static-extension/v2",
        "artifacts": {
            "extendedSequence": {
                "path": EXTENDED_ESM_NAME,
                "sha256": sha256_file(esm_path),
                "bytes": esm_path.stat().st_size,
                "shape": list(extended["feature_values"].shape),
            },
            "static577": {
                "path": STATIC_NAME,
                "sha256": sha256_file(static_path),
                "bytes": static_path.stat().st_size,
                "shape": list(static["feature_values"].shape),
            },
            "missingProteinRoster": {
                "path": "missing-protein-feature-ids.txt",
                "sha256": hashlib.sha256(missing_payload).hexdigest(),
                "rows": len(missing_ids),
            },
        },
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "taxon": TAXON,
            "ordering": "ascending taxon then codepoint entity ID",
            "staticRows": len(static["entity_id"]),
            "actionRows": len(prepared["actions"]),
        },
        "coverage": {
            "newEsmVectors": len(new_ids),
            "staticEsmPresent": int(static["esm_present"].sum()),
            "staticEsmMissing": int((~static["esm_present"]).sum()),
            "staticGoIdentityPresent": int(static["go_identity_present"].sum()),
            "staticGoDirectAnnotationPresent": int(
                static["go_direct_annotation_present"].sum()
            ),
            "staticPinnedSourceSequenceAvailable": int(
                static["pinned_source_sequence_available"].sum()
            ),
        },
        "preservation": {
            "allOriginalEsmVectorsBitExact": True,
            "allOriginalPresentStaticEsmVectorsBitExact": old_present_preserved,
            "allOriginalStaticGoVectorsBitExact": old_go_preserved,
        },
        "frozenRecipe": {
            "model": SOURCE.ESM_REPOSITORY,
            "revision": SOURCE.ESM_REVISION,
            "finalLayer": 6,
            "hiddenSize": 320,
            "maxResiduesPerWindow": SOURCE.ESM_MAX_RESIDUES,
            "overlapResidues": SOURCE.ESM_DEFAULT_OVERLAP,
            "pooling": "inverse-overlap-corrected residue representations, then uniform full-protein mean",
            "truncation": False,
            "batchSize": args.batch_size,
            "float32ModelInference": True,
            "float64OverlapAccumulation": True,
        },
        "runtime": {"inferenceSeconds": inference_seconds, **statistics},
        "inputs": {
            **{
                name: {"path": str(path).replace("\\", "/"), "sha256": EXPECTED[name]}
                for name, path in prepared["paths"].items()
            },
            "actionRoster": {
                "path": str(args.action_roster).replace("\\", "/"),
                "sha256": args.action_roster_sha256,
            },
            "modelFiles": SOURCE.verify_esm_model_dir(args.model_dir),
        },
        "accessBoundary": {
            "staticSequencesRead": True,
            "staticAnnotationsRead": True,
            "quantitativeOutcomesRead": False,
            "splitAssignmentsRead": False,
        },
        "limitations": [
            "Protein-missing current non-ORF features remain explicit zero ESM blocks.",
            "ESM vectors are externally pretrained static descriptors and establish no molecular forecast benefit.",
            "The output uses stable SGD identity only and contains no learned gene-ID embedding.",
        ],
    }
    (output_dir / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("profile", "build"), required=True)
    result.add_argument("--fasta", type=Path, required=True)
    result.add_argument("--original-esm", type=Path, required=True)
    result.add_argument("--base-static", type=Path, required=True)
    result.add_argument("--base-manifest", type=Path, required=True)
    result.add_argument("--shared-go", type=Path, required=True)
    result.add_argument("--current-orfs", type=Path, required=True)
    result.add_argument("--action-roster", type=Path, required=True)
    result.add_argument("--action-roster-sha256", required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--profile-count", type=int, default=64)
    result.add_argument("--concordance-count", type=int, default=8)
    result.add_argument("--concordance-atol", type=float, default=2e-5)
    result.add_argument("--profile-output", type=Path)
    result.add_argument("--output-dir", type=Path)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.mode == "profile" and parsed.profile_output is None:
        raise SystemExit("--profile-output is required for profile mode")
    if parsed.mode == "build" and parsed.output_dir is None:
        raise SystemExit("--output-dir is required for build mode")
    state = prepare(parsed)
    report = run_profile(parsed, state) if parsed.mode == "profile" else run_build(parsed, state)
    print(json.dumps(report, indent=2, sort_keys=True))
