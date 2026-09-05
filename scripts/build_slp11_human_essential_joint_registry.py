"""Build an outcome-free K562/RPE1 count-training registry and static pack."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/derived/slp11-human-essential-joint-training-registry-v1"

PATHS = {
    "k562_static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
    "k562_roster": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
    "k562_raw_manifest": ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2/manifest.json",
    "k562_mmap_manifest": ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1/manifest.json",
    "k562_counts": ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1/counts.uint16",
    "k562_rows": ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1/rows.npz",
    "k562_fit_manifest": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/manifest.json",
    "k562_fit_moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
    "k562_control_moments": ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2/control-gem-moments.npz",
    "k562_reference": ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1/reference.npz",
    "rpe1_static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
    "rpe1_roster": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
    "rpe1_raw_manifest": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/manifest.json",
    "rpe1_counts": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/reconstruction-train-counts.uint16",
    "rpe1_rows": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/reconstruction-train-row-metadata.npz",
    "rpe1_fit_moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
    "rpe1_control_moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/control-gem-moments.npz",
}

EXPECTED = {
    "k562_static": "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    "k562_roster": "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b",
    "k562_raw_manifest": "859b3fb0b0aeb830e25dce17e86edfc2d8ec3fcdbcec57beeeebf6d1a8faf685",
    "k562_mmap_manifest": "4be7a48848cb5c96d32e4da0097af23a27ca4bb3405c31d0e5a6569fe5c1c49d",
    "k562_counts": "e9bbfe69bd59cedf7131bd176632bb9fbd8dce59a0789ed7e18896ac34e4b511",
    "k562_rows": "5d8631e50b3dcabc9448eaa112eb94bc1335967e5b9098b6e278b6340a9a226b",
    "k562_fit_manifest": "68f87bf258307a9d9503407af61e7da0a22e89b65eb7172cbe10d8b6b4956a24",
    "k562_fit_moments": "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    "k562_control_moments": "51f4b53f1e24df5299e39c7d3354784c5da0cc7cd00995630d618f824e1c25c2",
    "k562_reference": "8020753e9e2597b08cb94c5351772be05986b286f61e0f7a26be26fbfabae4f6",
    "rpe1_static": "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    "rpe1_roster": "b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56",
    "rpe1_raw_manifest": "3d7ca31f945ffb193070eb463eaa328e374c9f12f3c0e3162a5e189f24d0fe9e",
    "rpe1_counts": "6df95d35bd725dd935e368859391a99fc7e82f2019b1700eabfc744c01481ba6",
    "rpe1_rows": "b7b035798415ce2bc55361b12a52d13739cb2555621456342f75cf1e7a15339a",
    "rpe1_fit_moments": "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    "rpe1_control_moments": "5aceba5fb4874811aac797be14d1947a9fca866d11178d5f8fe2bdc534df6f61",
}

CONTEXTS = {
    "k562": "replogle-2022-k562-essential-day-6",
    "rpe1": "replogle-2022-rpe1-essential-day-7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def validate_role_contract(rosters: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for source in sorted(rosters):
        ids = np.asarray(rosters[source]["action_ids"]).astype(str)
        source_roles = np.asarray(rosters[source]["action_role"]).astype(str)
        if len(ids) != len(set(ids)) or len(ids) != len(source_roles):
            raise ValueError(f"invalid action roster: {source}")
        for gene, role in zip(ids, source_roles):
            prior = roles.get(gene)
            if prior is not None and prior != role:
                raise ValueError(f"global action-role conflict for {gene}: {prior} != {role}")
            roles[gene] = role
    return roles


def make_context_ids(source: str, base_context: str, gem_groups: np.ndarray) -> np.ndarray:
    gem = np.asarray(gem_groups)
    if gem.ndim != 1 or len(gem) != len(np.unique(gem)):
        raise ValueError(f"invalid GEM roster: {source}")
    result = np.asarray([f"{base_context}::gem-group:{int(value):03d}" for value in gem])
    if len(result) != len(set(result)):
        raise ValueError(f"context collision within {source}")
    return result


def merge_static(
    statics: Mapping[str, Mapping[str, np.ndarray]], fitting_ids: np.ndarray, clip: float
) -> tuple[dict[str, np.ndarray], dict[str, int | bool]]:
    entity_ids = np.asarray(sorted(set().union(*[
        set(np.asarray(item["entity_id"]).astype(str)) for item in statics.values()
    ])))
    merged = np.zeros((len(entity_ids), 577), dtype=np.float32)
    present = {source: np.zeros(len(entity_ids), dtype=bool) for source in statics}
    availability_names = (
        "source_static_row_present", "esm_present", "go_direct_annotation_present",
        "go_exact_uniprot_mapping_present", "is_ensembl116_translated_gene",
    )
    availability = {name: np.zeros(len(entity_ids), dtype=bool) for name in availability_names}
    lookup = {gene: row for row, gene in enumerate(entity_ids)}
    overlap_checked = 0
    first_values: dict[str, np.ndarray] = {}
    for source in sorted(statics):
        item = statics[source]
        ids = np.asarray(item["entity_id"]).astype(str)
        values = np.asarray(item["feature_values"], dtype=np.float32)
        if values.shape != (len(ids), 577) or len(ids) != len(set(ids)):
            raise ValueError(f"static feature schema drift: {source}")
        if not np.all(np.asarray(item["entity_taxon"]) == 9606):
            raise ValueError(f"taxonomy drift: {source}")
        for local, gene in enumerate(ids):
            target = lookup[gene]
            if gene in first_values:
                overlap_checked += 1
                if not np.array_equal(first_values[gene], values[local]):
                    raise ValueError(f"overlapping static row differs: {gene}")
            else:
                first_values[gene] = values[local].copy()
                merged[target] = values[local]
            present[source][target] = True
            for name in availability_names:
                if name in item:
                    availability[name][target] |= bool(item[name][local])
    fitting = np.asarray(fitting_ids).astype(str)
    if len(fitting) != len(set(fitting)) or not set(fitting).issubset(lookup):
        raise ValueError("invalid fitting-action union")
    fit_rows = np.asarray([lookup[gene] for gene in fitting], dtype=np.int64)
    fit_values = merged[fit_rows].astype(np.float64)
    mean = fit_values.mean(axis=0, dtype=np.float64)
    sd = fit_values.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(sd <= 1e-5, 1.0, sd)
    normalized = np.clip((merged.astype(np.float64) - mean) / scale, -clip, clip).astype(np.float32)
    arrays = {
        "schema": np.asarray("slp.human-essential-joint-static577/v1"),
        "entity_id": entity_ids,
        "entity_taxon": np.full(len(entity_ids), 9606, dtype=np.int64),
        "feature_values": merged,
        "normalized_feature_values": normalized,
        "feature_mean": mean,
        "feature_sd": sd,
        "feature_scale": scale,
        "feature_clip": np.asarray(clip, dtype=np.float32),
        "normalizer_fitting_action_ids": fitting,
        "normalizer_fitting_entity_index": fit_rows,
        "present_k562_static_pack": present["k562"],
        "present_rpe1_static_pack": present["rpe1"],
        **availability,
    }
    audit = {
        "entities": len(entity_ids),
        "overlapRowsBitExact": overlap_checked,
        "fittingActions": len(fitting),
        "allZeroRows": int(np.sum(np.all(merged == 0, axis=1))),
        "allZeroFittingActions": int(np.sum(np.all(merged[fit_rows] == 0, axis=1))),
        "constantOrNearConstantColumns": int(np.sum(sd <= 1e-5)),
        "finiteNormalized": bool(np.isfinite(normalized).all()),
    }
    return arrays, audit


def source_index(
    source: str,
    roster: Mapping[str, np.ndarray],
    union_ids: np.ndarray,
    query_ids: np.ndarray,
    gem_groups: np.ndarray,
) -> dict[str, np.ndarray]:
    union_lookup = {gene: row for row, gene in enumerate(union_ids.astype(str))}
    roster_query = np.asarray(roster["query_ids"]).astype(str)
    if not np.array_equal(roster_query, np.asarray(query_ids).astype(str)):
        raise ValueError(f"source-native query order mismatch: {source}")
    action = np.asarray(roster["action_ids"]).astype(str)
    fitting = np.asarray(roster["fitting_action_ids"]).astype(str)
    return {
        "schema": np.asarray("slp.human-essential-joint-source-index/v1"),
        "source_id": np.asarray(source),
        "base_context_id": np.asarray(CONTEXTS[source]),
        "query_ids": roster_query,
        "query_entity_index": np.asarray([union_lookup[x] for x in roster_query], np.int64),
        "action_ids": action,
        "action_entity_index": np.asarray([union_lookup[x] for x in action], np.int64),
        "action_role": np.asarray(roster["action_role"]).astype(str),
        "fitting_action_ids": fitting,
        "fitting_action_entity_index": np.asarray([union_lookup[x] for x in fitting], np.int64),
        "gem_group": np.asarray(gem_groups),
        "context_ids": make_context_ids(source, CONTEXTS[source], gem_groups),
        "full_native_library_query_count": np.asarray(len(roster_query), np.int64),
        "library_denominator_definition": np.asarray(
            f"sum raw UMI counts across exact ordered {len(roster_query)}-query {source} source panel"
        ),
    }


def load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = set(names) - set(source.files)
        if missing:
            raise ValueError(f"missing arrays in {path.name}: {sorted(missing)}")
        return {name: source[name] for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable output already exists")
    for name, path in PATHS.items():
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"input checksum mismatch: {name}")

    static_names = (
        "entity_id", "entity_taxon", "feature_values", "source_static_row_present",
        "esm_present", "go_direct_annotation_present", "go_exact_uniprot_mapping_present",
        "is_ensembl116_translated_gene",
    )
    roster_names = (
        "query_ids", "action_ids", "action_role", "fitting_action_ids",
    )
    statics = {source: load_npz(PATHS[f"{source}_static"], static_names) for source in ("k562", "rpe1")}
    rosters = {source: load_npz(PATHS[f"{source}_roster"], roster_names) for source in ("k562", "rpe1")}
    roles = validate_role_contract(rosters)
    kfit = set(rosters["k562"]["fitting_action_ids"].astype(str))
    rfit = set(rosters["rpe1"]["fitting_action_ids"].astype(str))
    if not kfit < rfit or len(kfit) != 1443 or len(rfit) != 1666:
        raise ValueError("expected strict K562 fitting-action subset of RPE1")
    with np.load(PATHS["k562_reference"], allow_pickle=False) as reference:
        clip = float(reference["feature_clip"])
    if clip != float(np.finfo(np.float32).max):
        raise ValueError("original K562 feature clip drift")
    static_arrays, static_audit = merge_static(statics, np.asarray(sorted(rfit)), clip)

    moment_names = ("query_ids", "gem_group")
    controls = {
        source: load_npz(PATHS[f"{source}_control_moments"], moment_names)
        for source in ("k562", "rpe1")
    }
    fit_queries = {
        source: load_npz(PATHS[f"{source}_fit_moments"], ("query_ids",))["query_ids"]
        for source in ("k562", "rpe1")
    }
    indices = {}
    all_contexts: list[str] = []
    for source in ("k562", "rpe1"):
        if not np.array_equal(controls[source]["query_ids"], fit_queries[source]):
            raise ValueError(f"control/fitting query order mismatch: {source}")
        indices[source] = source_index(
            source, rosters[source], static_arrays["entity_id"], fit_queries[source],
            controls[source]["gem_group"],
        )
        all_contexts.extend(indices[source]["context_ids"].astype(str).tolist())
    if len(all_contexts) != len(set(all_contexts)):
        raise ValueError("cross-source context collision")

    args.output.mkdir(parents=True)
    static_path = args.output / "shared-static577.npz"
    deterministic_npz(static_path, static_arrays)
    index_artifacts = {}
    for source in ("k562", "rpe1"):
        path = args.output / f"{source}-index.npz"
        deterministic_npz(path, indices[source])
        index_artifacts[source] = {
            "path": path.name, "sha256": sha256(path),
            "queries": len(indices[source]["query_ids"]),
            "actions": len(indices[source]["action_ids"]),
            "fittingActions": len(indices[source]["fitting_action_ids"]),
            "gemContexts": len(indices[source]["context_ids"]),
        }

    registry = {
        "schema": "slp.human-essential-joint-training-registry/v1",
        "status": "outcome-free-training-registry",
        "taxonomy": 9606,
        "noJointTargetMatrix": True,
        "developmentOrTestCountsAccessed": False,
        "normalization": {
            "fittingActions": 1666,
            "definition": "float64 population mean/SD over unique union fitting actions; scale=1 when SD<=1e-5; normalize raw float32 through float64 then cast float32",
            "featureClip": clip,
            "featureClipSource": {"path": str(PATHS["k562_reference"].relative_to(ROOT)).replace("\\", "/"), "sha256": EXPECTED["k562_reference"]},
        },
        "static": {"path": static_path.name, "sha256": sha256(static_path), "audit": static_audit},
        "indices": index_artifacts,
        "globalRoles": {"genes": len(roles), "overlapConflicts": 0, "k562FittingStrictSubsetOfRpe1": True},
        "sources": {
            "k562": {
                "contextBase": CONTEXTS["k562"], "nativeQueries": 8563,
                "trainingRows": {"fit": 188195, "control": 9609},
                "reconstructionHeldRowsReferenced": 21900,
                "developmentValidationRowsReferencedOnly": 47914,
                "artifacts": ["k562_raw_manifest", "k562_mmap_manifest", "k562_counts", "k562_rows", "k562_fit_manifest", "k562_fit_moments", "k562_control_moments", "k562_static", "k562_roster"],
            },
            "rpe1": {
                "contextBase": CONTEXTS["rpe1"], "nativeQueries": 8749,
                "trainingRows": {"fit": 142601, "control": 10350},
                "reconstructionHeldRowsReferenced": 17072,
                "developmentValidationRowsAccessed": 0,
                "artifacts": ["rpe1_raw_manifest", "rpe1_counts", "rpe1_rows", "rpe1_fit_moments", "rpe1_control_moments", "rpe1_static", "rpe1_roster"],
            },
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": EXPECTED[name], "bytes": path.stat().st_size,
            }
            for name, path in PATHS.items()
        },
        "limitations": [
            "K562 and RPE1 retain separate query axes and full native library denominators.",
            "The registry supplies identities, features, and paths; it does not merge count outcomes.",
            "Source and GEM are both encoded in context_ids; numeric GEM labels are not shared context identities.",
        ],
    }
    registry_path = args.output / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"registrySha256": sha256(registry_path), "staticSha256": sha256(static_path), "indices": index_artifacts}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"registry": receipt, "staticAudit": static_audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
