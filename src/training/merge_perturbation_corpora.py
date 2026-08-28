"""Merge canonical perturbation packs without fitting across held-out studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    ROOT / "results/data/gse220974-crisprai-pseudobulk-v1/gse220974_crisprai_pseudobulk_v1.npz",
    ROOT / "results/data/gse221321-random-composite-cells-v1/gse221321_random_composite_cells_v1.npz",
    ROOT / "results/data/gse337988-dld1-multi-action-pseudobulk-v1/gse337988_dld1_multi_action_pseudobulk_v1.npz",
    ROOT / "results/data/gse213957-thp1-carpool-pseudobulk-v1/gse213957_thp1_carpool_pseudobulk_v1.npz",
    ROOT / "results/data/gse200201-molm13-multi-action-pseudobulk-v1/gse200201_molm13_multi_action_pseudobulk_v1.npz",
    ROOT / "results/data/gse208240-calu3-crispri-pseudobulk-v1/gse208240_calu3_crispri_pseudobulk_v1.npz",
    ROOT / "results/data/gse278572-primary-tcell-perturbseq-v1/gse278572_primary_tcell_perturbseq_v1.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(values: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), count)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    return array.astype(str)


def load(path: Path) -> dict[str, np.ndarray]:
    required = {
        "actions",
        "action_modes",
        "action_doses",
        "action_names",
        "target",
        "target_semantics",
        "target_feature_name",
        "source_id",
        "context_id",
        "experimental_condition_id",
    }
    with np.load(path, allow_pickle=False) as pack:
        missing = required - set(pack.files)
        if missing:
            raise ValueError(f"{path} lacks canonical fields: {sorted(missing)}")
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    if str(arrays["target_semantics"].reshape(-1)[0]) != "perturbation_delta":
        raise ValueError(f"{path} is not a perturbation-delta pack")
    count = len(arrays["actions"])
    for field in ("source_id", "context_id", "experimental_condition_id"):
        arrays[field] = rows(arrays[field], count, field)
    if "observation_unit" not in arrays:
        arrays["observation_unit"] = np.repeat("pseudobulk", count)
    else:
        arrays["observation_unit"] = rows(arrays["observation_unit"], count, "observation_unit")
    return arrays


def build(inputs: list[Path], output: Path, min_features: int = 64) -> dict[str, object]:
    inputs = [Path(path) for path in inputs]
    loaded = [load(path) for path in inputs]
    common_features = set(loaded[0]["target_feature_name"].astype(str))
    for pack in loaded[1:]:
        common_features &= set(pack["target_feature_name"].astype(str))
    feature_names = np.asarray(sorted(common_features))
    if len(feature_names) < min_features:
        raise ValueError(
            f"only {len(feature_names)} expression features are shared; require {min_features}"
        )
    action_names = np.asarray(
        sorted(set().union(*(set(pack["action_names"].astype(str)) for pack in loaded)))
    )
    action_id = {name: index for index, name in enumerate(action_names)}
    slots = max(pack["actions"].shape[1] for pack in loaded)
    merged: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "actions",
            "action_modes",
            "action_doses",
            "target",
            "source_id",
            "context_id",
            "experimental_condition_id",
            "observation_unit",
        )
    }
    sources = []
    for path, pack in zip(inputs, loaded):
        count = len(pack["actions"])
        actions = np.full((count, slots), -1, dtype="int32")
        valid = pack["actions"] >= 0
        local_names = pack["action_names"].astype(str)
        remapped = np.full(pack["actions"].shape, -1, dtype="int32")
        remapped[valid] = np.asarray(
            [action_id[name] for name in local_names[pack["actions"][valid]]], dtype="int32"
        )
        actions[:, : remapped.shape[1]] = remapped
        modes = np.full((count, slots), "", dtype="<U16")
        modes[:, : pack["action_modes"].shape[1]] = pack["action_modes"].astype(str)
        doses = np.zeros((count, slots), dtype="float32")
        doses[:, : pack["action_doses"].shape[1]] = pack["action_doses"].astype("float32")
        feature_id = {
            name: index for index, name in enumerate(pack["target_feature_name"].astype(str))
        }
        columns = np.asarray([feature_id[name] for name in feature_names])
        merged["actions"].append(actions)
        merged["action_modes"].append(modes)
        merged["action_doses"].append(doses)
        merged["target"].append(pack["target"][:, columns].astype("float32"))
        for field in (
            "source_id",
            "context_id",
            "experimental_condition_id",
            "observation_unit",
        ):
            merged[field].append(pack[field])
        sources.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "rows": count,
                "source_ids": sorted(np.unique(pack["source_id"]).tolist()),
                "context_ids": sorted(np.unique(pack["context_id"]).tolist()),
                "observation_units": sorted(np.unique(pack["observation_unit"]).tolist()),
            }
        )
    arrays = {name: np.concatenate(parts) for name, parts in merged.items()}
    source_counts = {
        source: int((arrays["source_id"] == source).sum())
        for source in np.unique(arrays["source_id"])
    }
    arrays["study_balance_weight"] = np.asarray(
        [1.0 / source_counts[source] for source in arrays["source_id"]], dtype="float64"
    )
    arrays["study_balance_weight"] /= arrays["study_balance_weight"].sum()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "public_multi_study_perturbation_atlas_v1.npz"
    np.savez_compressed(
        artifact,
        **arrays,
        action_names=action_names,
        target_semantics=np.asarray("perturbation_delta"),
        target_feature_name=feature_names,
        cardinality=(arrays["actions"] >= 0).sum(axis=1).astype("int8"),
    )
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbation-atlas/public-multi-study-v1",
        "sources": sources,
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "Each constituent release carries a source-specific redistribution audit and checksum.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Canonical source packs were remapped to one action vocabulary and restricted to the exact intersection of measured expression features. Source-native perturbation deltas were not pooled, standardized, projected, imputed, or fitted across studies. A normalized inverse-source-row weight is provided for study-balanced sampling.",
        "schema_description": "NPZ with padded target/mode/dose action sets, shared raw perturbation-delta expression endpoints, independent study/context/condition identifiers, observation units, and study-balancing weights.",
        "population": "Union of the human cell populations documented by each constituent release",
        "endpoints": ["source-normalized expression perturbation deltas over a shared measured gene panel"],
        "split_construction": "No split is embedded. Deterministic hard molecular folds are constructed by validate_generalization.py.",
        "exclusions": "Expression features absent from any source were excluded. No rows or genes were excluded using an SL benchmark.",
        "rows": len(arrays["actions"]),
        "source_rows": source_counts,
        "unique_action_targets": len(action_names),
        "expression_features": len(feature_names),
        "maximum_cardinality": int((arrays["actions"] >= 0).sum(axis=1).max()),
        "sl_labels_used": False,
        "files": [
            {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/public-multi-study-perturbation-atlas-v1",
    )
    parser.add_argument("--min-features", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(args.inputs, args.output, args.min_features), indent=2))


if __name__ == "__main__":
    main()
