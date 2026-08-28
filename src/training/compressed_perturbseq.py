"""Clean the GSE221321 random-composite Perturb-seq cells for world training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
CONTROL_TARGETS = {"non-targeting", "safe-targeting"}
SOURCES = (
    ("GSM6858448_KO_cell_pooled.h5ad", "knockout", "KO_cell_pooled"),
    ("GSM6858450_KD_guide_pooled.h5ad", "repression", "KD_guide_pooled"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(dataset) -> np.ndarray:
    return np.asarray(dataset).astype(str)


def members(condition: str) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        target.upper()
        for target in condition.split("--")
        if target.lower() not in CONTROL_TARGETS
    )
    return tuple(sorted(counts.items()))


def inspect(path: Path):
    with h5py.File(path) as handle:
        condition = strings(handle["obs/Guides_collapsed_by_gene"])
        features = strings(handle["var/_index"])
    action_names = {target for value in condition for target, _ in members(value)}
    return condition, features, action_names


def load_source(
    path: Path,
    mode: str,
    condition_name: str,
    action_id: dict[str, int],
    target_features: np.ndarray,
):
    with h5py.File(path) as handle:
        obs = handle["obs"]
        condition = strings(obs["Guides_collapsed_by_gene"])
        parsed = [members(value) for value in condition]
        channels = strings(obs["10X_channel"])
        barcodes = strings(obs["_index"])
        guide_load = np.asarray(obs["Total_number_of_guides"], dtype="float32")
        feature_names = strings(handle["var/_index"])
        feature_id = {name: index for index, name in enumerate(feature_names)}
        columns = np.asarray([feature_id[name] for name in target_features])
        shape = tuple(handle["X"].attrs["shape"])
        expression = sparse.csr_matrix(
            (
                np.asarray(handle["X/data"], dtype="float32"),
                np.asarray(handle["X/indices"], dtype="int32"),
                np.asarray(handle["X/indptr"], dtype="int64"),
            ),
            shape=shape,
        )[:, columns].toarray()
    cardinality = np.asarray([len(row) for row in parsed])
    keep = (cardinality >= 1) & (cardinality <= 8)
    control = np.asarray([value.lower() == "non-targeting" for value in condition])
    channel_control = {}
    for channel in np.unique(channels):
        rows = (channels == channel) & control
        if not np.any(rows):
            raise ValueError(f"{path.name} channel {channel} lacks non-targeting controls")
        channel_control[channel] = expression[rows].mean(axis=0)
    target = expression[keep] - np.stack([channel_control[channel] for channel in channels[keep]])
    selected = [row for row, retain in zip(parsed, keep) if retain]
    actions = np.full((len(selected), 8), -1, dtype="int32")
    action_doses = np.zeros((len(selected), 8), dtype="int8")
    for row, targets in enumerate(selected):
        actions[row, : len(targets)] = [action_id[target_name] for target_name, _ in targets]
        action_doses[row, : len(targets)] = [dose for _, dose in targets]
    action_modes = np.full(actions.shape, "", dtype="<U10")
    action_modes[actions >= 0] = mode
    audit = {
        "sample": path.stem,
        "mode": mode,
        "cells": len(condition),
        "retained_cells": int(keep.sum()),
        "control_cells": int(control.sum()),
        "retained_by_cardinality": {
            str(card): int((cardinality[keep] == card).sum())
            for card in np.unique(cardinality[keep])
        },
        "input_sha256": sha256(path),
    }
    return {
        "actions": actions,
        "action_modes": action_modes,
        "action_doses": action_doses,
        "target": target.astype("float32"),
        "condition": np.repeat(f"{mode}|LPS=3h|{condition_name}", len(actions)),
        "replicate": channels[keep],
        "cell_id": np.asarray([f"{condition_name}:{barcode}" for barcode in barcodes[keep]]),
        "guide_load": guide_load[keep],
        "audit": audit,
    }


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inspected = [inspect(raw / filename) for filename, _, _ in SOURCES]
    action_names = np.asarray(sorted(set().union(*(row[2] for row in inspected))))
    common_features = set(inspected[0][1])
    for _, features, _ in inspected[1:]:
        common_features &= set(features)
    target_features = np.asarray(sorted(common_features & set(action_names)))
    if len(target_features) < 500:
        raise ValueError(f"unexpectedly small source-defined expression panel: {len(target_features)}")
    action_id = {name: index for index, name in enumerate(action_names)}
    rows = [
        load_source(raw / filename, mode, condition, action_id, target_features)
        for filename, mode, condition in SOURCES
    ]
    actions = np.concatenate([row["actions"] for row in rows])
    valid = actions >= 0
    pack = output / "gse221321_random_composite_cells_v1.npz"
    np.savez_compressed(
        pack,
        actions=actions,
        action_modes=np.concatenate([row["action_modes"] for row in rows]),
        action_doses=np.concatenate([row["action_doses"] for row in rows]),
        action_names=action_names,
        target=np.concatenate([row["target"] for row in rows]),
        target_semantics=np.asarray("perturbation_delta"),
        target_feature_name=target_features,
        cardinality=valid.sum(axis=1).astype("int8"),
        source_id=np.asarray("GSE221321"),
        context_id=np.asarray("THP1_LPS_3H"),
        experimental_condition_id=np.concatenate([row["condition"] for row in rows]),
        replicate_id=np.concatenate([row["replicate"] for row in rows]),
        cell_id=np.concatenate([row["cell_id"] for row in rows]),
        guide_load=np.concatenate([row["guide_load"] for row in rows]),
        observation_unit=np.asarray("single_cell"),
    )
    source_manifest = raw / "source_manifest.json"
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse221321-random-composite-cells-v1",
        "source": {
            "name": "Compressed Perturb-seq: highly efficient screens for regulatory circuits using random composite perturbations",
            "accession": "GSE221321",
            "doi": "10.1038/s41587-023-01964-9",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221321",
            "source_manifest_sha256": sha256(source_manifest) if source_manifest.exists() else None,
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Source-normalized single-cell expression was restricted to the source-defined perturbation-target panel shared by both compressed screens and differenced from the non-targeting mean in the same 10x channel. Source-curated gene-collapsed assignments were retained at cardinalities one through eight; non-targeting and safe-targeting guides were removed from the biological action set while total guide load was retained.",
        "schema_description": f"NPZ with up to eight gene actions per cell, per-action knockout/repression modes and guide doses, a target vocabulary, {len(target_features)}-dimensional perturbation-delta expression state, THP-1/LPS context, assay condition, replicate, cell ID, and guide load.",
        "population": "LPS-stimulated THP-1 cells in cell-pooled CRISPR knockout and guide-pooled CRISPR interference screens",
        "endpoints": ["source-normalized single-cell expression change from channel-matched non-targeting control"],
        "split_construction": "No train/test split is embedded. The hard generalization gate groups exact intervention sets and constructs deterministic folds downstream.",
        "exclusions": "Control-only cells and cells with more than eight biological targets were excluded. The conventional single-guide screens and raw archive remain locally checksum-manifested but are not duplicated in this cleaned release.",
        "rows": len(actions),
        "unique_action_targets": len(action_names),
        "expression_features": len(target_features),
        "sources": [row["audit"] for row in rows],
        "sl_labels_used": False,
        "files": [{"path": pack.name, "bytes": pack.stat().st_size, "sha256": sha256(pack)}],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse221321")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse221321-random-composite-cells-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
