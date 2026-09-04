"""Build a mode-aware K562 perturbation-state pack from GEO GSE220974."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
CONTROL = "Non-Targeting"
MODE = {"a": "activation", "i": "repression"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_axis(path: Path, column: int = 0) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None).iloc[:, column].to_numpy(dtype=str)


def interventions(condition: str) -> tuple[tuple[str, str, int], ...]:
    if condition == CONTROL:
        return ()
    actions: dict[tuple[str, str], int] = {}
    for token in condition.split("|"):
        target, short_mode = token.rsplit("-", 1)
        if short_mode not in MODE:
            raise ValueError(f"unknown action mode in {condition!r}")
        identity = (target.upper(), MODE[short_mode])
        actions[identity] = actions.get(identity, 0) + 1
    return tuple((*identity, dose) for identity, dose in actions.items())


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    barcodes = read_axis(raw / "GSE220974_RNA_barcodes.tsv.gz")
    feature_names = read_axis(raw / "GSE220974_RNA_features.tsv.gz", 1)
    metadata = pd.read_csv(raw / "GSE220974_K562_cell_metadata.csv.gz").set_index("cell")
    if len(metadata) != len(barcodes) or set(metadata.index) != set(barcodes):
        raise ValueError("cell metadata and RNA barcodes do not match exactly")
    metadata = metadata.loc[barcodes]

    expression = mmread(raw / "GSE220974_RNA_matrix.mtx.gz").T.tocsr().astype("float32")
    if expression.shape != (len(barcodes), len(feature_names)):
        raise ValueError("declared expression matrix axes do not match metadata")
    library_size = np.asarray(expression.sum(axis=1)).ravel().clip(1)
    expression = expression.multiply((1e4 / library_size)[:, None]).tocsr()
    np.log1p(expression.data, out=expression.data)

    samples = metadata["orig.ident"].astype(str).to_numpy()
    conditions = metadata["guide_group2"].astype(str).to_numpy()
    group_keys = np.asarray([f"{sample}\0{condition}" for sample, condition in zip(samples, conditions)])
    unique_keys, group = np.unique(group_keys, return_inverse=True)
    counts = np.bincount(group, minlength=len(unique_keys))
    assignment = sparse.csr_matrix(
        (np.ones(len(group), dtype="float32"), (group, np.arange(len(group)))),
        shape=(len(unique_keys), len(group)),
    )
    means = (assignment @ expression).multiply((1 / counts)[:, None]).toarray()
    key_parts = [key.split("\0", 1) for key in unique_keys]
    controls = {
        sample: means[index]
        for index, (sample, condition) in enumerate(key_parts)
        if condition == CONTROL
    }
    if set(samples) - set(controls):
        raise ValueError("every experimental replicate must have a non-targeting control")

    selected = np.asarray([condition != CONTROL for _, condition in key_parts])
    selected_keys = [parts for parts, keep in zip(key_parts, selected) if keep]
    parsed = [interventions(condition) for _, condition in selected_keys]
    max_actions = max(map(len, parsed))
    if max_actions != 2:
        raise ValueError(f"expected singles and pairs, observed maximum cardinality {max_actions}")
    action_names = np.asarray(sorted({target for row in parsed for target, _, _ in row}))
    action_id = {name: index for index, name in enumerate(action_names)}
    actions = np.full((len(parsed), max_actions), -1, dtype="int32")
    action_modes = np.full((len(parsed), max_actions), "", dtype="<U10")
    action_doses = np.zeros((len(parsed), max_actions), dtype="int8")
    for row, members in enumerate(parsed):
        for slot, (target, mode, dose) in enumerate(members):
            actions[row, slot] = action_id[target]
            action_modes[row, slot] = mode
            action_doses[row, slot] = dose
    target = np.asarray(
        [means[index] - controls[sample] for index, (sample, _) in enumerate(key_parts) if selected[index]],
        dtype="float32",
    )
    replicate = np.asarray([sample for sample, _ in selected_keys])
    condition_labels = np.asarray([condition for _, condition in selected_keys])
    arrays = {
        "actions": actions,
        "action_modes": action_modes,
        "action_doses": action_doses,
        "action_names": action_names,
        "target": target,
        "target_semantics": np.asarray("perturbation_delta"),
        "target_feature_name": feature_names,
        "cardinality": (actions >= 0).sum(axis=1).astype("int8"),
        "source_id": np.asarray("GSE220974"),
        "context_id": np.asarray("K562"),
        "experimental_condition_id": np.asarray("CRISPRai|duration=unreported"),
        "replicate_id": replicate,
        "source_condition": condition_labels,
        "cell_count": counts[selected].astype("int32"),
    }
    pack = output / "gse220974_crisprai_pseudobulk_v1.npz"
    np.savez_compressed(pack, **arrays)
    manifest_path = raw / "source_manifest.json"
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse220974-crisprai-pseudobulk-v1",
        "source": {
            "name": "CRISPRai Perturb-seq",
            "accession": "GSE220974",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220974",
            "source_manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Source counts were normalized per cell as log1p counts per 10,000, averaged by source-curated replicate and intervention condition, and differenced from the replicate-matched non-targeting mean. Identical target-mode guide duplicates were represented as integer action dose.",
        "schema_description": "NPZ with target-gene action IDs, per-action activation/repression modes and guide doses, action-name vocabulary, full-gene perturbation-delta state, study/context/replicate metadata, and source cell counts.",
        "population": "K562 cells",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from replicate-matched non-targeting control"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "Non-targeting rows define controls and are not outcomes. Raw cell-level source files are checksum-manifested locally but not duplicated in this cleaned release. Duration was not reported in the GEO Series metadata.",
        "actions": "curated source guide_group2 targets with activation/repression retained per action",
        "duration": "not reported in the GEO Series metadata",
        "cells": len(barcodes),
        "rows": len(target),
        "single_rows": int(((actions >= 0).sum(axis=1) == 1).sum()),
        "pair_rows": int(((actions >= 0).sum(axis=1) == 2).sum()),
        "unique_action_targets": len(action_names),
        "unique_intervention_sets": len(np.unique(condition_labels)),
        "replicates": sorted(np.unique(replicate).tolist()),
        "expression_features": len(feature_names),
        "sl_labels_used": False,
        "files": [{"path": pack.name, "bytes": pack.stat().st_size, "sha256": sha256(pack)}],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse220974")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse220974-crisprai-pseudobulk-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
