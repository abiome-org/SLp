"""Build a leakage-bounded paired RNA/ADT Frangieh development corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_data.py"
SPEC = importlib.util.spec_from_file_location("frangieh_data", MODULE_PATH)
FD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FD)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode(values: object) -> np.ndarray:
    out = np.asarray(values)
    if out.dtype.kind == "S":
        return np.char.decode(out, "utf-8")
    if out.dtype.kind == "O":
        return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in out])
    return out


def column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group) and {"codes", "categories"} <= set(node):
        categories = decode(node["categories"])
        codes = np.asarray(node["codes"])
        out = np.empty(codes.shape, dtype=object)
        out[codes < 0] = ""
        out[codes >= 0] = categories[codes[codes >= 0]]
        return out
    return decode(node)


def _stable_source_mapping(handle: h5py.File) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    symbols = column(handle["var"], "gene_symbol").astype(str)
    identities = column(handle["var"], "ensembl_id").astype(str)
    stable = np.char.startswith(identities, "ENSG") & np.asarray(
        [len(x) == 15 and x[4:].isdigit() for x in identities]
    )
    stable_ids = identities[stable]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("duplicate stable RNA query identities")
    mapping = {symbol: identity for symbol, identity in zip(symbols[stable], stable_ids, strict=True)}
    return mapping, np.flatnonzero(stable), stable_ids


def _group_rows(classification: dict, environment: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    selected = np.flatnonzero(classification["allowed"])
    keys = []
    for source_row in selected:
        if classification["control"][source_row]:
            key = ("", str(environment[source_row]), "__VERIFIED_NONTARGETING_CONTROL__")
        else:
            key = (
                str(classification["action_id"][source_row]),
                str(environment[source_row]),
                str(classification["target_guide_set"][source_row]),
            )
        keys.append(key)
    roster = sorted(set(keys))
    lookup = {key: index for index, key in enumerate(roster)}
    return np.asarray([lookup[key] for key in keys], dtype=np.int64), roster


def _selected_dense_channels(handle: h5py.File, selected: np.ndarray) -> np.ndarray:
    x = handle["X"]
    indptr, indices, data = x["indptr"], x["indices"], x["data"]
    source_to_selected = np.full(int(x.attrs["shape"][0]), -1, dtype=np.int64)
    source_to_selected[selected] = np.arange(len(selected))
    result = np.zeros((len(selected), int(x.attrs["shape"][1])), dtype=np.float32)
    for channel in range(result.shape[1]):
        start, stop = int(indptr[channel]), int(indptr[channel + 1])
        rows = source_to_selected[np.asarray(indices[start:stop])]
        keep = rows >= 0
        positions = np.flatnonzero(keep).astype(np.int64) + start
        result[rows[keep], channel] = np.asarray(data[positions])
    return result


def build(rna_path: Path, protein_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(rna_path, "r") as rna, h5py.File(protein_path, "r") as protein:
        mapping, rna_source_columns, query_ids = _stable_source_mapping(rna)
        perturbation = column(rna["obs"], "perturbation").astype(str)
        guide_id = column(rna["obs"], "guide_id").astype(str)
        environment = column(rna["obs"], "perturbation_2").astype(str)
        cell_id = column(rna["obs"], "cell_name").astype(str)
        if not np.array_equal(cell_id, column(protein["obs"], "cell_name").astype(str)):
            raise ValueError("paired cell axes differ")
        classification = FD.classify_complete_guide_rows(perturbation, guide_id, mapping)
        selected = np.flatnonzero(classification["allowed"])
        group_index, groups = _group_rows(classification, environment)
        group_sizes = np.bincount(group_index, minlength=len(groups)).astype(np.int64)

        rna_x = rna["X"]
        read_rna = lambda positions: np.asarray(rna_x["data"][positions])
        denominators = FD.selected_row_sums_from_csc(
            rna_x["indptr"], rna_x["indices"], read_rna, selected
        )
        if np.any(denominators <= 0):
            raise ValueError("selected RNA cell has zero retained-panel UMI count")
        source_ncounts = column(rna["obs"], "ncounts").astype(np.float64)[selected]
        denominator_difference = np.abs(denominators - source_ncounts)
        rna_targets = FD.aggregate_transformed_csc_columns(
            rna_x["indptr"],
            rna_x["indices"],
            read_rna,
            selected,
            denominators,
            group_index,
            rna_source_columns,
        )

        raw_adt = _selected_dense_channels(protein, selected)
        protein_names = column(protein["var"], "protein").astype(str)
        channel_ids = column(protein["var"], "Barcode").astype(str)
        matched_isotype = column(protein["var"], "Isotype_control").astype(str)
        molecular_columns = np.flatnonzero(matched_isotype != "nan")
        isotype_columns = np.flatnonzero(matched_isotype == "nan")
        name_to_column = {name: index for index, name in enumerate(protein_names)}
        protein_targets = np.empty((len(groups), len(molecular_columns)), dtype=np.float32)
        for output_column, target_column in enumerate(molecular_columns):
            control_name = matched_isotype[target_column]
            if control_name not in name_to_column:
                raise ValueError(f"missing matched isotype channel: {control_name}")
            per_cell = FD.matched_isotype_transform(
                raw_adt[:, target_column], raw_adt[:, name_to_column[control_name]]
            )
            sums = np.bincount(group_index, weights=per_cell, minlength=len(groups))
            protein_targets[:, output_column] = (sums / group_sizes).astype(np.float32)

        action_id = np.asarray([key[0] for key in groups])
        context_id = np.asarray([key[1] for key in groups])
        target_guide_set = np.asarray([key[2] for key in groups])
        control_group = action_id == ""
        split_label = np.asarray(
            ["control" if control else FD.split_gene(action) for action, control in zip(action_id, control_group, strict=True)]
        )
        main = ~control_group
        main_indices = np.flatnonzero(main)

        np.savez_compressed(
            output_dir / "development.npz",
            rna_targets=rna_targets[main],
            rna_observed=np.ones(rna_targets[main].shape, dtype=bool),
            rna_query_ids=query_ids,
            rna_query_taxon=np.full(len(query_ids), 9606, dtype=np.int64),
            protein_targets=protein_targets[main],
            protein_observed=np.ones(protein_targets[main].shape, dtype=bool),
            protein_channel_ids=channel_ids[molecular_columns],
            protein_names=protein_names[molecular_columns],
            protein_matched_isotype=matched_isotype[molecular_columns],
            action_ids=action_id[main],
            action_taxon=np.full(np.sum(main), 9606, dtype=np.int64),
            context_ids=context_id[main],
            source_target_guide_sets=target_guide_set[main],
            num_cells=group_sizes[main],
            record_ids=np.asarray(
                [f"frangieh2021|{action_id[i]}|{context_id[i]}|{target_guide_set[i]}" for i in main_indices]
            ),
            split_train=np.flatnonzero(split_label[main] == "train"),
            split_validation=np.flatnonzero(split_label[main] == "validation"),
            control_rna_targets=rna_targets[control_group],
            control_protein_targets=protein_targets[control_group],
            control_context_ids=context_id[control_group],
            control_num_cells=group_sizes[control_group],
            raw_adt_channel_ids=channel_ids,
            raw_adt_channel_names=protein_names,
            raw_adt_isotype_mapping=matched_isotype,
            raw_adt_qc_isotype_index=isotype_columns,
        )
        np.savez_compressed(
            output_dir / "paired-cell-access.npz",
            source_row_index=selected,
            cell_ids=cell_id[selected],
            action_ids=classification["action_id"][selected],
            split=classification["split"][selected],
            context_ids=environment[selected],
            full_guide_ids=guide_id[selected],
            target_guide_sets=classification["target_guide_set"][selected].astype(str),
            rna_denominator=denominators.astype(np.float32),
        )
        test_ids, test_counts = np.unique(
            classification["action_id"][classification["split"] == "test"], return_counts=True
        )
        np.savez_compressed(output_dir / "test-gene-roster-metadata-only.npz", action_ids=test_ids, cell_counts=test_counts)

    report = {
        "schema": "slp.frangieh-paired-development-manifest/v1",
        "source": {
            "rna_sha256": digest(rna_path),
            "protein_sha256": digest(protein_path),
        },
        "counts": {
            "selected_cells": len(selected),
            "development_target_cells": int(np.sum(classification["split"] == "train") + np.sum(classification["split"] == "validation")),
            "verified_control_cells": int(np.sum(classification["split"] == "control")),
            "test_gene_cells_excluded": int(np.sum(classification["split"] == "test")),
            "quarantined_cells": int(np.sum(classification["split"] == "quarantine")),
            "development_records": int(np.sum(main)),
            "train_records": int(np.sum(split_label[main] == "train")),
            "validation_records": int(np.sum(split_label[main] == "validation")),
            "control_records": int(np.sum(control_group)),
            "rna_queries_stable_ensembl": len(query_ids),
            "protein_molecular_channels": len(molecular_columns),
            "protein_qc_isotype_channels": len(isotype_columns),
        },
        "row_classification_reasons": {str(k): int(v) for k, v in Counter(classification["reason"]).items()},
        "rna_denominator_audit": {
            "definition": "exact sum over all 23,712 retained source RNA columns for each selected cell",
            "maximum_absolute_difference_from_obs_ncounts": float(np.max(denominator_difference)),
            "cells_differing_from_obs_ncounts": int(np.sum(denominator_difference != 0)),
        },
        "artifacts": {},
        "limitations": [
            "Patient and plate/replicate assignments are absent from the source AnnData metadata.",
            "Co-culture measurements include survivor-selection effects and are not a pure acute signaling trajectory.",
            "Protein epitopes remain assay channels; no ambiguous channel is forced to a single gene identity.",
            "RNA and ADT values are separate heads and are not numerically comparable to Replogle author z-scores.",
        ],
    }
    for name in ("development.npz", "paired-cell-access.npz", "test-gene-roster-metadata-only.npz", "protocol.json"):
        path = output_dir / name
        report["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.rna, args.protein, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
