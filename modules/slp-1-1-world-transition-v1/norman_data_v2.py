"""Build corrected Norman 2019 single/double CRISPRa molecular records.

Every eligible cell is normalized by its full-source-gene library total before
condition aggregation. Targets are pseudobulk means of per-cell log2-CP10k,
standardized by per-query mean and population standard deviation estimated
only from eligible non-targeting control cells. Test-routed outcomes are written
to a separate artifact and are never analyzed by this builder.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import norman_data as v1
import numpy as np
from scipy import sparse
from scipy.io import mmread

VALUE_SPACE = "per-cell-full-library-log2-cp10k-control-zscore-pseudobulk-mean-v2"
CONTEXT_VALUE_SPACE = "per-cell-full-library-log2-cp10k-core-control-mean-v2"
NUM_CELLS_ROLE = "likelihood-exposure-only-not-mean-predictor-input"
CONTROL_GROUPS = 20
CONTROL_PARTITION_NAMESPACE = "slp11-norman-core-control-pseudobulk-v2"


@dataclass(frozen=True)
class Condition:
    construct_id: str
    actions: tuple[str, ...]


class NormanDataV2Error(ValueError):
    """Raised when the corrected Norman source contract is violated."""


def load_metadata_v2(
    barcodes_path: Path, identities_path: Path, gtf_path: Path
) -> dict[str, object]:
    """Retain exact author construct populations while mapping stable actions."""

    barcodes = v1._read_lines(barcodes_path)
    if len(barcodes) != len(set(barcodes)):
        raise NormanDataV2Error("filtered barcodes are not unique")
    identities: dict[str, dict[str, str]] = {}
    with gzip.open(identities_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            barcode = row.get("cell_barcode", "")
            if not barcode or barcode in identities:
                raise NormanDataV2Error("cell identity barcode is empty or duplicated")
            identities[barcode] = row
    if set(identities) - set(barcodes):
        raise NormanDataV2Error("cell identity table contains unknown barcodes")

    gene_mapping = v1._load_gtf_mapping(gtf_path)
    cell_condition: list[Condition | None] = []
    eligible = np.zeros(len(barcodes), dtype=np.bool_)
    unresolved_cells = 0
    construct_actions: dict[str, tuple[str, ...]] = {}
    for index, barcode in enumerate(barcodes):
        row = identities.get(barcode)
        if row is None or not (
            row.get("good_coverage") == "True"
            and row.get("number_of_cells") == "1"
            and row.get("guide_identity") != "*"
        ):
            cell_condition.append(None)
            continue
        guide_identity = row["guide_identity"]
        symbols = v1._parse_target_symbols(guide_identity)
        if symbols:
            mapped = tuple(gene_mapping.get(symbol, "") for symbol in symbols)
            if any(not item for item in mapped) or len(set(mapped)) != len(mapped):
                unresolved_cells += 1
                cell_condition.append(None)
                continue
            actions = tuple(sorted(mapped))
            construct_id = guide_identity
        else:
            actions = ()
            construct_id = "core-nontargeting-controls"
        if len(actions) > 2:
            raise NormanDataV2Error("more than two stable actions in one construct")
        previous = construct_actions.setdefault(construct_id, actions)
        if previous != actions:
            raise NormanDataV2Error("one construct maps to inconsistent stable actions")
        eligible[index] = True
        cell_condition.append(Condition(construct_id, actions))

    conditions = tuple(
        sorted(
            {item for item in cell_condition if item is not None},
            key=lambda item: (len(item.actions) != 0, item.construct_id),
        )
    )
    condition_index = {item: index for index, item in enumerate(conditions)}
    selected_columns = np.flatnonzero(eligible).astype(np.int64)
    selected_condition = np.asarray(
        [condition_index[cell_condition[index]] for index in selected_columns],
        dtype=np.int64,
    )
    cell_counts = np.bincount(selected_condition, minlength=len(conditions)).astype(np.int64)
    return {
        "barcodes": barcodes,
        "conditions": conditions,
        "selected_columns": selected_columns,
        "selected_condition": selected_condition,
        "cell_counts": cell_counts,
        "unresolved_cells": unresolved_cells,
    }


def _per_cell_log_cp10k(
    query_counts: sparse.spmatrix, full_library_totals: np.ndarray
) -> sparse.csc_matrix:
    """Normalize sparse query counts by full-source libraries before log transform."""

    totals = np.asarray(full_library_totals, dtype=np.float64)
    if query_counts.ndim != 2 or totals.shape != (query_counts.shape[1],):
        raise NormanDataV2Error("query counts and full library totals do not align")
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise NormanDataV2Error("full library totals must be finite and positive")
    normalized = query_counts.astype(np.float64).tocsc(copy=True)
    normalized = normalized.multiply(10_000.0 / totals[None, :]).tocsc()
    normalized.data = np.log2(1.0 + normalized.data)
    if not np.isfinite(normalized.data).all():
        raise NormanDataV2Error("per-cell normalization produced non-finite values")
    return normalized


def _means_by_group(
    query_by_cell: sparse.csc_matrix,
    group_index: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(group_index)
    if groups.shape != (query_by_cell.shape[1],) or groups.dtype.kind not in "iu":
        raise NormanDataV2Error("group index must align with normalized cells")
    if group_count < 1 or np.any(groups < 0) or np.any(groups >= group_count):
        raise NormanDataV2Error("group index is out of range")
    counts = np.bincount(groups, minlength=group_count).astype(np.int64)
    if np.any(counts == 0):
        raise NormanDataV2Error("every pseudobulk group requires cells")
    membership = sparse.csr_matrix(
        (
            np.ones(len(groups), dtype=np.float64),
            (np.arange(len(groups)), groups),
        ),
        shape=(len(groups), group_count),
    )
    sums = query_by_cell @ membership
    return sums.toarray().T / counts[:, None], counts


def _control_mean_std(control_cells: sparse.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    if control_cells.shape[1] < 2:
        raise NormanDataV2Error("at least two core control cells are required")
    mean = np.asarray(control_cells.mean(axis=1)).ravel()
    squared = control_cells.copy()
    squared.data **= 2
    mean_square = np.asarray(squared.mean(axis=1)).ravel()
    variance = np.maximum(mean_square - np.square(mean), 0.0)
    return mean, np.sqrt(variance)


def _control_partition(barcodes: Sequence[str], groups: int = CONTROL_GROUPS) -> np.ndarray:
    """Assign disjoint hash-shuffled controls to declared unequal group sizes."""

    if groups < 2 or len(barcodes) < groups:
        raise NormanDataV2Error("control partition requires at least one cell per group")
    order = sorted(
        range(len(barcodes)),
        key=lambda index: hashlib.sha256(
            f"{CONTROL_PARTITION_NAMESPACE}|{v1.SEED}|{barcodes[index]}".encode("ascii")
        ).digest(),
    )
    cumulative_weights = np.cumsum(np.arange(1, groups + 1, dtype=np.int64))
    boundaries = np.floor(len(barcodes) * cumulative_weights / cumulative_weights[-1]).astype(
        np.int64
    )
    boundaries[-1] = len(barcodes)
    sizes = np.diff(np.concatenate(([0], boundaries)))
    if np.any(sizes == 0) or sizes.sum() != len(barcodes):
        raise NormanDataV2Error("declared unequal control group sizes are invalid")
    result = np.empty(len(barcodes), dtype=np.int64)
    offset = 0
    for group, size in enumerate(sizes):
        result[np.asarray(order[offset : offset + size], dtype=np.int64)] = group
        offset += int(size)
    return result


def _bundle(
    targets: np.ndarray,
    observed: np.ndarray,
    conditions: tuple[Condition, ...],
    cell_counts: np.ndarray,
    query_ids: tuple[str, ...],
    selection: np.ndarray,
    split_names: np.ndarray,
    basal_target: np.ndarray,
    context_basal: np.ndarray,
    controls: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    chosen = [conditions[index] for index in selection]
    offsets = np.zeros(len(chosen) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(item.actions) for item in chosen])
    selected_splits = split_names[selection]
    arrays = {
        "targets": targets[selection],
        "observed": observed[selection],
        "action_ids": v1._string_array(
            [action for item in chosen for action in item.actions]
        ),
        "action_offsets": offsets,
        "construct_ids": v1._string_array([item.construct_id for item in chosen]),
        "query_ids": v1._string_array(query_ids),
        "context_ids": v1._string_array([v1.CONTEXT_ID]),
        "context_index": np.zeros(len(chosen), dtype=np.int64),
        "record_ids": v1._string_array(
            [f"norman-2019|{item.construct_id}" for item in chosen]
        ),
        "num_cells_filtered": cell_counts[selection],
        "num_cells_role": np.asarray(NUM_CELLS_ROLE),
        "basal_control": basal_target[None, :],
        "context_basal_expression": context_basal[None, :],
        "target_value_space": np.asarray(VALUE_SPACE),
        "context_value_space": np.asarray(CONTEXT_VALUE_SPACE),
        "split_train": np.flatnonzero(selected_splits == "train").astype(np.int64),
        "split_validation": np.flatnonzero(selected_splits == "validation").astype(np.int64),
        "split_test": np.flatnonzero(selected_splits == "test").astype(np.int64),
    }
    if controls is None:
        arrays.update(
            {
                "control_targets": np.empty((0, len(query_ids)), dtype=np.float32),
                "control_observed": np.empty((0, len(query_ids)), dtype=np.bool_),
                "control_context_index": np.empty(0, dtype=np.int64),
                "control_num_cells_filtered": np.empty(0, dtype=np.int64),
                "control_record_ids": np.empty(0, dtype="<U1"),
                "control_core": np.empty(0, dtype=np.bool_),
            }
        )
    else:
        arrays.update(controls)
    return arrays


def build_norman_v2(
    source_dir: str | Path,
    gtf_path: str | Path,
    query_path: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    destination_path = Path(destination)
    if destination_path.exists():
        raise NormanDataV2Error(f"destination already exists: {destination_path}")
    source = Path(source_dir)
    verified = {
        label: v1._verify(source / name, name, size, digest)
        for label, (name, size, digest) in v1.SOURCE_SPECS.items()
    }
    query_ids = v1._load_query_ids(Path(query_path))
    genes = v1._read_lines(verified["genes"])
    gene_ids = tuple(line.split("\t", 1)[0] for line in genes)
    if len(gene_ids) != 33_694 or len(set(gene_ids)) != len(gene_ids):
        raise NormanDataV2Error("filtered gene roster drift")
    metadata = load_metadata_v2(verified["barcodes"], verified["identities"], Path(gtf_path))
    conditions = metadata["conditions"]
    assert isinstance(conditions, tuple)
    selected_columns = np.asarray(metadata["selected_columns"])
    selected_condition = np.asarray(metadata["selected_condition"])
    cell_counts = np.asarray(metadata["cell_counts"])

    matrix = mmread(verified["matrix"]).tocsr()
    if matrix.shape != (len(gene_ids), len(metadata["barcodes"])):
        raise NormanDataV2Error("filtered matrix dimensions drift")
    full_library_totals = np.asarray(matrix.sum(axis=0)).ravel()[selected_columns]
    query_lookup = {gene: index for index, gene in enumerate(query_ids)}
    present_pairs = [
        (row, query_lookup[gene]) for row, gene in enumerate(gene_ids) if gene in query_lookup
    ]
    source_rows = np.asarray([item[0] for item in present_pairs], dtype=np.int64)
    output_rows = np.asarray([item[1] for item in present_pairs], dtype=np.int64)
    normalized = _per_cell_log_cp10k(
        matrix[source_rows][:, selected_columns], full_library_totals
    )
    del matrix
    means, recomputed_counts = _means_by_group(
        normalized, selected_condition, len(conditions)
    )
    if not np.array_equal(recomputed_counts, cell_counts):
        raise NormanDataV2Error("condition cell counts disagree")
    control_index = next(
        (index for index, item in enumerate(conditions) if not item.actions), None
    )
    if control_index is None or sum(not item.actions for item in conditions) != 1:
        raise NormanDataV2Error("exactly one merged core-control population is required")
    control_cells_mask = selected_condition == control_index
    control_cells = normalized[:, control_cells_mask]
    control_mean, control_std = _control_mean_std(control_cells)
    supported_local = control_std > 0.0
    supported_output = output_rows[supported_local]
    targets = np.zeros((len(conditions), len(query_ids)), dtype=np.float32)
    targets[:, supported_output] = (
        (means[:, supported_local] - control_mean[supported_local])
        / control_std[supported_local]
    ).astype(np.float32)
    observed_query = np.zeros(len(query_ids), dtype=np.bool_)
    observed_query[supported_output] = True
    observed = np.broadcast_to(observed_query, targets.shape).copy()
    basal_target = np.zeros(len(query_ids), dtype=np.float32)
    context_basal = np.zeros(len(query_ids), dtype=np.float32)
    context_basal[output_rows] = control_mean.astype(np.float32)

    selected_barcodes = np.asarray(metadata["barcodes"])[selected_columns]
    control_barcodes = selected_barcodes[control_cells_mask]
    control_partition = _control_partition(control_barcodes, CONTROL_GROUPS)
    control_means, control_group_counts = _means_by_group(
        control_cells, control_partition, CONTROL_GROUPS
    )
    control_targets = np.zeros((CONTROL_GROUPS, len(query_ids)), dtype=np.float32)
    control_targets[:, supported_output] = (
        (control_means[:, supported_local] - control_mean[supported_local])
        / control_std[supported_local]
    ).astype(np.float32)
    control_arrays = {
        "control_targets": control_targets,
        "control_observed": np.broadcast_to(
            observed_query, control_targets.shape
        ).copy(),
        "control_context_index": np.zeros(CONTROL_GROUPS, dtype=np.int64),
        "control_num_cells_filtered": control_group_counts,
        "control_record_ids": v1._string_array(
            [f"norman-2019|core-control-pseudobulk-{index:02d}" for index in range(CONTROL_GROUPS)]
        ),
        "control_core": np.ones(CONTROL_GROUPS, dtype=np.bool_),
    }

    split_names = np.asarray(
        ["control" if not item.actions else v1.route_actions(item.actions) for item in conditions]
    )
    perturbation = np.asarray(
        [index for index, item in enumerate(conditions) if item.actions], dtype=np.int64
    )
    development = perturbation[split_names[perturbation] != "test"]
    test = perturbation[split_names[perturbation] == "test"]
    destination_path.mkdir(parents=True, exist_ok=False)
    development_path = destination_path / "norman-2019-author-normalized-development-v2.npz"
    test_path = destination_path / "norman-2019-author-normalized-test-only-v2.npz"
    v1._write_npz(
        development_path,
        _bundle(
            targets, observed, conditions, cell_counts, query_ids, development,
            split_names, basal_target, context_basal, control_arrays,
        ),
    )
    v1._write_npz(
        test_path,
        _bundle(
            targets, observed, conditions, cell_counts, query_ids, test,
            split_names, basal_target, context_basal, None,
        ),
    )

    action_union = sorted({action for item in conditions for action in item.actions})
    manifest = {
        "schema": "slp.norman-2019-combination-response/v2",
        "status": "derived-development-and-routed-test-not-omf-admitted",
        "sourceVersion": "sources/norman-2019-gse133344-author-normalized-v2.yaml",
        "source": "GEO:GSE133344", "context": v1.CONTEXT_ID,
        "ncbiTaxon": v1.TAXON,
        "perturbation": "simultaneous-single-or-double-CRISPRa-activation",
        "targetValueSpace": VALUE_SPACE, "contextValueSpace": CONTEXT_VALUE_SPACE,
        "transform": (
            "full-source-gene library CP10k and log2(1+x) per eligible cell; "
            "condition mean; per-query centering/scaling by eligible core-control "
            "cell mean and population standard deviation"
        ),
        "controlCalibration": {
            "cells": int(control_cells.shape[1]), "pseudobulks": CONTROL_GROUPS,
            "partition": (
                "SHA-256 shuffled disjoint cells allocated to deterministic unequal "
                "sizes proportional to 1..20"
            ),
            "groupCellCounts": control_group_counts.tolist(),
            "purpose": "sampling-variance versus cell-exposure estimation only",
            "reusesTargetStandardizationCells": True,
        },
        "split": "per constituent sha256(slp11-development-v1|731|9606|ENSG), test > validation > train",
        "counts": {
            "sourceGenes": len(gene_ids), "sourceCells": len(metadata["barcodes"]),
            "eligibleCells": len(selected_columns),
            "unresolvedCellsQuarantined": metadata["unresolved_cells"],
            "queryGenes": len(query_ids), "sourcePresentQueryGenes": len(output_rows),
            "sourceAbsentQueryGenes": len(query_ids) - len(output_rows),
            "zeroControlVariancePresentQueries": int((~supported_local).sum()),
            "observedQueryGenes": int(observed_query.sum()),
            "uniqueActionGenes": len(action_union),
            "perturbationConstructRecords": len(perturbation),
            "recordsBySplit": {
                role: int(np.count_nonzero(split_names[perturbation] == role))
                for role in ("train", "validation", "test")
            },
        },
        "identity": {
            "queryNamespace": "Ensembl-gene", "queryRosterSha256": v1.QUERY_SHA256,
            "actionNamespace": "Ensembl-gene",
            "construct": "exact source guide_identity retained per perturbation population",
            "mapping": "author-pinned GRCh38-1.2.0 GTF exact gene_name to unique stable ENSG",
            "authorRepositoryRevision": v1.AUTHOR_REVISION,
        },
        "sourceFiles": {
            label: {"path": spec[0], "bytes": spec[1], "sha256": spec[2]}
            for label, spec in v1.SOURCE_SPECS.items()
        },
        "outputs": {
            "development": {"path": development_path.name,
                            "bytes": development_path.stat().st_size,
                            "sha256": v1._hash(development_path),
                            "contains": ["train", "validation", "core-control-calibration"]},
            "testOnly": {"path": test_path.name, "bytes": test_path.stat().st_size,
                         "sha256": v1._hash(test_path), "contains": ["test"],
                         "controlOutcomeRows": 0,
                         "access": "sealed until candidate and rule lock"},
        },
        "numCellsRole": NUM_CELLS_ROLE,
        "testOutcomeMetricsComputed": False, "benchmarkDataConsumed": False,
    }
    manifest_path = destination_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    action_path = destination_path / "action-ids.txt"
    action_path.write_text(
        "".join(f"{action}\n" for action in action_union),
        encoding="ascii", newline="\n",
    )
    return {
        "manifestPath": str(manifest_path), "manifestSha256": v1._hash(manifest_path),
        "manifest": manifest, "actionRosterSha256": v1._hash(action_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_norman_v2(args.source_dir, args.gtf, args.query_ids, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
