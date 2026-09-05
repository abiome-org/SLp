"""Leakage-bounded utilities for paired Frangieh Perturb-CITE-seq counts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

SPLIT_PREFIX = "slp11-development-v1|731"


def split_gene(ensembl_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_PREFIX}|9606|{ensembl_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def classify_source_rows(
    perturbation: np.ndarray,
    source_symbol_to_stable_id: dict[str, str],
) -> dict[str, np.ndarray]:
    """Classify rows using metadata before callers request any matrix values."""
    perturbation = np.asarray(perturbation, dtype=str)
    action_id = np.full(len(perturbation), "", dtype="U15")
    split = np.full(len(perturbation), "quarantine", dtype="U10")
    control = perturbation == "control"
    split[control] = "control"
    mapped = np.asarray([source_symbol_to_stable_id.get(x, "") for x in perturbation])
    targeting = mapped != ""
    action_id[targeting] = mapped[targeting]
    split[targeting] = np.asarray([split_gene(x) for x in mapped[targeting]])
    allowed = control | (split == "train") | (split == "validation")
    return {"allowed": allowed, "action_id": action_id, "split": split, "control": control}


def parse_complete_guide_actions(guide_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return targeting guide labels and genes, rejecting truncated provenance."""
    if guide_id in {"", "nan"}:
        return (), ()
    targeting_guides = []
    for token in guide_id.split(";"):
        match = re.fullmatch(r"(.+)_([0-9]+)", token)
        if match is None:
            raise ValueError("malformed or truncated guide provenance")
        if match.group(1) not in {"NO_SITE", "ONE_NON-GENE_SITE"}:
            targeting_guides.append(token)
    genes = tuple(sorted({token.rsplit("_", 1)[0] for token in targeting_guides}))
    return tuple(sorted(targeting_guides)), genes


def classify_complete_guide_rows(
    perturbation: np.ndarray,
    guide_id: np.ndarray,
    source_symbol_to_stable_id: dict[str, str],
) -> dict[str, np.ndarray]:
    """Allow only verified controls or one complete, stable target-gene action."""
    perturbation = np.asarray(perturbation, dtype=str)
    guide_id = np.asarray(guide_id, dtype=str)
    if perturbation.shape != guide_id.shape:
        raise ValueError("perturbation and guide arrays must align")
    action_id = np.full(len(perturbation), "", dtype="U15")
    split = np.full(len(perturbation), "quarantine", dtype="U10")
    target_guide_set = np.full(len(perturbation), "", dtype=object)
    reason = np.full(len(perturbation), "", dtype=object)
    cache: dict[str, tuple[tuple[str, ...], tuple[str, ...]] | None] = {}
    for label in set(guide_id):
        try:
            cache[label] = parse_complete_guide_actions(label)
        except ValueError:
            cache[label] = None
    for index, (primary, provenance) in enumerate(zip(perturbation, guide_id, strict=True)):
        parsed = cache[provenance]
        if parsed is None:
            reason[index] = "malformed-or-truncated-guide-provenance"
            continue
        guides, genes = parsed
        if not genes:
            if primary == "control":
                split[index] = "control"
                reason[index] = "verified-nontargeting-control"
            else:
                reason[index] = "primary-target-without-targeting-guide"
            continue
        if len(genes) != 1:
            reason[index] = "multiple-target-genes"
            continue
        gene = genes[0]
        if primary != gene:
            reason[index] = "harmonized-primary-disagrees-with-guide-provenance"
            continue
        stable_id = source_symbol_to_stable_id.get(gene, "")
        if not stable_id:
            reason[index] = "target-gene-without-stable-ensembl-id"
            continue
        action_id[index] = stable_id
        split[index] = split_gene(stable_id)
        target_guide_set[index] = ";".join(guides)
        reason[index] = "single-target-gene"
    allowed = (split == "control") | (split == "train") | (split == "validation")
    return {
        "allowed": allowed,
        "action_id": action_id,
        "split": split,
        "control": split == "control",
        "target_guide_set": target_guide_set,
        "reason": reason,
    }


def selected_row_sums_from_csc(
    indptr: np.ndarray,
    indices: np.ndarray,
    read_selected_values: Callable[[np.ndarray], np.ndarray],
    selected_rows: np.ndarray,
) -> np.ndarray:
    """Sum source columns for allowlisted rows without exposing excluded outputs.

    ``selected_rows`` must be frozen from metadata before this function is
    called. The value reader is invoked one source column at a time and only
    allowlisted entries contribute to the returned denominators.
    """
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    if len(np.unique(selected_rows)) != len(selected_rows):
        raise ValueError("selected rows must be unique")
    n_source_rows = int(max(np.max(indices, initial=-1), np.max(selected_rows, initial=-1))) + 1
    source_to_selected = np.full(n_source_rows, -1, dtype=np.int64)
    source_to_selected[selected_rows] = np.arange(len(selected_rows))
    totals = np.zeros(len(selected_rows), dtype=np.float64)
    for column in range(len(indptr) - 1):
        start, stop = int(indptr[column]), int(indptr[column + 1])
        source_index = np.asarray(indices[start:stop], dtype=np.int64)
        destination = source_to_selected[source_index]
        keep = destination >= 0
        if np.any(keep):
            positions = np.flatnonzero(keep).astype(np.int64) + start
            values = np.asarray(read_selected_values(positions), dtype=np.float64)
            np.add.at(totals, destination[keep], values)
    return totals


def aggregate_transformed_csc_columns(
    indptr: np.ndarray,
    indices: np.ndarray,
    read_selected_values: Callable[[np.ndarray], np.ndarray],
    selected_rows: np.ndarray,
    denominators: np.ndarray,
    group_index: np.ndarray,
    source_columns: np.ndarray,
    scale: float = 10_000.0,
) -> np.ndarray:
    """Aggregate per-cell log1p scaled counts without a dense cell matrix."""
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    denominators = np.asarray(denominators, dtype=np.float64)
    group_index = np.asarray(group_index, dtype=np.int64)
    if len(selected_rows) != len(denominators) or len(group_index) != len(selected_rows):
        raise ValueError("selected-row arrays have inconsistent lengths")
    if np.any(denominators <= 0) or np.any(group_index < 0):
        raise ValueError("denominators must be positive and groups nonnegative")
    n_groups = int(group_index.max(initial=-1)) + 1
    group_sizes = np.bincount(group_index, minlength=n_groups).astype(np.float64)
    n_source_rows = int(max(np.max(indices, initial=-1), np.max(selected_rows, initial=-1))) + 1
    source_to_selected = np.full(n_source_rows, -1, dtype=np.int64)
    source_to_selected[selected_rows] = np.arange(len(selected_rows))
    result = np.zeros((n_groups, len(source_columns)), dtype=np.float32)
    for output_column, source_column in enumerate(np.asarray(source_columns, dtype=np.int64)):
        start, stop = int(indptr[source_column]), int(indptr[source_column + 1])
        source_index = np.asarray(indices[start:stop], dtype=np.int64)
        destination = source_to_selected[source_index]
        keep = destination >= 0
        if not np.any(keep):
            continue
        positions = np.flatnonzero(keep).astype(np.int64) + start
        values = np.asarray(read_selected_values(positions), dtype=np.float64)
        selected = destination[keep]
        transformed = np.log1p(scale * values / denominators[selected])
        sums = np.bincount(group_index[selected], weights=transformed, minlength=n_groups)
        result[:, output_column] = (sums / group_sizes).astype(np.float32)
    return result


def matched_isotype_transform(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Author-defined ADT value: max(0, ln((target+1)/(isotype+1)))."""
    target = np.asarray(target, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if target.shape != control.shape or np.any(target < 0) or np.any(control < 0):
        raise ValueError("paired ADT counts must have equal shapes and be nonnegative")
    return np.maximum(0.0, np.log((target + 1.0) / (control + 1.0)))


def load_paired_cell_access(path: str | Path) -> dict[str, np.ndarray]:
    """Load the metadata-only allowlist used for future paired cell shards."""
    required = {
        "source_row_index",
        "cell_ids",
        "action_ids",
        "split",
        "context_ids",
        "full_guide_ids",
        "target_guide_sets",
        "rna_denominator",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("unexpected paired-cell access schema")
        result = {key: np.asarray(archive[key]) for key in required}
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1 or not np.all(np.isin(result["split"], ["train", "validation", "control"])):
        raise ValueError("paired-cell access contains inconsistent or forbidden rows")
    if len(np.unique(result["source_row_index"])) != len(result["source_row_index"]):
        raise ValueError("source row indices must be unique")
    return result


def iter_paired_cell_shards(access: dict[str, np.ndarray], shard_size: int = 4096):
    """Yield bounded metadata shards whose source rows are already allowlisted."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    length = len(access["source_row_index"])
    for start in range(0, length, shard_size):
        stop = min(length, start + shard_size)
        yield {key: value[start:stop] for key, value in access.items()}
