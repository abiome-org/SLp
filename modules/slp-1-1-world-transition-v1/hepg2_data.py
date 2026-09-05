"""Streaming control-normalized HepG2 molecular adapter.

The default CLI mode is metadata-only planning. Perturbed expression is read
only when ``--execute-perturbed`` is explicitly supplied. The molecular
endpoint is computed by SLp: individual raw cells are transformed with frozen
per-GEM non-targeting-control location/scale parameters, then averaged within
each exact source construct population.

This is neither the Nadig et al. DESeq2 log-fold-change endpoint nor a claim
that the HepG2 control cohort is source-equivalent to Replogle controls.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from control_normalization import GemGroupControlNormalizer

Array = np.ndarray

TAXON = 9606
SEED = 731
CONTEXT_ID = "nadig-2025-hepg2-day-7"
SOURCE_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
NORMALIZATION_SHA256 = "3f72db203e989cb60d9ecd65874a11d2c83af0772a8011bafcb559a65c459951"
CONTEXT_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
FIXED_PANEL_SHA256 = "046891d3ceb0766e3fd09441677d6ae078fa7ac7d81ddb1f1c30866007d0d959"
TARGET_VALUE_SPACE = "slp-per-gem-control-z-score-cell-mean-v1"
ENSG_RE = re.compile(r"^ENSG[0-9]+$")


class HepG2DataError(ValueError):
    """Raised when a source, pin, or aggregation contract is violated."""


@dataclass(frozen=True)
class PopulationTable:
    """Deterministic exact-construct population membership."""

    action_ids: Array
    population_ids: Array
    construct_ids: Array
    transcript_labels: Array
    cell_population_index: Array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink() or _sha256(resolved) != expected:
        raise HepG2DataError(f"{label} path or SHA-256 drift")
    return resolved


def _categorical(group: h5py.Group, name: str) -> Array:
    dataset = group[name]
    categories_path = f"__categories/{name}"
    if categories_path not in group:
        raise HepG2DataError(f"obs/{name} must be categorical")
    categories = np.asarray(group[categories_path][...]).astype(str)
    codes = np.asarray(dataset[...], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise HepG2DataError(f"obs/{name} has invalid category codes")
    return categories[codes]


def _route(action_id: str) -> str:
    payload = f"slp11-development-v1|{SEED}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def build_population_table(
    action_ids: Array,
    population_ids: Array,
    construct_ids: Array,
    transcript_labels: Array,
) -> PopulationTable:
    """Validate and index exact source construct populations."""

    actions = np.asarray(action_ids).astype(str)
    populations = np.asarray(population_ids).astype(str)
    constructs = np.asarray(construct_ids).astype(str)
    transcripts = np.asarray(transcript_labels).astype(str)
    if not (
        actions.ndim == 1
        and populations.shape == actions.shape
        and constructs.shape == actions.shape
        and transcripts.shape == actions.shape
        and actions.size > 0
    ):
        raise HepG2DataError("cell population metadata must be aligned vectors")
    if any(ENSG_RE.fullmatch(value) is None for value in actions):
        raise HepG2DataError("every targeted cell requires a stable ENSG action")
    if any(not value for array in (populations, constructs, transcripts) for value in array):
        raise HepG2DataError("source population and construct identities cannot be empty")

    unique_populations = np.asarray(sorted(set(populations.tolist())))
    lookup = {value: index for index, value in enumerate(unique_populations)}
    cell_index = np.asarray([lookup[value] for value in populations], dtype=np.int64)
    population_actions = np.empty(unique_populations.size, dtype=actions.dtype)
    population_constructs = np.empty(unique_populations.size, dtype=constructs.dtype)
    population_transcripts = np.empty(unique_populations.size, dtype=transcripts.dtype)
    for index, population in enumerate(unique_populations):
        selected = populations == population
        for source, destination, label in (
            (actions, population_actions, "action"),
            (constructs, population_constructs, "construct"),
            (transcripts, population_transcripts, "transcript"),
        ):
            values = np.unique(source[selected])
            if values.size != 1:
                raise HepG2DataError(
                    f"source population {population} has multiple {label} identities"
                )
            destination[index] = values[0]
    return PopulationTable(
        action_ids=population_actions,
        population_ids=unique_populations,
        construct_ids=population_constructs,
        transcript_labels=population_transcripts,
        cell_population_index=cell_index,
    )


def aggregate_normalized_cells(
    counts: Array,
    full_umi_count: Array,
    gem_group: Array,
    population_index: Array,
    normalizer: GemGroupControlNormalizer,
    *,
    population_count: int,
) -> tuple[Array, Array, Array, Array]:
    """Normalize cells and aggregate missing-aware population means.

    Returns ``targets``, ``observed``, per-query contributing cell counts, and
    total cells per population. The per-query counts are the likelihood
    exposure for a measurement; the total population count must not be used
    when a GEM/query control variance is unsupported.
    """

    values, supported = normalizer.transform(counts, full_umi_count, gem_group)
    membership = np.asarray(population_index)
    if membership.shape != (values.shape[0],) or membership.dtype.kind not in "iu":
        raise HepG2DataError("population_index must be one integer per cell")
    if population_count <= 0 or np.any(membership < 0) or np.any(membership >= population_count):
        raise HepG2DataError("population_index is out of range")
    totals = np.zeros((population_count, values.shape[1]), dtype=np.float64)
    contributing = np.zeros((population_count, values.shape[1]), dtype=np.uint32)
    population_cells = np.bincount(membership, minlength=population_count).astype(np.uint32)
    np.add.at(totals, membership, np.where(supported, values, 0.0))
    np.add.at(contributing, membership, supported.astype(np.uint32))
    observed = contributing > 0
    targets = np.divide(
        totals,
        contributing,
        out=np.zeros_like(totals),
        where=observed,
    )
    return targets, observed, contributing, population_cells


def _load_artifacts(
    normalization_path: Path,
    context_path: Path,
) -> tuple[GemGroupControlNormalizer, dict[str, Array]]:
    _verify(normalization_path, NORMALIZATION_SHA256, "control normalization")
    _verify(context_path, CONTEXT_SHA256, "context descriptor")
    with np.load(normalization_path, allow_pickle=False) as source:
        normalizer = GemGroupControlNormalizer(
            gem_groups_=source["gem_groups"],
            target_umi_=float(source["target_umi"]),
            control_mean_=source["control_mean"].astype(np.float64),
            control_std_=source["control_std"].astype(np.float64),
            control_observed_=source["control_observed"],
            control_counts_=source["control_counts"],
        )
        source_query_ids = source["query_ids"].astype(str)
        source_hash = str(source["source_sha256"])
    with np.load(context_path, allow_pickle=False) as source:
        context = {name: source[name] for name in source.files}
    if (
        source_hash != SOURCE_SHA256
        or str(context["fixed_panel_query_sha256"]) != FIXED_PANEL_SHA256
        or int(context["perturbed_expression_rows_read"]) != 0
    ):
        raise HepG2DataError("frozen control artifact provenance drift")
    context["source_query_ids"] = source_query_ids
    return normalizer, context


def plan(
    source_path: Path,
    normalization_path: Path,
    context_path: Path,
) -> dict[str, object]:
    """Validate pins and metadata without indexing the expression matrix."""

    source_path = _verify(source_path, SOURCE_SHA256, "HepG2 source")
    normalizer, context = _load_artifacts(normalization_path, context_path)
    with h5py.File(source_path, "r") as source:
        matrix_shape = tuple(source["X"].shape)  # metadata only
        obs = source["obs"]
        action_ids = _categorical(obs, "gene_id")
        populations = _categorical(obs, "gene_transcript")
        constructs = _categorical(obs, "sgID_AB")
        transcripts = _categorical(obs, "transcript")
        gems = np.asarray(obs["gem_group"][...], dtype=np.int64)
        full_umi = np.asarray(obs["UMI_count"][...], dtype=np.float64)
        source_queries = np.asarray(source["var/gene_id"][...]).astype(str)
    targeted = np.asarray([ENSG_RE.fullmatch(value) is not None for value in action_ids])
    unresolved = ~(targeted | (action_ids == "non-targeting"))
    table = build_population_table(
        action_ids[targeted],
        populations[targeted],
        constructs[targeted],
        transcripts[targeted],
    )
    output_queries = context["query_ids"].astype(str)
    if (
        matrix_shape != (145_473, 9_624)
        or not np.array_equal(source_queries, context["source_query_ids"])
        or output_queries.shape != (7_036,)
        or context["context_basal_expression"].shape != (1, 7_036)
        or context["context_basal_observed"].shape != (1, 7_036)
        or int(context["context_basal_observed"].sum()) != 6_789
        or set(np.unique(gems[targeted]).tolist()) - set(normalizer.gem_groups_.tolist())
        or not np.isfinite(full_umi[targeted]).all()
        or np.any(full_umi[targeted] <= 0.0)
    ):
        raise HepG2DataError("HepG2 plan contract drift")
    roles = np.asarray([_route(action) for action in table.action_ids])
    return {
        "schema": "slp.hepg2-streaming-adapter-plan/v1",
        "status": "planned-not-executed",
        "sourceSha256": SOURCE_SHA256,
        "controlNormalizationSha256": NORMALIZATION_SHA256,
        "contextDescriptorSha256": CONTEXT_SHA256,
        "targetValueSpace": TARGET_VALUE_SPACE,
        "endpointProvenance": "SLp-computed; not Nadig DESeq2",
        "matrixShape": list(matrix_shape),
        "targetedCells": int(targeted.sum()),
        "unresolvedTargetCellsQuarantined": int(unresolved.sum()),
        "exactConstructPopulations": int(table.population_ids.size),
        "stableTargetGenes": len(set(table.action_ids.tolist())),
        "outputQueries": int(output_queries.size),
        "sourceMeasuredOutputQueries": len(set(source_queries) & set(output_queries)),
        "fixedContextTokens": int(context["context_basal_observed"].sum()),
        "roles": {
            role: int(np.count_nonzero(roles == role))
            for role in ("train", "validation", "test")
        },
        "execution": {
            "requiredFlag": "--execute-perturbed",
            "expressionRowsReadDuringPlan": 0,
            "streamingShardRows": 128,
        },
    }


def _string_array(values: Sequence[str]) -> Array:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _write_npz(path: Path, arrays: dict[str, Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name in sorted(arrays):
                stream = io.BytesIO()
                np.save(stream, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, stream.getvalue())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def execute(
    source_path: Path,
    normalization_path: Path,
    context_path: Path,
    output_path: Path,
    *,
    shard_rows: int = 128,
) -> dict[str, object]:
    """Stream targeted cells and write the frozen SLp molecular endpoint."""

    if shard_rows <= 0 or shard_rows > 512:
        raise HepG2DataError("shard_rows must be between 1 and 512")
    source_path = _verify(source_path, SOURCE_SHA256, "HepG2 source")
    normalizer, context = _load_artifacts(normalization_path, context_path)
    with h5py.File(source_path, "r") as source:
        obs = source["obs"]
        all_actions = _categorical(obs, "gene_id")
        all_populations = _categorical(obs, "gene_transcript")
        all_constructs = _categorical(obs, "sgID_AB")
        all_transcripts = _categorical(obs, "transcript")
        all_gems = np.asarray(obs["gem_group"][...], dtype=np.int64)
        all_umi = np.asarray(obs["UMI_count"][...], dtype=np.float64)
        targeted_rows = np.flatnonzero(
            [ENSG_RE.fullmatch(value) is not None for value in all_actions]
        )
        table = build_population_table(
            all_actions[targeted_rows],
            all_populations[targeted_rows],
            all_constructs[targeted_rows],
            all_transcripts[targeted_rows],
        )
        source_queries = context["source_query_ids"].astype(str)
        output_queries = context["query_ids"].astype(str)
        output_lookup = {value: index for index, value in enumerate(output_queries)}
        source_columns = np.asarray(
            [index for index, value in enumerate(source_queries) if value in output_lookup],
            dtype=np.int64,
        )
        output_columns = np.asarray(
            [output_lookup[source_queries[index]] for index in source_columns],
            dtype=np.int64,
        )
        normalizer_subset = GemGroupControlNormalizer(
            gem_groups_=normalizer.gem_groups_,
            target_umi_=normalizer.target_umi_,
            control_mean_=normalizer.control_mean_[:, source_columns],
            control_std_=normalizer.control_std_[:, source_columns],
            control_observed_=normalizer.control_observed_[:, source_columns],
            control_counts_=normalizer.control_counts_,
        )
        population_count = table.population_ids.size
        sums = np.zeros((population_count, output_queries.size), dtype=np.float64)
        contributing = np.zeros((population_count, output_queries.size), dtype=np.uint32)
        population_cells = np.bincount(
            table.cell_population_index, minlength=population_count
        ).astype(np.uint32)
        matrix = source["X"]
        for offset in range(0, targeted_rows.size, shard_rows):
            rows = targeted_rows[offset : offset + shard_rows]
            raw = np.asarray(matrix[rows, :], dtype=np.float64)[:, source_columns]
            values, supported = normalizer_subset.transform(
                raw,
                all_umi[rows],
                all_gems[rows],
            )
            memberships = table.cell_population_index[offset : offset + rows.size]
            for local, population in enumerate(memberships):
                local_output = output_columns[supported[local]]
                sums[population, local_output] += values[local, supported[local]]
                contributing[population, local_output] += 1

    observed = contributing > 0
    targets = np.divide(sums, contributing, out=np.zeros_like(sums), where=observed)
    roles = np.asarray([_route(action) for action in table.action_ids])
    record_ids = _string_array(
        [f"{CONTEXT_ID}|{population}" for population in table.population_ids]
    )
    arrays = {
        "targets": targets.astype(np.float32),
        "observed": observed,
        "query_num_cells_filtered": contributing,
        "num_cells_filtered": population_cells,
        "action_ids": _string_array(table.action_ids.tolist()),
        "query_ids": _string_array(output_queries.tolist()),
        "record_ids": record_ids,
        "source_population_ids": _string_array(table.population_ids.tolist()),
        "source_construct_ids": _string_array(table.construct_ids.tolist()),
        "source_transcript_labels": _string_array(table.transcript_labels.tolist()),
        "context_index": np.zeros(population_count, dtype=np.int64),
        "context_ids": np.asarray([CONTEXT_ID]),
        "context_basal_expression": context["context_basal_expression"],
        "context_basal_observed": context["context_basal_observed"],
        "context_value_space": context["context_value_space"],
        "target_value_space": np.asarray(TARGET_VALUE_SPACE),
        "split_role": _string_array(roles.tolist()),
        "split_train": np.flatnonzero(roles == "train").astype(np.int64),
        "split_validation": np.flatnonzero(roles == "validation").astype(np.int64),
        "split_test": np.flatnonzero(roles == "test").astype(np.int64),
        "source_sha256": np.asarray(SOURCE_SHA256),
        "control_normalization_sha256": np.asarray(NORMALIZATION_SHA256),
        "context_descriptor_sha256": np.asarray(CONTEXT_SHA256),
        "num_cells_role": np.asarray("likelihood-only; query counts override record count"),
    }
    _write_npz(output_path, arrays)
    return {
        "schema": "slp.hepg2-control-normalized-molecular/v1",
        "status": "computed-not-scored-not-fitted",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "records": int(population_count),
        "cells": int(population_cells.sum()),
        "queries": int(output_queries.size),
        "endpoint": TARGET_VALUE_SPACE,
        "metricsComputed": False,
        "fittingPerformed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root
        / "data/sources/nadig-2025-gse264667-hepg2-v1/GSE264667_hepg2_raw_singlecell_01.h5ad",
    )
    parser.add_argument(
        "--control-normalization",
        type=Path,
        default=root
        / "data/derived/slp11-human/nadig-hepg2-control-normalization-v1/control-normalization.npz",
    )
    parser.add_argument(
        "--context-descriptor",
        type=Path,
        default=root
        / "data/derived/slp11-human-gwps-fixed-panel-context-v1/nadig-hepg2-fixed-panel-control-context-v1.npz",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--execute-perturbed", action="store_true")
    args = parser.parse_args(argv)
    if args.execute_perturbed:
        if args.output is None:
            parser.error("--output is required with --execute-perturbed")
        result = execute(
            args.source,
            args.control_normalization,
            args.context_descriptor,
            args.output,
            shard_rows=args.shard_rows,
        )
    else:
        result = plan(args.source, args.control_normalization, args.context_descriptor)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.plan_output is not None:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
