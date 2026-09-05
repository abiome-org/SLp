"""Append the distinct K562 genome-scale screen to the human v2 development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import h5py
import human_data_v2 as human
import numpy as np

TAXON = 9606
CONTEXT_ID = "replogle-2022-k562-gwps-day-8"
CONTROL_SUFFIX = "_non-targeting_non-targeting_non-targeting"
ACTION_RE = re.compile(r"_(ENSG[0-9]+)$")
ENSG_RE = re.compile(r"^ENSG[0-9]+$")
QUERY_COUNT = 7_226
QUERY_BYTES = 115_616
QUERY_SHA256 = "645b8d563b440a4b7ab6a3bb42450594b408c4e7cb84e4fe2789a6620174f12c"
OLD_DEVELOPMENT_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
OLD_ACTION_SHA256 = "2884efd414949bfc3c7dc5f376aa69f0470080afdcab255b4a88f67cc53ac9ed"
STATIC_ENTITY_SHA256 = "6f282a37e7aa303e23b9f6c3bf61127c83c850438a7e16740c53e6cf85a5944e"
NORMALIZED = {
    "name": "K562_gwps_normalized_bulk_01.h5ad",
    "bytes": 374_587_922,
    "md5": "a3dfaa94ea8724217f5ecb1e14a5f0c8",
    "sha256": "37e48c474d8b5dead4151f96ea8f5fe7bbe6beb10eeea48685b740c3f74490a2",
    "figshareFileId": 35773217,
}
RAW = {
    "name": "K562_gwps_raw_bulk_01.h5ad",
    "bytes": 374_587_922,
    "md5": "4570b53c9d62ff6df281e622f0350060",
    "sha256": "7cec96b3b76169abbf6b6ab9d10bf00d71d942d89e63292351f745e130b154db",
    "figshareFileId": 35774443,
}


class GpwsDataError(ValueError):
    """The genome-scale source or combination contract is invalid."""


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: dict[str, object]) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or resolved.name != spec["name"]:
        raise GpwsDataError(f"invalid source path for {spec['name']}")
    if (
        resolved.stat().st_size != spec["bytes"]
        or _hash(resolved) != spec["sha256"]
        or _hash(resolved, "md5") != spec["md5"]
    ):
        raise GpwsDataError(f"source byte identity drift for {spec['name']}")
    return resolved


def _strings(dataset: h5py.Dataset, label: str) -> tuple[str, ...]:
    if dataset.ndim != 1 or len(dataset) > 20_000:
        raise GpwsDataError(f"{label} exceeds its bound")
    values = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in dataset[:])
    if any(not item or item != item.strip() for item in values):
        raise GpwsDataError(f"{label} contains an invalid string")
    return values


def _load_ids(path: Path, *, count: int, digest: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest or not payload.endswith(b"\n") or b"\r" in payload:
        raise GpwsDataError(f"stable ID roster byte identity drift: {path}")
    values = tuple(payload.decode("ascii").splitlines())
    if len(values) != count or list(values) != sorted(set(values)):
        raise GpwsDataError(f"stable ID roster count or ordering drift: {path}")
    return values


def _metadata(path: Path, spec: dict[str, object]) -> dict[str, object]:
    with h5py.File(_verify(path, spec), "r") as handle:
        if handle["X"].shape != (11_258, 8_248) or handle["X"].dtype != np.float32:
            raise GpwsDataError("unexpected K562 GWPS matrix contract")
        records = _strings(handle["obs/gene_transcript"], "GWPS gene_transcript")
        genes = _strings(handle["var/gene_id"], "GWPS gene_id")
        cells = np.asarray(handle["obs/num_cells_filtered"][:], dtype=np.float64)
        core = np.asarray(handle["obs/core_control"][:], dtype=np.bool_)
    if len(set(records)) != len(records) or len(set(genes)) != len(genes):
        raise GpwsDataError("GWPS records or genes are duplicated")
    if any(ENSG_RE.fullmatch(item) is None for item in genes):
        raise GpwsDataError("GWPS readouts are not stable unversioned ENSG IDs")
    actions: list[str | None] = []
    unresolved = 0
    for record in records:
        match = ACTION_RE.search(record)
        action = match.group(1) if match else None
        actions.append(action)
        if action is None and not record.endswith(CONTROL_SUFFIX):
            unresolved += 1
    controls = np.asarray([item.endswith(CONTROL_SUFFIX) for item in records])
    core_controls = controls & core
    known = np.asarray([item is not None for item in actions])
    retained = known | core_controls
    if np.any(core & ~controls) or not core_controls.any():
        raise GpwsDataError("GWPS core-control annotations are inconsistent")
    if np.any(~np.isfinite(cells[retained])) or np.any(cells[retained] <= 0):
        raise GpwsDataError("GWPS retained population cell counts are invalid")
    return {
        "records": records,
        "genes": genes,
        "cells": cells,
        "core_controls": core_controls,
        "known": known,
        "actions": tuple(actions),
        "unresolved": unresolved,
    }


def _read_aligned_rows(
    path: Path,
    rows: np.ndarray,
    source_columns: np.ndarray,
    output_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(rows), QUERY_COUNT), dtype=np.float32)
    observed = np.zeros((len(rows), QUERY_COUNT), dtype=np.bool_)
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        for offset in range(0, len(rows), 256):
            selected = rows[offset : offset + 256]
            block = np.asarray(matrix[selected, :], dtype=np.float32)[:, source_columns]
            finite = np.isfinite(block)
            values[offset : offset + len(selected), output_columns] = np.where(
                finite, block, 0.0
            )
            observed[offset : offset + len(selected), output_columns] = finite
    return values, observed


def _read_gwps(
    normalized_path: Path,
    raw_path: Path,
    query_ids: tuple[str, ...],
) -> dict[str, object]:
    normalized = _metadata(normalized_path, NORMALIZED)
    raw = _metadata(raw_path, RAW)
    for key in ("records", "genes", "core_controls", "known", "actions"):
        left, right = normalized[key], raw[key]
        if isinstance(left, np.ndarray):
            same = np.array_equal(left, right)
        else:
            same = left == right
        if not same:
            raise GpwsDataError(f"raw and normalized GWPS {key} metadata disagree")
    genes = normalized["genes"]
    assert isinstance(genes, tuple)
    lookup = {gene: column for column, gene in enumerate(genes)}
    present_ids = tuple(gene for gene in query_ids if gene in lookup)
    source_columns = np.asarray([lookup[gene] for gene in present_ids], dtype=np.int64)
    output_lookup = {gene: column for column, gene in enumerate(query_ids)}
    output_columns = np.asarray([output_lookup[gene] for gene in present_ids], dtype=np.int64)
    known_rows = np.flatnonzero(np.asarray(normalized["known"])).astype(np.int64)
    core_rows = np.flatnonzero(np.asarray(normalized["core_controls"])).astype(np.int64)
    targets, observed = _read_aligned_rows(
        normalized_path, known_rows, source_columns, output_columns
    )
    control_targets, control_observed = _read_aligned_rows(
        normalized_path, core_rows, source_columns, output_columns
    )
    counts = control_observed.sum(0)
    basal = np.divide(
        np.where(control_observed, control_targets, 0.0).sum(0),
        counts,
        out=np.zeros(QUERY_COUNT, dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)

    with h5py.File(raw_path, "r") as handle:
        raw_values = np.asarray(handle["X"][core_rows, :], dtype=np.float64)[
            :, source_columns
        ]
    if not np.isfinite(raw_values).all() or np.any(raw_values < 0):
        raise GpwsDataError("raw GWPS core-control values are invalid")
    denominator = raw_values.sum(1)
    if np.any(denominator <= 0):
        raise GpwsDataError("raw GWPS core control has no source-present query counts")
    raw_transformed = np.log2(1.0 + 10_000.0 * raw_values / denominator[:, None])
    context_basal = np.zeros(QUERY_COUNT, dtype=np.float32)
    context_basal[output_columns] = raw_transformed.mean(0).astype(np.float32)
    actions = normalized["actions"]
    records = normalized["records"]
    cells = np.asarray(normalized["cells"])
    assert isinstance(actions, tuple) and isinstance(records, tuple)
    return {
        "targets": targets,
        "observed": observed,
        "action_ids": np.asarray([actions[row] for row in known_rows]),
        "record_ids": np.asarray(
            [f"{CONTEXT_ID}|{records[row]}" for row in known_rows]
        ),
        "context_index": np.full(len(known_rows), 2, dtype=np.int64),
        "num_cells_filtered": cells[known_rows].astype(np.float32),
        "basal": basal,
        "context_basal": context_basal,
        "control_targets": control_targets,
        "control_observed": control_observed,
        "control_context_index": np.full(len(core_rows), 2, dtype=np.int64),
        "control_num_cells_filtered": cells[core_rows].astype(np.float32),
        "control_record_ids": np.asarray(
            [f"{CONTEXT_ID}|{records[row]}" for row in core_rows]
        ),
        "control_core": np.ones(len(core_rows), dtype=np.bool_),
        "source_present_query_ids": present_ids,
        "unresolved": normalized["unresolved"],
        "all_action_ids": tuple(sorted({item for item in actions if item is not None})),
    }


def _load_old_development(path: Path, query_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    if _hash(path) != OLD_DEVELOPMENT_SHA256:
        raise GpwsDataError("existing human v2 development artifact SHA-256 drift")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if (
        data["query_ids"].tolist() != list(query_ids)
        or data["context_ids"].tolist() != list(human.CONTEXT_IDS)
        or len(data["split_test"])
    ):
        raise GpwsDataError("existing human v2 development contract drift")
    return data


def _bundle(
    combined: dict[str, np.ndarray],
    selection: np.ndarray,
    split_names: np.ndarray,
    query_ids: tuple[str, ...],
    basal: np.ndarray,
    context_basal: np.ndarray,
    controls: dict[str, np.ndarray],
    context_ids: Sequence[str] = (*human.CONTEXT_IDS, CONTEXT_ID),
) -> dict[str, np.ndarray]:
    selected_splits = split_names[selection]
    return {
        "targets": combined["targets"][selection],
        "observed": combined["observed"][selection],
        "action_ids": human._string_array(combined["action_ids"][selection].tolist()),
        "query_ids": human._string_array(query_ids),
        "context_index": combined["context_index"][selection],
        "context_ids": human._string_array(context_ids),
        "basal_control": basal,
        "context_basal_expression": context_basal,
        "context_value_space": np.asarray(human.CONTEXT_VALUE_SPACE),
        "record_ids": human._string_array(combined["record_ids"][selection].tolist()),
        "num_cells_filtered": combined["num_cells_filtered"][selection],
        "target_value_space": np.asarray(human.VALUE_SPACE),
        "num_cells_role": np.asarray(human.NUM_CELLS_ROLE),
        "control_targets": controls["control_targets"],
        "control_observed": controls["control_observed"],
        "control_context_index": controls["control_context_index"],
        "control_num_cells_filtered": controls["control_num_cells_filtered"],
        "control_record_ids": human._string_array(controls["control_record_ids"].tolist()),
        "control_core": controls["control_core"],
        "split_train": np.flatnonzero(selected_splits == "train").astype(np.int64),
        "split_validation": np.flatnonzero(selected_splits == "validation").astype(np.int64),
        "split_test": np.flatnonzero(selected_splits == "test").astype(np.int64),
    }


def build_gwps_development(
    normalized_path: str | Path,
    raw_path: str | Path,
    old_development_path: str | Path,
    query_path: str | Path,
    old_action_path: str | Path,
    static_entity_path: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    query_ids = _load_ids(
        Path(query_path), count=QUERY_COUNT, digest=QUERY_SHA256
    )
    old_actions = frozenset(
        _load_ids(Path(old_action_path), count=2_392, digest=OLD_ACTION_SHA256)
    )
    static_entities = frozenset(
        _load_ids(Path(static_entity_path), count=7_605, digest=STATIC_ENTITY_SHA256)
    )
    old = _load_old_development(Path(old_development_path), query_ids)
    gwps = _read_gwps(Path(normalized_path), Path(raw_path), query_ids)
    split_gwps = np.asarray([human._split_name(str(item)) for item in gwps["action_ids"]])
    gwps_development = np.flatnonzero(split_gwps != "test").astype(np.int64)
    gwps_test = np.flatnonzero(split_gwps == "test").astype(np.int64)

    row_fields = (
        "targets",
        "observed",
        "action_ids",
        "record_ids",
        "context_index",
        "num_cells_filtered",
    )
    combined = {
        field: np.concatenate((old[field], np.asarray(gwps[field])[gwps_development]))
        for field in row_fields
    }
    split_names = np.asarray([human._split_name(str(item)) for item in combined["action_ids"]])
    if np.any(split_names == "test"):
        raise GpwsDataError("held test action remained in combined development rows")
    controls = {
        field: np.concatenate((old[field], np.asarray(gwps[field])))
        for field in (
            "control_targets",
            "control_observed",
            "control_context_index",
            "control_num_cells_filtered",
            "control_record_ids",
            "control_core",
        )
    }
    basal = np.concatenate((old["basal_control"], np.asarray(gwps["basal"])[None, :]))
    context_basal = np.concatenate(
        (old["context_basal_expression"], np.asarray(gwps["context_basal"])[None, :])
    )
    development_selection = np.arange(len(combined["action_ids"]), dtype=np.int64)
    test_combined = {field: np.asarray(gwps[field]) for field in row_fields}
    test_controls = {
        field: np.asarray(gwps[field])
        for field in (
            "control_targets",
            "control_observed",
            "control_context_index",
            "control_num_cells_filtered",
            "control_record_ids",
            "control_core",
        )
    }
    test_controls["control_context_index"] = np.zeros_like(
        test_controls["control_context_index"]
    )
    test_combined["context_index"] = np.zeros_like(test_combined["context_index"])

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    development_path = destination_path / "replogle-k562-rpe1-gwps-author-normalized-development-v3.npz"
    test_path = destination_path / "replogle-k562-gwps-author-normalized-test-only-v3.npz"
    human._write_npz(
        development_path,
        _bundle(
            combined,
            development_selection,
            split_names,
            query_ids,
            basal,
            context_basal,
            controls,
        ),
    )
    # The test file contains only mechanically routed GWPS rows. Existing v2
    # test data is neither opened nor copied by this builder.
    human._write_npz(
        test_path,
        _bundle(
            test_combined,
            gwps_test,
            split_gwps,
            query_ids,
            np.asarray(gwps["basal"])[None, :],
            np.asarray(gwps["context_basal"])[None, :],
            test_controls,
            context_ids=(CONTEXT_ID,),
        ),
    )

    action_roster = tuple(gwps["all_action_ids"])
    action_payload = "".join(f"{item}\n" for item in action_roster).encode("ascii")
    (destination_path / "gwps-action-ids.txt").write_bytes(action_payload)
    present_query_payload = "".join(
        f"{item}\n" for item in gwps["source_present_query_ids"]
    ).encode("ascii")
    (destination_path / "gwps-source-present-query-ids.txt").write_bytes(
        present_query_payload
    )
    action_sets = {
        role: {
            str(action)
            for action in gwps["action_ids"]
            if human._split_name(str(action)) == role
        }
        for role in ("train", "validation", "test")
    }
    train_validation = action_sets["train"] | action_sets["validation"]
    roster_sets = {
        "newTrain": tuple(sorted(action_sets["train"] - old_actions)),
        "newValidation": tuple(sorted(action_sets["validation"] - old_actions)),
        "trainValidationMissingStatic": tuple(
            sorted(train_validation - static_entities)
        ),
    }
    roster_manifests = {}
    roster_names = {
        "newTrain": "gwps-new-train-action-ids.txt",
        "newValidation": "gwps-new-validation-action-ids.txt",
        "trainValidationMissingStatic": "gwps-train-validation-missing-static-ids.txt",
    }
    for label, values in roster_sets.items():
        payload = "".join(f"{item}\n" for item in values).encode("ascii")
        (destination_path / roster_names[label]).write_bytes(payload)
        roster_manifests[label] = {
            "path": roster_names[label],
            "count": len(values),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema": "slp.human-perturbation-author-normalized/v3-gwps-extension",
        "status": "derived-development-and-routed-gwps-test-not-omf-admitted",
        "ncbiTaxon": TAXON,
        "valueSpace": human.VALUE_SPACE,
        "queryPanel": {
            "count": QUERY_COUNT,
            "sha256": QUERY_SHA256,
            "gwpsSourcePresent": len(gwps["source_present_query_ids"]),
            "gwpsSourceAbsent": QUERY_COUNT - len(gwps["source_present_query_ids"]),
            "gwpsSourcePresentRoster": {
                "path": "gwps-source-present-query-ids.txt",
                "bytes": len(present_query_payload),
                "sha256": hashlib.sha256(present_query_payload).hexdigest(),
            },
            "absentRule": "retain columns as observed=false and value zero",
        },
        "contexts": [
            *human.CONTEXT_IDS,
            CONTEXT_ID,
        ],
        "sourceDistinction": {
            "gwps": "K562 genome-scale CRISPRi, day 8 post-transduction",
            "essential": "K562 essential-gene CRISPRi, day 6 post-transduction",
            "sameAssayOrCells": False,
            "evidence": "official Figshare v1 description plus different cell counts for 2262 of 2276 shared population identifiers",
            "sharedPopulationIdentifierSuffixes": 2_276,
            "deduplication": "retain across distinct context IDs; raw/normalized twins are one population and raw contributes context controls only",
        },
        "counts": {
            "combinedDevelopmentRecords": len(combined["action_ids"]),
            "combinedTrainRecords": int(np.count_nonzero(split_names == "train")),
            "combinedValidationRecords": int(np.count_nonzero(split_names == "validation")),
            "gwpsInterventionRecords": len(gwps["action_ids"]),
            "gwpsTrainRecords": int(np.count_nonzero(split_gwps == "train")),
            "gwpsValidationRecords": int(np.count_nonzero(split_gwps == "validation")),
            "gwpsTestRecordsRouted": len(gwps_test),
            "gwpsCoreControls": len(gwps["control_core"]),
            "gwpsUnresolvedRowsQuarantined": gwps["unresolved"],
        },
        "actionCoverage": {
            "gwpsAll": len(action_roster),
            "gwpsRoster": {
                "path": "gwps-action-ids.txt",
                "bytes": len(action_payload),
                "sha256": hashlib.sha256(action_payload).hexdigest(),
            },
            "train": len(action_sets["train"]),
            "validation": len(action_sets["validation"]),
            "test": len(action_sets["test"]),
            "newTrainVersusExistingHuman": len(action_sets["train"] - old_actions),
            "newValidationVersusExistingHuman": len(
                action_sets["validation"] - old_actions
            ),
            "newTrainValidationVersusExistingHuman": len(
                train_validation - old_actions
            ),
            "trainValidationMissingStaticFeatures": len(
                train_validation - static_entities
            ),
            "allMissingStaticFeatures": len(set(action_roster) - static_entities),
            "rosters": roster_manifests,
        },
        "sources": {
            "normalized": NORMALIZED,
            "rawContext": RAW,
            "figshareVersionDoi": "10.25452/figshare.plus.20029387.v1",
        },
        "outputs": {
            "development": {
                "path": development_path.name,
                "bytes": development_path.stat().st_size,
                "sha256": _hash(development_path),
                "contains": ["train", "validation"],
            },
            "gwpsTestOnly": {
                "path": test_path.name,
                "bytes": test_path.stat().st_size,
                "sha256": _hash(test_path),
                "contains": ["test"],
                "access": "sealed; no numeric test analysis performed",
            },
        },
        "oldV2TestArtifactOpened": False,
        "testOutcomeMetricsComputed": False,
        "benchmarkDataAccessed": False,
    }
    manifest_path = destination_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": manifest,
        "manifestPath": str(manifest_path),
        "manifestSha256": _hash(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--old-development", required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--old-action-ids", required=True)
    parser.add_argument("--static-entity-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_gwps_development(
        args.normalized,
        args.raw,
        args.old_development,
        args.query_ids,
        args.old_action_ids,
        args.static_entity_ids,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
