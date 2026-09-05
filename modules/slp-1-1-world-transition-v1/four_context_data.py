"""Fail-closed merge of Replogle development with retired HepG2 development."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

Array = np.ndarray
TAXON = 9606
SEED = 731
HEPG2_CONTEXT = "nadig-2025-hepg2-day-7"
MIXED_TARGET_SPACE = (
    "heterogeneous-context-indexed-see-target_value_space_by_context-v1"
)
ENSG = re.compile(r"^ENSG[0-9]+$")


class FourContextDataError(ValueError):
    """Raised when a merge input violates the frozen development contract."""


def split_name(action_id: str, seed: int = SEED) -> str:
    payload = f"slp11-development-v1|{seed}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def _strings(values: Array | list[str]) -> Array:
    items = np.asarray(values).astype(str).tolist()
    return np.asarray(items, dtype=f"<U{max(1, *(len(item) for item in items))}")


def _require_keys(source: object, required: set[str], label: str) -> None:
    available = set(source.files)  # type: ignore[attr-defined]
    missing = required - available
    if missing:
        raise FourContextDataError(f"{label} missing fields: {sorted(missing)}")


def _same(left: Array, right: Array, label: str) -> None:
    if (
        left.dtype != right.dtype
        or left.shape != right.shape
        or not np.array_equal(left, right)
    ):
        raise FourContextDataError(f"{label} changed")


def build_four_context_arrays(
    replogle: object,
    hepg2: object,
    normalization: object,
) -> tuple[dict[str, Array], dict[str, object]]:
    """Return a four-context development bundle and a mechanical audit."""

    row_fields = {
        "targets",
        "observed",
        "action_ids",
        "context_index",
        "record_ids",
        "num_cells_filtered",
    }
    control_fields = {
        "control_targets",
        "control_observed",
        "control_context_index",
        "control_num_cells_filtered",
        "control_record_ids",
        "control_core",
    }
    _require_keys(
        replogle,
        row_fields
        | control_fields
        | {
            "query_ids",
            "context_ids",
            "basal_control",
            "context_basal_expression",
            "context_basal_observed",
            "context_value_space",
            "target_value_space",
            "num_cells_role",
            "split_train",
            "split_validation",
            "split_test",
        },
        "Replogle",
    )
    _require_keys(
        hepg2,
        row_fields
        | {
            "query_ids",
            "query_num_cells_filtered",
            "context_ids",
            "context_basal_expression",
            "context_basal_observed",
            "context_value_space",
            "target_value_space",
            "source_population_ids",
            "source_construct_ids",
            "source_transcript_labels",
            "split_role",
            "split_train",
            "split_validation",
            "split_test",
            "source_sha256",
            "control_normalization_sha256",
            "context_descriptor_sha256",
        },
        "HepG2",
    )
    _require_keys(
        normalization,
        {
            "query_ids",
            "gem_groups",
            "control_mean",
            "control_std",
            "control_observed",
            "control_counts",
            "value_space",
            "fit_provenance",
        },
        "HepG2 normalization",
    )

    query_ids = replogle["query_ids"].astype(str)
    if query_ids.shape != (7036,) or not np.array_equal(
        query_ids, hepg2["query_ids"].astype(str)
    ):
        raise FourContextDataError("query identity/order mismatch")
    if len(set(query_ids)) != len(query_ids) or any(
        ENSG.fullmatch(value) is None for value in query_ids
    ):
        raise FourContextDataError("query IDs must be unique stable ENSG IDs")
    replogle_contexts = replogle["context_ids"].astype(str)
    if replogle_contexts.shape != (3,) or hepg2["context_ids"].astype(str).tolist() != [
        HEPG2_CONTEXT
    ]:
        raise FourContextDataError("context contract drift")
    if str(replogle["context_value_space"].item()) != str(
        hepg2["context_value_space"].item()
    ):
        raise FourContextDataError("basal context value spaces differ")

    hepg2_actions = hepg2["action_ids"].astype(str)
    computed_roles = np.asarray([split_name(action) for action in hepg2_actions])
    if not np.array_equal(computed_roles, hepg2["split_role"].astype(str)):
        raise FourContextDataError(
            "HepG2 stored roles disagree with global action hash"
        )
    selected = np.flatnonzero(computed_roles != "test")
    excluded = np.flatnonzero(computed_roles == "test")
    if not (
        np.array_equal(np.flatnonzero(computed_roles == "train"), hepg2["split_train"])
        and np.array_equal(
            np.flatnonzero(computed_roles == "validation"), hepg2["split_validation"]
        )
        and np.array_equal(excluded, hepg2["split_test"])
    ):
        raise FourContextDataError("HepG2 split indices drifted")
    if any(ENSG.fullmatch(value) is None for value in hepg2_actions):
        raise FourContextDataError("HepG2 action IDs must be stable ENSG IDs")
    if len(replogle["split_test"]):
        raise FourContextDataError("Replogle development contains test rows")

    replogle_rows = len(replogle["action_ids"])
    combined_hepg2_index = np.arange(
        replogle_rows, replogle_rows + len(selected), dtype=np.int64
    )
    hepg2_observed = hepg2["observed"][selected]
    hepg2_query_counts = hepg2["query_num_cells_filtered"][selected]
    if not np.array_equal(hepg2_observed, hepg2_query_counts > 0):
        raise FourContextDataError("HepG2 observed mask and query counts disagree")

    norm_query_ids = normalization["query_ids"].astype(str)
    norm_lookup = {value: index for index, value in enumerate(norm_query_ids)}
    if len(norm_lookup) != len(norm_query_ids):
        raise FourContextDataError("normalization query IDs are not unique")
    output_columns = np.asarray(
        [index for index, value in enumerate(query_ids) if value in norm_lookup],
        dtype=np.int64,
    )
    norm_columns = np.asarray(
        [norm_lookup[query_ids[index]] for index in output_columns], dtype=np.int64
    )
    gem_count = len(normalization["gem_groups"])
    norm_mean = np.zeros((gem_count, len(query_ids)), dtype=np.float32)
    norm_std = np.zeros((gem_count, len(query_ids)), dtype=np.float32)
    norm_observed = np.zeros((gem_count, len(query_ids)), dtype=bool)
    norm_mean[:, output_columns] = normalization["control_mean"][:, norm_columns]
    norm_std[:, output_columns] = normalization["control_std"][:, norm_columns]
    norm_observed[:, output_columns] = normalization["control_observed"][
        :, norm_columns
    ]
    if not np.isfinite(norm_mean).all() or not np.isfinite(norm_std).all():
        raise FourContextDataError("HepG2 normalization contains nonfinite values")

    blank = np.full(replogle_rows, "", dtype="<U1")
    hepg2_target_space = str(hepg2["target_value_space"].item())
    replogle_target_space = str(replogle["target_value_space"].item())
    basal_hepg2_observed = np.any(hepg2_observed, axis=0)
    if not np.array_equal(basal_hepg2_observed, hepg2["context_basal_observed"][0]):
        raise FourContextDataError("HepG2 target and basal support masks differ")
    if not np.array_equal(np.any(norm_observed, axis=0), basal_hepg2_observed):
        raise FourContextDataError("HepG2 normalization and molecular support differ")

    arrays: dict[str, Array] = {
        "targets": np.concatenate(
            [replogle["targets"], hepg2["targets"][selected]], axis=0
        ),
        "observed": np.concatenate([replogle["observed"], hepg2_observed], axis=0),
        "action_ids": _strings(
            np.concatenate(
                [replogle["action_ids"].astype(str), hepg2_actions[selected]]
            )
        ),
        "action_taxon": np.full(replogle_rows + len(selected), TAXON, dtype=np.int64),
        "query_ids": _strings(query_ids),
        "query_taxon": np.full(len(query_ids), TAXON, dtype=np.int64),
        "context_index": np.concatenate(
            [replogle["context_index"], np.full(len(selected), 3, dtype=np.int64)]
        ),
        "context_ids": _strings(np.concatenate([replogle_contexts, [HEPG2_CONTEXT]])),
        "source_index": np.concatenate(
            [replogle["context_index"], np.full(len(selected), 3, dtype=np.int64)]
        ),
        "source_ids": _strings(
            [
                "replogle-2022-k562-essential",
                "replogle-2022-rpe1-essential",
                "replogle-2022-k562-gwps",
                "nadig-2025-hepg2",
            ]
        ),
        "record_ids": _strings(
            np.concatenate(
                [
                    replogle["record_ids"].astype(str),
                    hepg2["record_ids"].astype(str)[selected],
                ]
            )
        ),
        "num_cells_filtered": np.concatenate(
            [
                replogle["num_cells_filtered"],
                hepg2["num_cells_filtered"][selected].astype(np.float32),
            ]
        ),
        "num_cells_role": replogle["num_cells_role"],
        "hepg2_query_num_cells_filtered": hepg2_query_counts,
        "hepg2_combined_row_index": combined_hepg2_index,
        "basal_control": np.concatenate(
            [
                replogle["basal_control"],
                np.zeros((1, len(query_ids)), dtype=np.float32),
            ],
            axis=0,
        ),
        "basal_control_observed": np.concatenate(
            [
                np.ones_like(replogle["basal_control"], dtype=bool),
                basal_hepg2_observed[None, :],
            ],
            axis=0,
        ),
        "context_basal_expression": np.concatenate(
            [replogle["context_basal_expression"], hepg2["context_basal_expression"]],
            axis=0,
        ),
        "context_basal_observed": np.concatenate(
            [replogle["context_basal_observed"], hepg2["context_basal_observed"]],
            axis=0,
        ),
        "context_value_space": replogle["context_value_space"],
        "target_value_space": np.asarray(MIXED_TARGET_SPACE),
        "target_value_space_by_context": _strings(
            [
                replogle_target_space,
                replogle_target_space,
                replogle_target_space,
                hepg2_target_space,
            ]
        ),
        "split_train": np.concatenate(
            [
                replogle["split_train"],
                combined_hepg2_index[computed_roles[selected] == "train"],
            ]
        ),
        "split_validation": np.concatenate(
            [
                replogle["split_validation"],
                combined_hepg2_index[computed_roles[selected] == "validation"],
            ]
        ),
        "split_test": np.asarray([], dtype=np.int64),
        "source_population_ids": _strings(
            np.concatenate(
                [blank, hepg2["source_population_ids"].astype(str)[selected]]
            )
        ),
        "source_population_ids_observed": np.concatenate(
            [np.zeros(replogle_rows, dtype=bool), np.ones(len(selected), dtype=bool)]
        ),
        "source_construct_ids": _strings(
            np.concatenate([blank, hepg2["source_construct_ids"].astype(str)[selected]])
        ),
        "source_construct_ids_observed": np.concatenate(
            [np.zeros(replogle_rows, dtype=bool), np.ones(len(selected), dtype=bool)]
        ),
        "source_transcript_labels": _strings(
            np.concatenate(
                [blank, hepg2["source_transcript_labels"].astype(str)[selected]]
            )
        ),
        "source_transcript_labels_observed": np.concatenate(
            [np.zeros(replogle_rows, dtype=bool), np.ones(len(selected), dtype=bool)]
        ),
        "control_targets": replogle["control_targets"],
        "control_observed": replogle["control_observed"],
        "control_context_index": replogle["control_context_index"],
        "control_num_cells_filtered": replogle["control_num_cells_filtered"],
        "control_record_ids": replogle["control_record_ids"],
        "control_core": replogle["control_core"],
        "control_target_pseudobulks_available_by_context": np.asarray(
            [True, True, True, False]
        ),
        "hepg2_control_gem_groups": normalization["gem_groups"],
        "hepg2_control_cells_per_gem": normalization["control_counts"],
        "hepg2_control_mean": norm_mean,
        "hepg2_control_std": norm_std,
        "hepg2_control_observed": norm_observed,
        "hepg2_control_normalization_value_space": normalization["value_space"],
        "hepg2_control_fit_provenance": normalization["fit_provenance"],
        "hepg2_control_stats_compatible_with_control_targets": np.asarray(False),
        "hepg2_excluded_test_records": np.asarray(len(excluded), dtype=np.int64),
        "hepg2_source_sha256": hepg2["source_sha256"],
        "hepg2_control_normalization_sha256": hepg2["control_normalization_sha256"],
        "hepg2_context_descriptor_sha256": hepg2["context_descriptor_sha256"],
    }
    # Role vector follows combined row order; Replogle source is already train then validation by index masks,
    # not necessarily by physical row order, so fill it from indices rather than concatenating labels.
    role = np.empty(len(arrays["action_ids"]), dtype="<U10")
    role[arrays["split_train"]] = "train"
    role[arrays["split_validation"]] = "validation"
    arrays["split_role"] = role

    audit = audit_replogle_preservation(replogle, arrays)
    train_actions = set(arrays["action_ids"][arrays["split_train"]].tolist())
    validation_actions = set(arrays["action_ids"][arrays["split_validation"]].tolist())
    if train_actions & validation_actions:
        raise FourContextDataError("action identity overlaps train and validation")
    audit.update(
        {
            "records": len(arrays["action_ids"]),
            "train_records": len(arrays["split_train"]),
            "validation_records": len(arrays["split_validation"]),
            "hepg2_selected_records": len(selected),
            "hepg2_excluded_test_records": len(excluded),
            "hepg2_observed_queries": int(basal_hepg2_observed.sum()),
            "train_validation_action_overlap": 0,
        }
    )
    return arrays, audit


def audit_replogle_preservation(
    replogle: object, merged: dict[str, Array]
) -> dict[str, object]:
    """Require exact Replogle prefixes and unchanged control/context/query values."""

    rows = len(replogle["action_ids"])
    for field in (
        "targets",
        "observed",
        "action_ids",
        "context_index",
        "record_ids",
        "num_cells_filtered",
    ):
        _same(replogle[field], merged[field][:rows], f"Replogle {field} prefix")
    for field in ("query_ids", "context_value_space", "num_cells_role"):
        _same(replogle[field], merged[field], f"Replogle {field}")
    for field in (
        "context_ids",
        "basal_control",
        "context_basal_expression",
        "context_basal_observed",
    ):
        _same(replogle[field], merged[field][:3], f"Replogle {field} prefix")
    for field in (
        "control_targets",
        "control_observed",
        "control_context_index",
        "control_num_cells_filtered",
        "control_record_ids",
        "control_core",
    ):
        _same(replogle[field], merged[field], f"Replogle {field}")
    _same(
        replogle["split_train"],
        merged["split_train"][: len(replogle["split_train"])],
        "train indices",
    )
    _same(
        replogle["split_validation"],
        merged["split_validation"][: len(replogle["split_validation"])],
        "validation indices",
    )
    return {
        "replogle_prefix_records_exact": rows,
        "replogle_query_axis_exact": True,
        "replogle_context_prefix_exact": True,
        "replogle_control_arrays_exact": True,
        "replogle_split_index_prefixes_exact": True,
        "replogle_target_value_space_preserved_by_context": str(
            replogle["target_value_space"].item()
        ),
    }


def write_npz(path: Path, arrays: dict[str, Array]) -> None:
    """Write deterministic, pickle-free compressed NPZ."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for name in sorted(arrays):
                stream = io.BytesIO()
                value = arrays[name]
                payload = value if value.ndim == 0 else np.ascontiguousarray(value)
                np.save(stream, payload, allow_pickle=False)
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
