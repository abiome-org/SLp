"""Typed metadata-only contract for molecular intervention observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "slp.action-observation-metadata/v1"
MODES = frozenset({"crispri-repression", "crispra-activation", "gene-deletion"})


class ActionObservationMetadataError(ValueError):
    """A metadata sidecar violates the typed contract."""


@dataclass(frozen=True)
class ActionObservationMetadata:
    source_id: str
    record_ids: np.ndarray
    context_ids: np.ndarray
    context_index: np.ndarray
    action_offsets: np.ndarray
    action_taxon: np.ndarray
    action_ids: np.ndarray
    action_mode: np.ndarray
    action_mode_present: np.ndarray
    action_dose: np.ndarray
    action_dose_present: np.ndarray
    action_efficacy: np.ndarray
    action_efficacy_present: np.ndarray
    exposure_days: np.ndarray
    exposure_days_present: np.ndarray
    construct_ids: np.ndarray
    construct_present: np.ndarray
    replicate_ids: np.ndarray
    replicate_present: np.ndarray
    observation_unit_ids: np.ndarray
    observation_unit_present: np.ndarray
    query_taxon: np.ndarray
    query_ids: np.ndarray
    query_modality: str
    target_value_space: str


def _scalar_string(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = archive[name]
    if value.shape != () or value.dtype.kind not in "US":
        raise ActionObservationMetadataError(f"{name} must be a scalar string")
    result = str(value.item())
    if not result or result != result.strip():
        raise ActionObservationMetadataError(f"{name} is empty or untrimmed")
    return result


def _require_vector(array: np.ndarray, length: int, kind: str, name: str) -> None:
    if array.shape != (length,) or array.dtype.kind not in kind:
        raise ActionObservationMetadataError(f"{name} must have shape ({length},)")


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "schema", "source_id", "source_sha256", "record_ids", "context_ids",
        "context_index", "action_offsets", "action_taxon", "action_ids",
        "action_mode", "action_mode_present", "action_dose", "action_dose_present",
        "action_efficacy", "action_efficacy_present", "exposure_days",
        "exposure_days_present", "construct_ids", "construct_present",
        "replicate_ids", "replicate_present", "observation_unit_ids",
        "observation_unit_present", "query_taxon", "query_ids", "query_modality",
        "target_value_space", "normalization_group", "dose_status", "efficacy_status",
        "construct_role", "time_source_confounded", "guide_ids_model_input_allowed",
    }
    if set(arrays) != required:
        raise ActionObservationMetadataError(
            f"sidecar fields differ: missing={sorted(required-set(arrays))}, "
            f"extra={sorted(set(arrays)-required)}"
        )
    for name in (
        "schema", "source_id", "source_sha256", "query_modality", "target_value_space",
        "normalization_group", "dose_status", "efficacy_status", "construct_role",
    ):
        value = arrays[name]
        if value.shape != () or value.dtype.kind not in "US" or not str(value.item()).strip():
            raise ActionObservationMetadataError(f"{name} must be a nonempty scalar string")
    if str(arrays["schema"].item()) != SCHEMA:
        raise ActionObservationMetadataError("unsupported metadata schema")
    if arrays["source_sha256"].item().__len__() != 64:
        raise ActionObservationMetadataError("source_sha256 must be lowercase SHA-256")
    for name in ("time_source_confounded", "guide_ids_model_input_allowed"):
        if arrays[name].shape != () or arrays[name].dtype != np.bool_:
            raise ActionObservationMetadataError(f"{name} must be scalar bool")
    if bool(arrays["guide_ids_model_input_allowed"].item()):
        raise ActionObservationMetadataError("guide or construct IDs cannot be model inputs")

    records = len(arrays["record_ids"])
    contexts = len(arrays["context_ids"])
    actions = len(arrays["action_ids"])
    queries = len(arrays["query_ids"])
    if not records or not contexts or not actions or not queries:
        raise ActionObservationMetadataError("sidecar axes must be nonempty")
    _require_vector(arrays["record_ids"], records, "US", "record_ids")
    _require_vector(arrays["context_ids"], contexts, "US", "context_ids")
    if len(set(arrays["record_ids"].tolist())) != records:
        raise ActionObservationMetadataError("record IDs must be unique")
    if len(set(arrays["context_ids"].tolist())) != contexts:
        raise ActionObservationMetadataError("context IDs must be unique")
    _require_vector(arrays["context_index"], records, "iu", "context_index")
    if np.any(arrays["context_index"] < 0) or np.any(arrays["context_index"] >= contexts):
        raise ActionObservationMetadataError("context index is out of range")
    offsets = arrays["action_offsets"]
    _require_vector(offsets, records + 1, "iu", "action_offsets")
    if offsets[0] != 0 or offsets[-1] != actions or np.any(np.diff(offsets) < 1):
        raise ActionObservationMetadataError("every record needs one or more action tokens")

    for name, kind in (
        ("action_taxon", "iu"), ("action_ids", "US"), ("action_mode", "US"),
        ("action_mode_present", "b"), ("action_dose", "f"),
        ("action_dose_present", "b"), ("action_efficacy", "f"),
        ("action_efficacy_present", "b"),
    ):
        _require_vector(arrays[name], actions, kind, name)
    if np.any(arrays["action_taxon"] <= 0) or any(not item for item in arrays["action_ids"]):
        raise ActionObservationMetadataError("actions need stable taxon-qualified IDs")
    if not np.all(arrays["action_mode_present"]):
        raise ActionObservationMetadataError("verified intervention mode is required")
    if set(arrays["action_mode"].tolist()) - MODES:
        raise ActionObservationMetadataError("unknown intervention mode")
    for name in ("action_dose", "action_efficacy"):
        values = arrays[name]
        present = arrays[f"{name}_present"]
        if not np.isfinite(values).all() or np.any(values[~present] != 0.0):
            raise ActionObservationMetadataError(f"{name} missing storage must be finite zero")
    if np.any(arrays["action_dose"][arrays["action_dose_present"]] <= 0.0):
        raise ActionObservationMetadataError("present action doses must be positive")

    for name, kind in (
        ("exposure_days", "f"), ("exposure_days_present", "b"),
        ("construct_ids", "US"), ("construct_present", "b"),
        ("replicate_ids", "US"), ("replicate_present", "b"),
        ("observation_unit_ids", "US"), ("observation_unit_present", "b"),
    ):
        _require_vector(arrays[name], records, kind, name)
    if not np.isfinite(arrays["exposure_days"]).all():
        raise ActionObservationMetadataError("exposure days must be finite")
    if np.any(arrays["exposure_days"][~arrays["exposure_days_present"]] != 0.0):
        raise ActionObservationMetadataError("missing exposure storage must be zero")
    if np.any(arrays["exposure_days"][arrays["exposure_days_present"]] <= 0.0):
        raise ActionObservationMetadataError("present exposure days must be positive")
    for value_name, mask_name in (
        ("construct_ids", "construct_present"), ("replicate_ids", "replicate_present"),
        ("observation_unit_ids", "observation_unit_present"),
    ):
        values, present = arrays[value_name], arrays[mask_name]
        if np.any((values != "") != present):
            raise ActionObservationMetadataError(f"{value_name} and mask disagree")

    _require_vector(arrays["query_taxon"], queries, "iu", "query_taxon")
    _require_vector(arrays["query_ids"], queries, "US", "query_ids")
    if np.any(arrays["query_taxon"] <= 0) or any(not item for item in arrays["query_ids"]):
        raise ActionObservationMetadataError("queries need stable taxon-qualified IDs")
    if len(set(zip(arrays["query_taxon"].tolist(), arrays["query_ids"].tolist()))) != queries:
        raise ActionObservationMetadataError("composite query identities must be unique")


def load_action_observation_metadata(path: str | Path) -> ActionObservationMetadata:
    """Load and validate one metadata-only sidecar without pickle support."""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
        validate_arrays(arrays)
        return ActionObservationMetadata(
            source_id=_scalar_string(archive, "source_id"),
            record_ids=arrays["record_ids"], context_ids=arrays["context_ids"],
            context_index=arrays["context_index"], action_offsets=arrays["action_offsets"],
            action_taxon=arrays["action_taxon"], action_ids=arrays["action_ids"],
            action_mode=arrays["action_mode"], action_mode_present=arrays["action_mode_present"],
            action_dose=arrays["action_dose"], action_dose_present=arrays["action_dose_present"],
            action_efficacy=arrays["action_efficacy"],
            action_efficacy_present=arrays["action_efficacy_present"],
            exposure_days=arrays["exposure_days"],
            exposure_days_present=arrays["exposure_days_present"],
            construct_ids=arrays["construct_ids"], construct_present=arrays["construct_present"],
            replicate_ids=arrays["replicate_ids"], replicate_present=arrays["replicate_present"],
            observation_unit_ids=arrays["observation_unit_ids"],
            observation_unit_present=arrays["observation_unit_present"],
            query_taxon=arrays["query_taxon"], query_ids=arrays["query_ids"],
            query_modality=_scalar_string(archive, "query_modality"),
            target_value_space=_scalar_string(archive, "target_value_space"),
        )
