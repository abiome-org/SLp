"""V2 action metadata contract with explicit assay-head routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import action_observation_metadata as v1


SCHEMA = "slp.action-observation-metadata/v2"
EXTRA_FIELDS = {
    "numeric_transform", "assay_head_id", "assay_head_routing",
    "assay_head_separation_required", "mode_source_confounded",
}


class ActionObservationMetadataV2Error(ValueError):
    """A v2 metadata sidecar violates assay routing or transform metadata."""


@dataclass(frozen=True)
class ActionObservationMetadataV2:
    core: v1.ActionObservationMetadata
    numeric_transform: str
    assay_head_id: str
    assay_head_routing: str
    assay_head_separation_required: bool
    mode_source_confounded: bool


def _scalar_string(arrays: dict[str, np.ndarray], name: str) -> str:
    value = arrays[name]
    if value.shape != () or value.dtype.kind not in "US" or not str(value.item()).strip():
        raise ActionObservationMetadataV2Error(f"{name} must be a nonempty scalar string")
    return str(value.item())


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Validate the v1 typed axes plus explicit normalization/head metadata."""

    if set(arrays).issuperset(EXTRA_FIELDS | {"schema"}) is False:
        raise ActionObservationMetadataV2Error("v2 assay metadata fields are incomplete")
    if _scalar_string(arrays, "schema") != SCHEMA:
        raise ActionObservationMetadataV2Error("unsupported v2 metadata schema")
    legacy = {name: value for name, value in arrays.items() if name not in EXTRA_FIELDS}
    legacy["schema"] = np.asarray(v1.SCHEMA)
    try:
        v1.validate_arrays(legacy)
    except v1.ActionObservationMetadataError as error:
        raise ActionObservationMetadataV2Error(str(error)) from error
    for name in ("numeric_transform", "assay_head_id", "assay_head_routing"):
        _scalar_string(arrays, name)
    for name in ("assay_head_separation_required", "mode_source_confounded"):
        value = arrays[name]
        if value.shape != () or value.dtype != np.bool_:
            raise ActionObservationMetadataV2Error(f"{name} must be scalar bool")
    if not bool(arrays["assay_head_separation_required"].item()):
        raise ActionObservationMetadataV2Error("unmatched value spaces require separate assay heads")
    if not bool(arrays["mode_source_confounded"].item()):
        raise ActionObservationMetadataV2Error("current corpora must retain mode/source confounding")


def load_action_observation_metadata_v2(path: str | Path) -> ActionObservationMetadataV2:
    """Load a v2 sidecar without pickle support."""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_arrays(arrays)
    legacy = {name: value for name, value in arrays.items() if name not in EXTRA_FIELDS}
    legacy["schema"] = np.asarray(v1.SCHEMA)
    # Construct the typed v1 core directly after validation to avoid a temporary archive.
    core = v1.ActionObservationMetadata(
        source_id=str(legacy["source_id"].item()), record_ids=legacy["record_ids"],
        context_ids=legacy["context_ids"], context_index=legacy["context_index"],
        action_offsets=legacy["action_offsets"], action_taxon=legacy["action_taxon"],
        action_ids=legacy["action_ids"], action_mode=legacy["action_mode"],
        action_mode_present=legacy["action_mode_present"], action_dose=legacy["action_dose"],
        action_dose_present=legacy["action_dose_present"],
        action_efficacy=legacy["action_efficacy"],
        action_efficacy_present=legacy["action_efficacy_present"],
        exposure_days=legacy["exposure_days"],
        exposure_days_present=legacy["exposure_days_present"],
        construct_ids=legacy["construct_ids"], construct_present=legacy["construct_present"],
        replicate_ids=legacy["replicate_ids"], replicate_present=legacy["replicate_present"],
        observation_unit_ids=legacy["observation_unit_ids"],
        observation_unit_present=legacy["observation_unit_present"],
        query_taxon=legacy["query_taxon"], query_ids=legacy["query_ids"],
        query_modality=str(legacy["query_modality"].item()),
        target_value_space=str(legacy["target_value_space"].item()),
    )
    return ActionObservationMetadataV2(
        core=core, numeric_transform=str(arrays["numeric_transform"].item()),
        assay_head_id=str(arrays["assay_head_id"].item()),
        assay_head_routing=str(arrays["assay_head_routing"].item()),
        assay_head_separation_required=bool(arrays["assay_head_separation_required"].item()),
        mode_source_confounded=bool(arrays["mode_source_confounded"].item()),
    )
