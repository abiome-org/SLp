from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/slp-1-1-world-transition-v1"))

from action_observation_metadata import (  # noqa: E402
    ActionObservationMetadataError,
    load_action_observation_metadata,
    validate_arrays,
)


def fixture() -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray("slp.action-observation-metadata/v1"),
        "source_id": np.asarray("fixture:source"),
        "source_sha256": np.asarray("a" * 64),
        "record_ids": np.asarray(["r0", "r1"]),
        "context_ids": np.asarray(["c0"]),
        "context_index": np.asarray([0, 0], dtype=np.int64),
        "action_offsets": np.asarray([0, 1, 3], dtype=np.int64),
        "action_taxon": np.asarray([9606, 9606, 9606], dtype=np.int64),
        "action_ids": np.asarray(["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]),
        "action_mode": np.asarray(["crispri-repression"] * 3),
        "action_mode_present": np.ones(3, dtype=np.bool_),
        "action_dose": np.zeros(3, dtype=np.float32),
        "action_dose_present": np.zeros(3, dtype=np.bool_),
        "action_efficacy": np.zeros(3, dtype=np.float32),
        "action_efficacy_present": np.zeros(3, dtype=np.bool_),
        "exposure_days": np.asarray([6.0, 6.0], dtype=np.float32),
        "exposure_days_present": np.ones(2, dtype=np.bool_),
        "construct_ids": np.asarray(["g0", "g1"]),
        "construct_present": np.ones(2, dtype=np.bool_),
        "replicate_ids": np.asarray(["", ""]),
        "replicate_present": np.zeros(2, dtype=np.bool_),
        "observation_unit_ids": np.asarray(["", ""]),
        "observation_unit_present": np.zeros(2, dtype=np.bool_),
        "query_taxon": np.asarray([9606, 9606], dtype=np.int64),
        "query_ids": np.asarray(["ENSG00000000004", "ENSG00000000005"]),
        "query_modality": np.asarray("pseudobulk-RNA"),
        "target_value_space": np.asarray("control-zscore"),
        "normalization_group": np.asarray("fixture-normalization"),
        "dose_status": np.asarray("unknown"),
        "efficacy_status": np.asarray("post-intervention-forbidden"),
        "construct_role": np.asarray("audit-only"),
        "time_source_confounded": np.asarray(True, dtype=np.bool_),
        "guide_ids_model_input_allowed": np.asarray(False, dtype=np.bool_),
    }


def test_contract_accepts_single_and_double_actions_with_explicit_missingness() -> None:
    arrays = fixture()
    validate_arrays(arrays)
    assert np.diff(arrays["action_offsets"]).tolist() == [1, 2]
    assert not arrays["action_dose_present"].any()
    assert not arrays["action_efficacy_present"].any()


def test_contract_rejects_nonzero_storage_for_missing_strength() -> None:
    arrays = fixture()
    arrays["action_dose"][0] = 1.0
    with pytest.raises(ActionObservationMetadataError, match="missing storage"):
        validate_arrays(arrays)


def test_contract_rejects_guide_ids_as_predictor_inputs() -> None:
    arrays = fixture()
    arrays["guide_ids_model_input_allowed"] = np.asarray(True, dtype=np.bool_)
    with pytest.raises(ActionObservationMetadataError, match="cannot be model inputs"):
        validate_arrays(arrays)


def test_contract_rejects_empty_action_sets() -> None:
    arrays = fixture()
    arrays["action_offsets"] = np.asarray([0, 0, 3], dtype=np.int64)
    with pytest.raises(ActionObservationMetadataError, match="one or more action"):
        validate_arrays(arrays)


def test_loader_disables_pickle_and_preserves_composite_query_identity(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.npz"
    np.savez(path, **fixture())
    loaded = load_action_observation_metadata(path)
    assert loaded.source_id == "fixture:source"
    assert loaded.action_offsets.tolist() == [0, 1, 3]
    assert list(zip(loaded.query_taxon, loaded.query_ids)) == [
        (9606, "ENSG00000000004"), (9606, "ENSG00000000005")
    ]
