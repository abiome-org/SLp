from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/slp-1-1-world-transition-v1"))

import action_observation_metadata_v2 as v2  # noqa: E402
from test_slp11_action_observation_metadata import fixture  # noqa: E402


def v2_fixture() -> dict[str, np.ndarray]:
    arrays = fixture()
    arrays.update(
        schema=np.asarray(v2.SCHEMA),
        numeric_transform=np.asarray("per-cell CP10K then log2(1+x) then control z-score"),
        assay_head_id=np.asarray("rna-example-head-v1"),
        assay_head_routing=np.asarray("mechanical source plus target value-space"),
        assay_head_separation_required=np.asarray(True, dtype=np.bool_),
        mode_source_confounded=np.asarray(True, dtype=np.bool_),
    )
    return arrays


def test_v2_requires_explicit_numeric_transform_and_assay_head() -> None:
    arrays = v2_fixture()
    v2.validate_arrays(arrays)
    assert "log2(1+x)" in str(arrays["numeric_transform"].item())


def test_v2_rejects_normalization_equivalence_by_shared_head() -> None:
    arrays = v2_fixture()
    arrays["assay_head_separation_required"] = np.asarray(False, dtype=np.bool_)
    with pytest.raises(v2.ActionObservationMetadataV2Error, match="separate assay heads"):
        v2.validate_arrays(arrays)


def test_v2_retains_mode_source_confounding() -> None:
    arrays = v2_fixture()
    arrays["mode_source_confounded"] = np.asarray(False, dtype=np.bool_)
    with pytest.raises(v2.ActionObservationMetadataV2Error, match="confounding"):
        v2.validate_arrays(arrays)
