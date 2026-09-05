from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_slp11_yeast_shared_static_features",
    ROOT / "scripts" / "build_slp11_yeast_shared_static_features.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_assemble_features_preserves_blocks_and_missing_semantics() -> None:
    ids = ["SGD:S000000001", "SGD:S000000002", "SGD:S000000003"]
    esm = {ids[0]: np.arange(320, dtype=np.float32)}
    go = {
        ids[0]: np.arange(256, dtype=np.float32),
        ids[1]: np.ones(256, dtype=np.float32),
    }
    arrays = BUILDER.assemble_features(
        ids,
        esm,
        go,
        {ids[0]: True, ids[1]: False},
        {ids[0]},
        {ids[0], ids[2]},
        {ids[0], ids[1]},
        {ids[1], ids[2]},
    )
    values = arrays["feature_values"]
    assert values.shape == (3, 577)
    np.testing.assert_array_equal(values[0, :320], esm[ids[0]])
    assert values[0, 320] == 1.0
    np.testing.assert_array_equal(values[0, 321:], go[ids[0]])
    np.testing.assert_array_equal(values[1, :321], 0.0)
    np.testing.assert_array_equal(values[1, 321:], go[ids[1]])
    np.testing.assert_array_equal(values[2], 0.0)
    np.testing.assert_array_equal(arrays["esm_present"], [True, False, False])
    np.testing.assert_array_equal(
        arrays["go_identity_present"], [True, True, False]
    )
    np.testing.assert_array_equal(
        arrays["go_direct_annotation_present"], [True, False, False]
    )
    np.testing.assert_array_equal(
        arrays["pinned_source_sequence_available"], [True, False, True]
    )


def test_requires_unique_sorted_identity_and_deterministic_npz() -> None:
    with pytest.raises(BUILDER.FeatureBuildError, match="unique and sorted"):
        BUILDER.assemble_features(
            ["SGD:S2", "SGD:S1"], {}, {}, {}, set(), set(), set(), set()
        )
    arrays = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    assert BUILDER.deterministic_npz(arrays) == BUILDER.deterministic_npz(arrays)
