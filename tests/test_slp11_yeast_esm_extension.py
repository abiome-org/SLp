from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extend_slp11_yeast_esm_features",
    ROOT / "scripts" / "extend_slp11_yeast_esm_features.py",
)
assert SPEC and SPEC.loader
EXTEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTEND)


def keyed(ids: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "feature_values": values.astype(np.float32),
        "entity_taxon": np.full(len(ids), 4932, dtype=np.int64),
        "entity_id": np.asarray(ids),
    }


def test_extended_esm_preserves_old_rows_bit_exact() -> None:
    old = keyed(["SGD:S1", "UniProtKB:P1"], np.arange(640).reshape(2, 320))
    new = keyed(["SGD:S0", "SGD:S2"], np.arange(640, 1280).reshape(2, 320))
    result = EXTEND.extend_esm(old, new)
    mapping = EXTEND.feature_map(result, 320)
    old_mapping = EXTEND.feature_map(old, 320)
    for key, vector in old_mapping.items():
        assert np.array_equal(mapping[key], vector)


def test_extended_esm_rejects_overlapping_identity() -> None:
    old = keyed(["SGD:S1"], np.zeros((1, 320)))
    new = keyed(["SGD:S1"], np.ones((1, 320)))
    with pytest.raises(EXTEND.ExtensionError, match="overlaps"):
        EXTEND.extend_esm(old, new)


def test_profile_selection_is_length_stratified_and_deterministic() -> None:
    peptides = {
        "SGD:S1": b"M",
        "SGD:S2": b"MM",
        "SGD:S3": b"MMM",
        "SGD:S4": b"MMMM",
        "SGD:S5": b"MMMMM",
    }
    assert EXTEND.select_profile_ids(list(reversed(peptides)), peptides, 3) == [
        "SGD:S1",
        "SGD:S3",
        "SGD:S5",
    ]


def test_chunk_recipe_covers_long_protein_without_truncation() -> None:
    windows = EXTEND.SOURCE.chunk_windows(2500, 1022, 128)
    coverage, weights = EXTEND.SOURCE.window_inverse_coverage(2500, windows)
    assert coverage.min() == 1
    reconstructed = np.zeros(2500, dtype=np.float64)
    for (start, end), weight in zip(windows, weights, strict=True):
        reconstructed[start:end] += weight
    np.testing.assert_array_equal(reconstructed, 1.0)
