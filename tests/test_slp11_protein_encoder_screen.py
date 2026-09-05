from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "protein_encoder_screen", ROOT / "scripts/run_slp11_protein_encoder_screen.py")
assert SPEC is not None and SPEC.loader is not None
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


def test_feature_alignment_uses_taxonomy_and_exact_query_order(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez(path, entity_taxon=[4932, 9606, 9606], entity_id=["A", "A", "B"],
             feature_values=np.asarray([[99., 99.], [1., 2.], [3., 4.]], dtype=np.float32))
    np.testing.assert_array_equal(SCREEN.aligned_features(path, np.asarray(["B", "A", "B"])),
                                  [[3., 4.], [1., 2.], [3., 4.]])


def test_duplicate_feature_identity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    np.savez(path, entity_taxon=[9606, 9606], entity_id=["A", "A"],
             feature_values=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate feature identities"):
        SCREEN.aligned_features(path, np.asarray(["A"]))


def test_missing_human_feature_cannot_fall_back_to_other_species(tmp_path: Path) -> None:
    path = tmp_path / "other_species.npz"
    np.savez(path, entity_taxon=[4932], entity_id=["A"], feature_values=np.ones((1, 2)))
    with pytest.raises(KeyError):
        SCREEN.aligned_features(path, np.asarray(["A"]))
