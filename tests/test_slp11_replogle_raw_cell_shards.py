from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_replogle_k562_essential_raw_cell_shards.py"
SPEC = importlib.util.spec_from_file_location("replogle_raw_shards", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_role_masks_exclude_test_and_separate_reconstruction() -> None:
    routing = {
        "intervention_role": np.asarray(["train", "train", "control", "control", "validation", "test-excluded"]),
        "reconstruction_role": np.asarray(["train", "validation", "train", "validation", "none", "none"]),
        "is_control": np.asarray([False, False, True, True, False, False]),
    }
    masks = MOD.role_masks(routing)
    assert {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()} == {
        "fit": [0], "control": [2], "reconstruction-held": [1, 3], "development-validation": [4]
    }
    assert not any(mask[5] for mask in masks.values())


def test_raw_validation_requires_integer_counts_and_full_library_match() -> None:
    counts, library = MOD.validate_raw_block(
        np.asarray([[0, 2, 3], [1, 0, 0]], dtype=np.float32), np.asarray([5, 1], dtype=np.float32)
    )
    assert counts.dtype == np.int32
    assert library.tolist() == [5, 1]
    for invalid, expected in [
        (np.asarray([[0, -1]], np.float32), np.asarray([-1], np.float32)),
        (np.asarray([[0, 1.5]], np.float32), np.asarray([1.5], np.float32)),
    ]:
        try:
            MOD.validate_raw_block(invalid, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid raw count block was accepted")
    _, differing_library = MOD.validate_raw_block(
        np.asarray([[0, 2]], np.float32), np.asarray([3], np.float32)
    )
    comparison = MOD.umi_comparison(differing_library, np.asarray([3], np.float32))
    assert comparison["obsGreaterRows"] == 1
    assert comparison["maximumObsMinusRetained"] == 1


def test_cp10k_moments_are_additive_and_group_exactly() -> None:
    counts = sparse.csr_matrix(np.asarray([[1, 1], [2, 0], [0, 4]], dtype=np.int32))
    result = MOD.cp10k_group_moments(
        counts, np.asarray([2, 2, 4]), np.asarray(["ENSG1", "ENSG1", "ENSG2"]), np.asarray([1, 1, 2])
    )
    sums = sparse.csr_matrix((result["sum_data"], result["sum_indices"], result["sum_indptr"]), shape=tuple(result["sum_shape"])).toarray()
    sums2 = sparse.csr_matrix((result["sum_squares_data"], result["sum_squares_indices"], result["sum_squares_indptr"]), shape=tuple(result["sum_squares_shape"])).toarray()
    assert result["action_ids"].tolist() == ["ENSG1", "ENSG2"]
    assert result["gem_group"].tolist() == [1, 2]
    assert result["num_cells"].tolist() == [2, 1]
    np.testing.assert_allclose(sums, [[15000, 5000], [0, 10000]])
    np.testing.assert_allclose(sums2, [[125_000_000, 25_000_000], [0, 100_000_000]])


def test_deterministic_npz_is_pickle_free(tmp_path: Path) -> None:
    arrays = {"a": np.asarray([1, 2], np.int32), "s": np.asarray(["x", "yy"])}
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    MOD.write_npz(first, arrays)
    MOD.write_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive["s"].tolist() == ["x", "yy"]
