from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_replogle_rpe1_essential_raw_cell_shards.py"
SPEC = importlib.util.spec_from_file_location("rpe1_raw_shards", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_role_masks_exclude_development_test_and_unresolved() -> None:
    routing = {
        "intervention_role": np.asarray(["train", "train", "control", "control", "validation", "test-excluded", "unresolved-excluded"]),
        "reconstruction_role": np.asarray(["train", "validation", "train", "validation", "none", "none", "none"]),
        "is_control": np.asarray([False, False, True, True, False, False, False]),
        "unresolved_action": np.asarray([False, False, False, False, False, False, True]),
    }
    old = MOD.EXPECTED_ROWS
    MOD.EXPECTED_ROWS = {"fit": 1, "control": 1, "reconstruction-held": 2}
    try:
        masks = MOD.role_masks(routing)
    finally:
        MOD.EXPECTED_ROWS = old
    assert {key: np.flatnonzero(value).tolist() for key, value in masks.items()} == {
        "fit": [0], "control": [2], "reconstruction-held": [1, 3]
    }


def test_bounded_row_span_reader_matches_exact_rows(tmp_path: Path) -> None:
    prefix = b"metadata-header"
    matrix = np.arange(8 * 5, dtype=np.float32).reshape(8, 5)
    path = tmp_path / "source.bin"
    path.write_bytes(prefix + matrix.tobytes(order="C"))
    selected = MOD.read_rows_bounded(path, len(prefix), matrix.shape, np.asarray([1, 4, 7]))
    np.testing.assert_array_equal(selected, matrix[[1, 4, 7]])


def test_raw_count_validation_uses_retained_sum_and_reports_obs_difference() -> None:
    counts, library, comparison = MOD.validate_raw_block(
        np.asarray([[0, 2, 3], [1, 0, 0]], np.float32), np.asarray([6, 2], np.float32)
    )
    assert counts.dtype == np.int32
    assert library.tolist() == [5, 1]
    assert comparison["obsGreaterRows"] == 2
    assert comparison["sumObsMinusRetained"] == 2
    try:
        MOD.validate_raw_block(np.asarray([[1.5, 0]], np.float32), np.asarray([2]))
    except ValueError:
        pass
    else:
        raise AssertionError("fractional raw count accepted")
