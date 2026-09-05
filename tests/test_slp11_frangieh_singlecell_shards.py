from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_frangieh_singlecell_shards.py"
SPEC = importlib.util.spec_from_file_location("slp11_frangieh_singlecell_shards", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_fitting_selection_excludes_validation_before_values() -> None:
    access = {
        "source_row_index": np.asarray([1, 3, 8, 9]),
        "cell_ids": np.asarray(["a", "b", "c", "d"]),
        "action_ids": np.asarray(["ENSG1", "ENSG2", "", "ENSG3"]),
        "split": np.asarray(["train", "validation", "control", "validation"]),
        "context_ids": np.asarray(["x"] * 4),
        "full_guide_ids": np.asarray(["g"] * 4),
        "target_guide_sets": np.asarray(["t"] * 4),
        "rna_denominator": np.asarray([10, 20, 30, 40], dtype=np.float32),
    }
    selected = MOD.select_fitting_access(access)
    assert selected["source_row_index"].tolist() == [1, 8]
    assert selected["split"].tolist() == ["train", "control"]


def test_reconstruction_split_matches_frozen_hash_formula() -> None:
    cells = np.asarray(["cell-a", "cell-b", "cell-c"])
    actual = MOD.reconstruction_split(cells)
    expected = []
    for cell in cells:
        digest = hashlib.sha256(f"slp11-cell-state-v1|731|{cell}".encode()).digest()
        expected.append("train" if int.from_bytes(digest[:8], "big") % 100 < 90 else "validation")
    assert actual.tolist() == expected


def test_rna_and_adt_normalization() -> None:
    values = MOD.transform_rna_values(np.asarray([0, 2, 8]), np.asarray([10, 10, 20]))
    expected = np.log1p(10_000 * np.asarray([0, 2, 8]) / np.asarray([10, 10, 20]))
    np.testing.assert_allclose(values, expected.astype(np.float32), rtol=0, atol=0)
    adt = MOD.matched_isotype_transform(np.asarray([0, 9, 1]), np.asarray([0, 1, 9]))
    np.testing.assert_allclose(adt, np.asarray([0, np.log(5), 0], dtype=np.float32), rtol=0, atol=1e-7)


def test_deterministic_shard_is_pickle_free(tmp_path: Path) -> None:
    arrays = {
        "rna_data": np.asarray([1.0, 2.0], dtype=np.float32),
        "rna_indices": np.asarray([0, 2], dtype=np.int32),
        "cell_ids": np.asarray(["a", "b"]),
    }
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    MOD.write_deterministic_npz(first, arrays)
    MOD.write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive["cell_ids"].tolist() == ["a", "b"]


def test_group_mask_never_selects_validation() -> None:
    access = {
        "split": np.asarray(["train", "validation", "control"]),
        "action_ids": np.asarray(["ENSG1", "ENSG1", ""]),
        "context_ids": np.asarray(["ctx", "ctx", "ctx"]),
        "target_guide_sets": np.asarray(["g", "g", ""]),
    }
    target = {"kind": "target", "action_id": "ENSG1", "context_id": "ctx", "target_guide_set": "g"}
    control = {"kind": "control", "action_id": "", "context_id": "ctx", "target_guide_set": ""}
    assert MOD._group_mask(access, target).tolist() == [True, False, False]
    assert MOD._group_mask(access, control).tolist() == [False, False, True]
