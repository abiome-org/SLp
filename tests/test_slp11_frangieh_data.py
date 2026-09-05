import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix

PATH = Path(__file__).parents[1] / "modules/slp-1-1-world-transition-v1/frangieh_data.py"
SPEC = importlib.util.spec_from_file_location("frangieh_data", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_allowlist_precedes_exact_retained_panel_denominator():
    dense = np.array([[1, 2, 0], [900, 800, 700], [3, 0, 4]], dtype=np.float32)
    matrix = csc_matrix(dense)
    selected = np.array([0, 2])
    calls = []

    def reader(positions):
        calls.append(positions.copy())
        return matrix.data[positions]

    totals = MOD.selected_row_sums_from_csc(matrix.indptr, matrix.indices, reader, selected)
    np.testing.assert_array_equal(totals, [3, 7])
    assert calls


def test_filtered_row_cannot_change_aggregated_targets():
    base = np.array([[1, 2], [9, 9], [3, 4]], dtype=np.float32)
    changed = base.copy()
    changed[1] = [9999, 9999]

    def run(dense):
        matrix = csc_matrix(dense)
        selected = np.array([0, 2])
        reader = lambda positions: matrix.data[positions]
        denom = MOD.selected_row_sums_from_csc(matrix.indptr, matrix.indices, reader, selected)
        return MOD.aggregate_transformed_csc_columns(
            matrix.indptr,
            matrix.indices,
            reader,
            selected,
            denom,
            np.array([0, 0]),
            np.array([0, 1]),
        )

    np.testing.assert_array_equal(run(base), run(changed))


def test_author_matched_isotype_formula_and_clipping():
    got = MOD.matched_isotype_transform([9, 0], [4, 9])
    np.testing.assert_allclose(got, [np.log(2.0), 0.0])


def test_complete_guide_classifier_rejects_secondary_target_and_truncation():
    got = MOD.classify_complete_guide_rows(
        np.array(["A", "A", "control", "A"]),
        np.array(["A_1;A_2;NO_SITE_3", "A_1;B_1", "NO_SITE_2", "A_1;BROKEN"]),
        {"A": "ENSG00000121410", "B": "ENSG00000204518"},
    )
    np.testing.assert_array_equal(got["allowed"], [True, False, True, False])
    assert got["target_guide_set"][0] == "A_1;A_2"
    assert got["reason"][1] == "multiple-target-genes"
    assert got["reason"][3] == "malformed-or-truncated-guide-provenance"


def test_paired_cell_access_rejects_test_rows_and_shards(tmp_path):
    path = tmp_path / "access.npz"
    fields = {
        "source_row_index": np.array([2, 8]),
        "cell_ids": np.array(["a", "b"]),
        "action_ids": np.array(["ENSG00000121410", ""]),
        "split": np.array(["train", "control"]),
        "context_ids": np.array(["Control", "IFNγ"]),
        "full_guide_ids": np.array(["A_1", "NO_SITE_1"]),
        "target_guide_sets": np.array(["A_1", ""]),
        "rna_denominator": np.array([3.0, 4.0], dtype=np.float32),
    }
    np.savez(path, **fields)
    loaded = MOD.load_paired_cell_access(path)
    shards = list(MOD.iter_paired_cell_shards(loaded, shard_size=1))
    assert [shard["cell_ids"].item() for shard in shards] == ["a", "b"]
    fields["split"] = np.array(["test", "control"])
    np.savez(path, **fields)
    with np.testing.assert_raises(ValueError):
        MOD.load_paired_cell_access(path)
