import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).parents[1]
    / "modules/slp-1-1-world-transition-v1/four_context_data.py"
)
SPEC = importlib.util.spec_from_file_location("four_context_data_test", PATH)
DATA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATA)


def _gene_for(role: str) -> str:
    for value in range(10000):
        gene = f"ENSG{value:011d}"
        if DATA.split_name(gene) == role:
            return gene
    raise AssertionError(role)


def test_split_is_global_and_test_fails_closed_from_development():
    train = _gene_for("train")
    validation = _gene_for("validation")
    test = _gene_for("test")
    assert DATA.split_name(train) == "train"
    assert DATA.split_name(validation) == "validation"
    assert DATA.split_name(test) == "test"
    assert len({train, validation, test}) == 3


def test_replogle_audit_requires_exact_prefixes_and_controls():
    replogle = {
        "action_ids": np.array(["ENSG00000000001"]),
        "targets": np.array([[1.0]], dtype=np.float32),
        "observed": np.array([[True]]),
        "context_index": np.array([0], dtype=np.int64),
        "record_ids": np.array(["r"]),
        "num_cells_filtered": np.array([2.0], dtype=np.float32),
        "query_ids": np.array(["ENSG00000000002"]),
        "context_ids": np.array(["c0", "c1", "c2"]),
        "basal_control": np.zeros((3, 1), dtype=np.float32),
        "context_basal_expression": np.zeros((3, 1), dtype=np.float32),
        "context_basal_observed": np.ones((3, 1), dtype=bool),
        "context_value_space": np.asarray("context"),
        "num_cells_role": np.asarray("precision-only"),
        "target_value_space": np.asarray("replogle-space"),
        "control_targets": np.zeros((1, 1), dtype=np.float32),
        "control_observed": np.ones((1, 1), dtype=bool),
        "control_context_index": np.array([0]),
        "control_num_cells_filtered": np.array([2.0], dtype=np.float32),
        "control_record_ids": np.array(["control"]),
        "control_core": np.array([True]),
        "split_train": np.array([0]),
        "split_validation": np.array([], dtype=np.int64),
    }
    merged = {name: value.copy() for name, value in replogle.items()}
    merged["action_ids"] = np.array(["ENSG00000000001", "ENSG00000000003"])
    merged["targets"] = np.array([[1.0], [2.0]], dtype=np.float32)
    merged["observed"] = np.ones((2, 1), dtype=bool)
    merged["context_index"] = np.array([0, 3])
    merged["record_ids"] = np.array(["r", "h"])
    merged["num_cells_filtered"] = np.array([2.0, 3.0], dtype=np.float32)
    merged["context_ids"] = np.array(["c0", "c1", "c2", "c3"])
    merged["basal_control"] = np.zeros((4, 1), dtype=np.float32)
    merged["context_basal_expression"] = np.zeros((4, 1), dtype=np.float32)
    merged["context_basal_observed"] = np.ones((4, 1), dtype=bool)
    assert DATA.audit_replogle_preservation(replogle, merged)[
        "replogle_control_arrays_exact"
    ]
    merged["targets"][0, 0] = 9.0
    with pytest.raises(DATA.FourContextDataError, match="targets prefix"):
        DATA.audit_replogle_preservation(replogle, merged)


def test_pickle_free_writer_is_deterministic(tmp_path: Path):
    arrays = {
        "b": np.array(["x"]),
        "a": np.array([[1.0]], dtype=np.float32),
        "scalar": np.asarray("label"),
    }
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    DATA.write_npz(first, arrays)
    DATA.write_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as loaded:
        assert loaded.files == ["a", "b", "scalar"]
        assert loaded["scalar"].shape == ()


def test_array_fingerprint_contract_in_verifier():
    path = Path(__file__).parents[1] / "scripts/verify_slp11_human_four_context.py"
    spec = importlib.util.spec_from_file_location("four_context_verify_test", path)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    value = np.array([[1, 2]], dtype=np.int64)
    assert verifier.array_sha256(value) == verifier.array_sha256(value.copy())
    assert verifier.array_sha256(value) != verifier.array_sha256(value.astype(np.int32))
    assert verifier.array_sha256(value) != verifier.array_sha256(value.reshape(2, 1))
