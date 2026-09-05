from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


def _load_script():
    module_root = (
        Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
    )
    sys.path.insert(0, str(module_root))
    path = module_root / "norman_data_v2.py"
    spec = importlib.util.spec_from_file_location("slp11_norman_data_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NORMAN = _load_script()


def test_per_cell_normalization_uses_full_source_library_before_mean() -> None:
    query_counts = sparse.csc_matrix(np.array([[5, 5], [0, 5]], dtype=np.float64))
    full_library = np.array([10.0, 100.0])

    normalized = NORMAN._per_cell_log_cp10k(query_counts, full_library).toarray()

    expected = np.log2(1.0 + np.array([[5000.0, 500.0], [0.0, 500.0]]))
    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_allclose(normalized.mean(axis=1), expected.mean(axis=1))


def test_control_standardization_recovers_zero_mean_unit_population_variance() -> None:
    controls = sparse.csc_matrix(
        np.array([[0.0, 1.0, 2.0, 3.0], [2.0, 2.0, 4.0, 4.0]])
    )
    mean, std = NORMAN._control_mean_std(controls)
    standardized = (controls.toarray() - mean[:, None]) / std[:, None]

    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.square(standardized).mean(axis=1), 1.0)


def test_control_partition_is_deterministic_disjoint_and_unequal() -> None:
    barcodes = [f"CELL-{index:04d}" for index in range(210)]
    first = NORMAN._control_partition(barcodes, groups=10)
    second = NORMAN._control_partition(barcodes, groups=10)

    np.testing.assert_array_equal(first, second)
    sizes = np.bincount(first, minlength=10)
    assert sizes.sum() == len(barcodes)
    assert np.all(sizes > 0)
    assert len(set(sizes.tolist())) > 1


def test_condition_means_are_cell_means_with_declared_exposures() -> None:
    cells = sparse.csc_matrix(
        np.array([[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]])
    )
    means, counts = NORMAN._means_by_group(cells, np.array([0, 0, 1]), 2)

    np.testing.assert_array_equal(counts, np.array([2, 1]))
    np.testing.assert_allclose(means, np.array([[2.0, 3.0], [10.0, 20.0]]))


def test_construct_identity_distinguishes_repeated_action_populations() -> None:
    left = NORMAN.Condition("GENE_NegCtrl0__GENE_NegCtrl0_1", ("ENSG1",))
    right = NORMAN.Condition("GENE_NegCtrl0__GENE_NegCtrl0_2", ("ENSG1",))

    assert left != right
    assert left.actions == right.actions
