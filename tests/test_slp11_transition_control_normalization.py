from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from control_normalization import (
    BASAL_VALUE_SPACE,
    VALUE_SPACE,
    ControlNormalizationError,
    control_basal_expression,
    fit_control_normalizer,
)


def test_matches_pinned_linear_umi_per_gem_control_formula() -> None:
    controls = np.asarray(
        [
            [10, 2],
            [25, 6],
            [45, 10],
            [4, 12],
            [10, 16],
            [18, 20],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([20, 40, 60, 20, 40, 60], dtype=np.float64)
    groups = np.asarray([1, 1, 1, 2, 2, 2])
    model = fit_control_normalizer(controls, depth, groups)

    expected_scaled = controls * (40.0 / depth)[:, None]
    for index, group in enumerate([1, 2]):
        local = expected_scaled[groups == group]
        np.testing.assert_allclose(model.control_mean_[index], local.mean(axis=0))
        np.testing.assert_allclose(model.control_std_[index], local.std(axis=0, ddof=1))
    assert model.target_umi_ == 40.0
    assert model.value_space == VALUE_SPACE
    assert model.author_endpoint_equivalent is False

    cells = np.asarray([[70, 1], [1, 60]], dtype=np.float64)
    cell_depth = np.asarray([100, 100], dtype=np.float64)
    cell_groups = np.asarray([1, 2])
    values, observed = model.transform(cells, cell_depth, cell_groups)
    manual = (
        cells * (40.0 / cell_depth)[:, None]
        - model.control_mean_[np.asarray([0, 1])]
    ) / model.control_std_[np.asarray([0, 1])]
    np.testing.assert_allclose(values, manual)
    assert observed.all()
    assert np.max(np.abs(values)) > 3.0  # no clipping


def test_full_library_umi_is_used_instead_of_selected_panel_sum() -> None:
    controls = np.asarray([[5, 5], [10, 10]], dtype=np.float64)
    full_depth = np.asarray([100, 200], dtype=np.float64)
    groups = np.asarray([7, 7])
    model = fit_control_normalizer(controls, full_depth, groups)

    # The two controls have identical composition after scaling by the full
    # library depth, even though selected-panel sums are only 10 and 20.
    np.testing.assert_allclose(model.control_mean_[0], [7.5, 7.5])
    assert not model.control_observed_[0].any()

    basal = control_basal_expression(controls, full_depth)
    expected = np.log2(1.0 + 10_000.0 * controls / full_depth[:, None]).mean(axis=0)
    panel_denominator = np.log2(
        1.0 + 10_000.0 * controls / controls.sum(axis=1)[:, None]
    ).mean(axis=0)
    np.testing.assert_allclose(basal, expected)
    assert not np.allclose(basal, panel_denominator)
    assert BASAL_VALUE_SPACE.endswith("full-library-v1")


def test_zero_variance_queries_are_masked_and_unseen_gems_are_rejected() -> None:
    controls = np.asarray([[1, 4], [1, 6], [2, 5]], dtype=np.float64)
    depth = np.asarray([10, 10, 10], dtype=np.float64)
    groups = np.asarray([3, 3, 4])
    model = fit_control_normalizer(controls, depth, groups)

    values, observed = model.transform(
        np.asarray([[1, 7], [2, 5]], dtype=np.float64),
        np.asarray([10, 10], dtype=np.float64),
        np.asarray([3, 4]),
    )
    assert not observed[0, 0]
    assert observed[0, 1]
    assert not observed[1].any()  # only one fitting control in GEM 4
    assert np.all(values[~observed] == 0.0)

    with pytest.raises(ControlNormalizationError, match="no fitted controls"):
        model.transform(
            np.asarray([[1, 1]], dtype=np.float64),
            np.asarray([10], dtype=np.float64),
            np.asarray([99]),
        )


def test_raw_count_and_full_depth_contracts_prevent_panel_normalization() -> None:
    controls = np.asarray([[3, 4], [5, 6]], dtype=np.float64)
    groups = np.asarray([1, 1])
    with pytest.raises(ControlNormalizationError, match="cannot exceed"):
        fit_control_normalizer(controls, np.asarray([6, 10]), groups)
    with pytest.raises(ControlNormalizationError, match="raw integer"):
        fit_control_normalizer(controls + 0.5, np.asarray([20, 20]), groups)
    with pytest.raises(ControlNormalizationError, match="one integer"):
        fit_control_normalizer(controls, np.asarray([20, 20]), groups.astype(str))
