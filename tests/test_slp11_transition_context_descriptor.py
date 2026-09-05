from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_descriptor import (
    FIXED_PANEL_VALUE_SPACE,
    VALUE_SPACE,
    ContextDescriptorError,
    pooled_control_fixed_panel_log2_cp10k,
    pooled_control_log2_cp10k,
)


def test_aggregate_means_equal_the_same_pooled_single_cell_statistic() -> None:
    cells = np.asarray(
        [[4, 1, 0], [6, 3, 2], [2, 8, 1], [4, 4, 3]], dtype=np.float64
    )
    full_umi = np.asarray([10, 15, 20, 25], dtype=np.float64)
    cell_descriptor, cell_observed = pooled_control_log2_cp10k(
        cells,
        full_umi,
        np.ones(4),
    )

    populations = np.asarray([0, 0, 1, 1])
    aggregate_means = np.stack(
        [cells[populations == group].mean(axis=0) for group in [0, 1]]
    )
    aggregate_full_mean = np.asarray(
        [full_umi[populations == group].mean() for group in [0, 1]]
    )
    aggregate_n = np.asarray([2, 2])
    aggregate_descriptor, aggregate_observed = pooled_control_log2_cp10k(
        aggregate_means,
        aggregate_full_mean,
        aggregate_n,
    )

    np.testing.assert_allclose(aggregate_descriptor, cell_descriptor, atol=1e-12)
    np.testing.assert_array_equal(aggregate_observed, cell_observed)
    assert VALUE_SPACE.endswith("full-library-v1")


def test_full_umi_denominator_is_not_selected_panel_sum() -> None:
    means = np.asarray([[4, 1], [6, 3]], dtype=np.float64)
    full_mean = np.asarray([20, 40], dtype=np.float64)
    weights = np.asarray([2, 3])
    descriptor, observed = pooled_control_log2_cp10k(means, full_mean, weights)
    expected = np.log2(
        1.0
        + 10_000.0
        * (means * weights[:, None]).sum(axis=0)
        / np.sum(full_mean * weights)
    )
    panel_denominator = np.log2(
        1.0
        + 10_000.0
        * (means * weights[:, None]).sum(axis=0)
        / np.sum(means.sum(axis=1) * weights)
    )
    np.testing.assert_allclose(descriptor, expected)
    assert observed.all()
    assert not np.allclose(descriptor, panel_denominator)


def test_missing_queries_have_explicit_support_and_no_zero_imputation() -> None:
    means = np.asarray([[2.0, np.nan], [4.0, np.nan]])
    mask = np.asarray([[True, False], [True, False]])
    descriptor, observed = pooled_control_log2_cp10k(
        means,
        np.asarray([10, 20]),
        np.asarray([1, 2]),
        mask,
    )
    assert observed.tolist() == [True, False]
    assert descriptor[1] == 0.0


def test_invalid_cell_weights_and_denominators_are_rejected() -> None:
    means = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ContextDescriptorError, match="integer cell counts"):
        pooled_control_log2_cp10k(means, np.asarray([10, 10]), np.asarray([1, 1.5]))
    with pytest.raises(ContextDescriptorError, match="must be positive"):
        pooled_control_log2_cp10k(means, np.asarray([10, 0]), np.asarray([1, 1]))


def test_fixed_panel_excludes_other_tokens_and_matches_cell_pooling() -> None:
    cells = np.asarray(
        [[3, 7, 100], [5, 5, 200], [2, 8, 300], [6, 4, 400]], dtype=np.float64
    )
    panel = np.asarray([True, True, False])
    by_cell, cell_mask = pooled_control_fixed_panel_log2_cp10k(
        cells,
        np.ones(4),
        panel,
    )
    aggregate_means = np.stack([cells[:2].mean(0), cells[2:].mean(0)])
    by_aggregate, aggregate_mask = pooled_control_fixed_panel_log2_cp10k(
        aggregate_means,
        np.asarray([2, 2]),
        panel,
    )
    np.testing.assert_allclose(by_aggregate, by_cell, atol=1e-12)
    assert cell_mask.tolist() == [True, True, False]
    np.testing.assert_array_equal(aggregate_mask, cell_mask)
    assert by_cell[2] == 0.0
    assert FIXED_PANEL_VALUE_SPACE.endswith("fixed-shared-panel-v1")
