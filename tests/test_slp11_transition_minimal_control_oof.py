from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from minimal_control_oof import (
    MinimalControlOofError,
    advancement_checks,
    collect_oof_predictions,
    make_fold_plan,
    pooled_delta_amplitude,
    select_common_context_tokens,
)


def test_global_fold_plan_keeps_gene_rows_together_and_validation_out() -> None:
    actions = np.asarray(["a", "a", "b", "b", "c", "c", "v", "v"])
    train = np.arange(6, dtype=np.int64)
    validation = np.arange(6, 8, dtype=np.int64)
    plan = make_fold_plan(actions, train, validation, np.asarray([0, 0, 1, 1, 2, 2]))

    assert np.array_equal(np.sort(np.concatenate([item["heldRows"] for item in plan])), train)
    assert all(not np.intersect1d(item["fittingRows"], validation).size for item in plan)
    assert all(not (set(actions[item["fittingRows"]]) & set(actions[item["heldRows"]])) for item in plan)


def test_fold_plan_rejects_gene_crossing_folds() -> None:
    with pytest.raises(MinimalControlOofError, match="crosses OOF folds"):
        make_fold_plan(
            np.asarray(["a", "a", "b", "b", "v"]),
            np.arange(4),
            np.asarray([4]),
            np.asarray([0, 1, 1, 2]),
        )


def test_oof_collection_never_populates_outer_validation() -> None:
    actions = np.asarray(["a", "a", "b", "b", "c", "c", "v"])
    train = np.arange(6)
    plan = make_fold_plan(actions, train, np.asarray([6]), np.asarray([0, 0, 1, 1, 2, 2]))
    result = collect_oof_predictions(
        7, 2, plan, lambda _fitting, held, fold: np.full((held.size, 2), fold)
    )
    assert np.isfinite(result[train]).all()
    assert np.isnan(result[6]).all()


def test_common_control_tokens_use_exact_mask_and_stable_variance_order() -> None:
    basal = np.asarray([[0.0, 2.0, 1.0, 8.0], [0.0, 4.0, 3.0, 2.0]])
    observed = np.asarray([[True, True, True, False], [True, True, True, True]])
    selected, normalized = select_common_context_tokens(
        basal, observed, tokens=2, expected_common=3
    )
    np.testing.assert_array_equal(selected, [1, 2])
    assert normalized.shape == basal.shape
    assert normalized[0, 3] == 0.0


def test_amplitude_and_advancement_are_fixed_and_mean_only() -> None:
    targets = np.asarray([[1.0, 3.0], [3.0, 7.0]])
    oof_mean = np.asarray([[0.0, 1.0], [2.0, 5.0]])
    amplitude = pooled_delta_amplitude(targets, oof_mean, np.ones_like(targets, dtype=np.bool_))
    np.testing.assert_allclose(amplitude, [1.0, 2.0])
    checks = advancement_checks(-1.03, -1.00, -1.01, 0.10)
    assert all(checks.values())
