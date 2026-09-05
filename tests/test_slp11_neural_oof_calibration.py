"""Fitting-isolation tests for neural held-gene OOF uncertainty calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_slp11_neural_oof_calibration.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_neural_oof_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def test_global_gene_folds_strictly_exclude_outer_validation() -> None:
    actions = np.asarray(["A", "A", "B", "C", "D", "D", "V", "V"])
    train = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    validation = np.asarray([6, 7], dtype=np.int64)
    fold_ids = np.asarray([0, 0, 1, 2, 1, 1], dtype=np.int64)
    plan = CALIBRATION.make_fold_plan(actions, train, validation, fold_ids)
    for item in plan:
        fitting = item["fittingRows"]
        held = item["heldRows"]
        assert not np.intersect1d(fitting, validation).size
        assert not np.intersect1d(held, validation).size
        assert not set(actions[fitting]) & set(actions[held])
    np.testing.assert_array_equal(
        np.sort(np.concatenate([item["heldRows"] for item in plan])), train
    )


def test_fold_predictor_is_refit_three_times_and_fills_only_training_rows() -> None:
    actions = np.asarray(["A", "A", "B", "C", "D", "D", "V", "V"])
    train = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    validation = np.asarray([6, 7], dtype=np.int64)
    fold_ids = np.asarray([0, 0, 1, 2, 1, 1], dtype=np.int64)
    plan = CALIBRATION.make_fold_plan(actions, train, validation, fold_ids)
    calls: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

    def refit(fitting: np.ndarray, held: np.ndarray, fold: int) -> np.ndarray:
        calls.append((fold, tuple(fitting.tolist()), tuple(held.tolist())))
        assert not np.intersect1d(fitting, validation).size
        return np.full((len(held), 2), fold, dtype=np.float32)

    prediction = CALIBRATION.collect_oof_predictions(8, 2, plan, refit)
    assert len(calls) == 3
    assert len({fitting for _, fitting, _ in calls}) == 3
    assert np.isfinite(prediction[train]).all()
    assert np.isnan(prediction[validation]).all()
