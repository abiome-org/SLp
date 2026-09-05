import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "modules/slp-1-1-batch-ridge-v1/batch_ridge.py"
SPEC = importlib.util.spec_from_file_location("batch_ridge", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matches_augmented_weighted_least_squares_and_chunking():
    rng = np.random.default_rng(731)
    x = rng.normal(size=(29, 4))
    batch = np.arange(29) % 3
    y = rng.normal(size=(29, 5)) + batch[:, None] * 5
    weight = rng.uniform(0.1, 3, 29)
    alpha = 7.0
    stats = MODULE.BatchRidgeStatistics(4, 5)
    for b in range(3):
        for rows in np.array_split(np.flatnonzero(batch == b), 2):
            stats.update(str(b), x[rows], y[rows], weight[rows])
    model = stats.solve(alpha)
    design = np.concatenate([x, np.eye(3)[batch]], axis=1)
    regularizer = np.diag([alpha] * 4 + [0] * 3)
    oracle = np.linalg.solve(
        design.T @ (weight[:, None] * design) + regularizer,
        design.T @ (weight[:, None] * y),
    )
    np.testing.assert_allclose(model["coefficients"], oracle[:4], atol=1e-12)
    np.testing.assert_allclose(model["batch_offsets"], oracle[4:], atol=1e-12)
    for b in range(3):
        rows = batch == b
        np.testing.assert_allclose(MODULE.predict(model, str(b), x[rows]), design[rows] @ oracle)
    with pytest.raises(ValueError, match="intercept"):
        MODULE.predict(model, "unseen", x)


def test_constant_batch_shift_does_not_change_feature_effect():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(15, 3)), rng.normal(size=(15, 2))
    first, second = (MODULE.BatchRidgeStatistics(3, 2) for _ in range(2))
    first.update("a", x, y, np.ones(15))
    second.update("a", x, y + [20, -30], np.ones(15))
    a, b = first.solve(5), second.solve(5)
    np.testing.assert_allclose(a["coefficients"], b["coefficients"], atol=1e-12)
    np.testing.assert_allclose(b["batch_offsets"] - a["batch_offsets"], [[20, -30]])


def test_invalid_blocks_do_not_update_statistics():
    stats = MODULE.BatchRidgeStatistics(2, 1)
    with pytest.raises(ValueError):
        stats.update("a", np.ones((2, 2)), np.array([[1], [np.nan]]), np.ones(2))
    assert not stats.batches
    assert np.all(stats.xx == 0)
