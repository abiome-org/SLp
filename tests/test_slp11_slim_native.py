import importlib.util
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-slim-baseline-v1/slim_native.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tested_slim_native", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adaptation_matches_official_slim_pca_and_closed_form():
    module = load_module()
    rng = np.random.default_rng(731)
    features = rng.normal(size=(24, 7))
    targets = rng.normal(size=(24, 13))
    rank, ridge = 5, 0.1
    model = module.fit(features, targets, rank=rank, lambda_reg=ridge)

    y = targets.T
    bias = y.mean(axis=1, keepdims=True)
    centered = y - bias
    g = PCA(n_components=rank).fit_transform(centered)
    z = np.linalg.solve(g.T @ g + ridge * np.eye(rank), g.T @ centered)
    w = np.linalg.solve(features.T @ features + ridge * np.eye(features.shape[1]), (z @ features).T).T
    expected = (g @ w @ features.T + bias).T

    np.testing.assert_allclose(model.predict_residual(features), expected, rtol=1e-10, atol=1e-10)
    assert model.rank == rank
    assert model.lambda_reg == ridge


def test_predictions_reject_wrong_or_nonfinite_features():
    module = load_module()
    model = module.fit(np.eye(4), np.arange(24, dtype=float).reshape(4, 6), rank=2)
    for invalid in (np.ones((2, 3)), np.full((2, 4), np.nan)):
        try:
            model.predict_residual(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid feature matrix accepted")
