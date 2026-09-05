from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-reduced-rank-response-v1/response_model.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MOD = load(PATH, "test_reduced_rank_response")


def independent_reference(features, targets, rank, alpha):
    x = np.asarray(features, np.float32)
    y = np.asarray(targets, np.float64)
    mean = x.mean(0, dtype=np.float64)
    scale = x.std(0, dtype=np.float64)
    scale = np.where(scale > 1e-5, scale, 1)
    design = (x.astype(np.float64) - mean) / scale
    design_mean = design.mean(0)
    xc = design - design_mean
    ym = y.mean(0)
    values, vectors = np.linalg.eigh(xc.T @ xc)
    keep = values > 1e-8
    values, vectors = values[keep], vectors[:, keep]
    rhs = (xc @ vectors).T @ (y - ym)
    root = np.sqrt(values + alpha)
    whitened = rhs / root[:, None]
    _, response = np.linalg.eigh(whitened @ whitened.T)
    response = response[:, -min(rank, len(response)) :]
    latent = ((design - design_mean) @ vectors) / root
    return ym + (latent @ response) @ (response.T @ whitened)


def test_fit_matches_independent_whitened_reduced_rank_formula():
    rng = np.random.default_rng(731)
    x = rng.normal(size=(31, 9)).astype(np.float32)
    y = rng.normal(size=(31, 13))
    model = MOD.fit(x, y, rank=4, alpha=7.0)
    np.testing.assert_allclose(
        model.predict(x), independent_reference(x, y, 4, 7.0), rtol=1e-11, atol=1e-11
    )
    assert np.linalg.matrix_rank(model.predict(x) - model.intercept) <= 4


def test_query_subset_and_save_reload_are_exact(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 6)).astype(np.float32)
    y = rng.normal(size=(20, 8))
    model = MOD.fit(x, y, rank=3, alpha=10)
    path = tmp_path / "model.npz"
    MOD.save(path, model, query_ids=np.asarray([f"q{i}" for i in range(8)]), source_id="s")
    restored = MOD.load(path)
    selected = np.asarray([7, 0, 3])
    np.testing.assert_array_equal(
        restored.predict(x[:5], selected), model.predict(x[:5])[:, selected]
    )


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="positive rank"):
        MOD.fit(np.ones((3, 2)), np.ones((3, 4)), rank=0)
    model = MOD.fit(np.eye(4, dtype=np.float32), np.eye(4), rank=2)
    with pytest.raises(ValueError, match="query_indices"):
        model.predict(np.ones((1, 4)), np.asarray([4]))
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.asarray([[np.nan, 0, 0, 0]]))
