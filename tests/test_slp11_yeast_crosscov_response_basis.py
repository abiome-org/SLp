from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_slp11_yeast_crosscov_response_basis",
    ROOT / "scripts" / "run_slp11_yeast_crosscov_response_basis.py",
)
assert SPEC and SPEC.loader
BASIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASIS)


def test_crosscov_basis_prefers_reproducible_signal_over_independent_noise() -> None:
    rng = np.random.default_rng(731)
    genes = 240
    queries = 80
    signal_basis = np.zeros(queries)
    signal_basis[:5] = 1.0 / np.sqrt(5.0)
    scores = rng.normal(size=(genes, 1))
    shared = 2.0 * scores * signal_basis
    noise_a = rng.normal(scale=1.0, size=(genes, queries))
    noise_b = rng.normal(scale=1.0, size=(genes, queries))
    noise_a[:, 10] *= 7.0
    noise_b[:, 11] *= 7.0
    fitted = BASIS.fit_bases(shared + noise_a, shared + noise_b, rank=2, seed=731)
    cross_alignment = np.max(np.abs(fitted["cross_basis"].T @ signal_basis))
    pca_alignment = np.max(np.abs(fitted["pca_basis"].T @ signal_basis))
    assert cross_alignment > 0.9
    assert cross_alignment > pca_alignment + 0.2


def test_projection_and_trace_metrics_use_only_supplied_fit_statistics() -> None:
    mean = np.asarray([1.0, 2.0, 3.0])
    basis = np.asarray([[1.0], [0.0], [0.0]])
    values = np.asarray([[3.0, 20.0, 30.0]])
    np.testing.assert_array_equal(BASIS.project(values, mean, basis), [[2.0, 0.0, 0.0]])
    metric = BASIS.metrics(
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([[1.0, 2.0, 3.0]]),
        full_trace=14.0,
    )
    assert metric["geneMeanProjectedMse"] == 0.0
    assert metric["geneMeanIndependentQueryCenteredPearson"] == 1.0
    assert metric["fractionOfFullHeldCrossCovarianceTraceCaptured"] == 1.0
