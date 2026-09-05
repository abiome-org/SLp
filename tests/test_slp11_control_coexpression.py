import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules" / "slp-1-1-control-coexpression-v1" / "control_coexpression.py"
SPEC = importlib.util.spec_from_file_location("slp11_control_coexpression_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_library_regression_matches_dense_lstsq_and_removes_affine_signal():
    rng = np.random.default_rng(4)
    group = np.repeat(np.arange(3), [9, 11, 8])
    log_library = rng.normal(size=len(group))
    intercept = rng.normal(size=(3, 5))
    slope = rng.normal(size=(3, 5))
    noise = rng.normal(scale=0.03, size=(len(group), 5))
    x = intercept[group] + (log_library - np.array([log_library[group == g].mean() for g in range(3)])[group])[:, None] * slope[group] + noise
    stats = MODULE.empty_first_pass(3, 5)
    MODULE.update_first_pass(stats, x[:13], log_library[:13], group[:13])
    MODULE.update_first_pass(stats, x[13:], log_library[13:], group[13:])
    mean_x, beta, mean_l = MODULE.regression_parameters(stats)
    residual = MODULE.residualize(x, log_library, group, mean_x, beta, mean_l)
    for g in range(3):
        take = group == g
        design = np.column_stack([np.ones(take.sum()), log_library[take] - log_library[take].mean()])
        expected = x[take] - design @ np.linalg.lstsq(design, x[take], rcond=None)[0]
        np.testing.assert_allclose(residual[take], expected, atol=2e-14)
        np.testing.assert_allclose(residual[take].sum(axis=0), 0.0, atol=2e-14)


def test_leave_self_formula_matches_explicit_dense_correlations():
    rng = np.random.default_rng(7)
    residual = rng.normal(size=(31, 6))
    residual -= residual.mean(axis=0)
    common = np.array([0, 2, 4, 5])
    weights = rng.normal(size=(4, 3))
    native_weights = np.zeros((6, 3))
    native_weights[common] = weights
    moments = MODULE.empty_second_pass(6, 3)
    MODULE.update_second_pass(moments, residual, common, weights)
    feature, present, _ = MODULE.fingerprints_from_moments(
        moments["cross"], moments["var_x"], moments["var_z"], native_weights
    )
    z = residual[:, common] @ weights
    expected = np.zeros_like(feature)
    for q in range(6):
        leave = z - residual[:, q, None] * native_weights[q]
        for d in range(3):
            expected[q, d] = np.corrcoef(residual[:, q], leave[:, d])[0, 1]
    assert present.all()
    np.testing.assert_allclose(feature, expected, atol=2e-14)


def test_native_only_zero_weight_reduces_to_direct_correlation():
    rng = np.random.default_rng(11)
    residual = rng.normal(size=(25, 5))
    residual -= residual.mean(axis=0)
    common = np.array([0, 1, 2, 3])
    weights = rng.normal(size=(4, 2))
    native_weights = np.zeros((5, 2))
    native_weights[common] = weights
    moments = MODULE.empty_second_pass(5, 2)
    MODULE.update_second_pass(moments, residual, common, weights)
    feature, present, _ = MODULE.fingerprints_from_moments(
        moments["cross"], moments["var_x"], moments["var_z"], native_weights
    )
    z = residual[:, common] @ weights
    direct = [np.corrcoef(residual[:, 4], z[:, d])[0, 1] for d in range(2)]
    assert present[4].all()
    np.testing.assert_allclose(feature[4], direct, atol=2e-14)


def test_anchor_and_barcode_hash_are_deterministic():
    x = np.arange(35, dtype=np.float32).reshape(7, 5)
    a, ga = MODULE.fixed_anchor_weights(x, dimensions=3, seed=731)
    b, gb = MODULE.fixed_anchor_weights(x, dimensions=3, seed=731)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ga, gb)
    np.testing.assert_allclose(a.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(a, axis=0), 1.0, atol=1e-15)
    assert MODULE.barcode_half("AAAC", 731) == MODULE.barcode_half("AAAC", 731)
    assert MODULE.barcode_half("AAAC", 731) in (0, 1)
