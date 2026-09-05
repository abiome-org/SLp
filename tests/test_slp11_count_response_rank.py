import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rank_audit_test', ROOT / 'scripts/audit_slp11_count_response_rank.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
CORE = AUDIT.load_module(AUDIT.CORE, 'rank_test_core')


def test_full_rank_matches_unconstrained_ridge():
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(31, 7)), rng.normal(size=(31, 13))
    held = rng.normal(size=(11, 7))
    state = CORE.fit_state(x, y)
    np.testing.assert_allclose(AUDIT.reduced_rank_prediction(CORE, state, held, 7),
                               CORE.predict_residual(state, held, '1000'), atol=1e-14)


def test_rank_one_recovers_rank_one_response_and_intercept():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 5))
    y = (x @ rng.normal(size=(5, 1))) @ rng.normal(size=(1, 9)) + np.arange(9)
    state = CORE.fit_state(x, y)
    np.testing.assert_allclose(AUDIT.reduced_rank_prediction(CORE, state, x, 1),
                               CORE.predict_residual(state, x, '1000'), atol=1e-13)


def test_direct_augmented_design_svd_oracle():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(22, 6)), rng.normal(size=(22, 9))
    state = CORE.fit_state(x, y)
    design = CORE.transform_features(x, state) - state['design_mean']
    augmented = np.vstack((design, np.sqrt(1000) * np.eye(6)))
    targets = np.vstack((y - y.mean(0), np.zeros((6, 9))))
    q, r = np.linalg.qr(augmented)
    u, s, vt = np.linalg.svd(q.T @ targets, full_matrices=False)
    coefficient = np.linalg.solve(r, (u[:, :2] * s[:2]) @ vt[:2])
    oracle = y.mean(0) + design @ coefficient
    np.testing.assert_allclose(AUDIT.reduced_rank_prediction(CORE, state, x, 2), oracle, atol=1e-13)
