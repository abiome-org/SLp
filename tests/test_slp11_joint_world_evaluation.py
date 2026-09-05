import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_slp11_joint_world.py"
SPEC = importlib.util.spec_from_file_location("slp11_joint_world_evaluation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_response_metrics_match_independent_formula():
    truth = np.array([[1., 3., 2.], [2., 2., 5.], [4., 1., 3.]])
    prediction = np.array([[1., 3., 2.], [3., 1., 4.], [3., 2., 4.]])
    anchor = np.array([[.5, .5, .5]] * 3)
    result = MODULE.response_metrics(truth, prediction, anchor)
    t = truth - truth[:1]
    p = prediction - prediction[:1]
    t -= t.mean(0)
    p -= p.mean(0)
    expected = []
    for left, right in zip(t, p):
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator > 1e-12:
            expected.append(left @ right / denominator)
    assert result["geneProfileMse"] == np.square(truth - prediction).mean()
    np.testing.assert_allclose(result["independentlyQueryCenteredResidualPearson"], np.mean(expected))
    assert result["finiteCorrelationGenes"] == len(expected)


def test_composition_metrics_use_observed_additive_residual():
    additive = np.array([[1., 2., 3.], [2., 3., 5.]])
    truth = additive + np.array([[1., -1., 0.], [0., 2., -2.]])
    prediction = additive + 2 * (truth - additive)
    result = MODULE.composition_metrics(truth, prediction, additive)
    np.testing.assert_allclose(result["nonadditivePearson"], 1.0)
    assert result["finiteNonadditiveRows"] == 2
    assert result["mse"] == np.square(prediction - truth).mean()


def test_single_action_padding_has_one_active_slot():
    features = np.arange(12, dtype=np.float32).reshape(2, 6)
    actions, mask = MODULE._single_action(features)
    assert actions.shape == (2, 2, 6)
    np.testing.assert_array_equal(actions[:, 0], features)
    assert np.count_nonzero(actions[:, 1]) == 0
    np.testing.assert_array_equal(mask, [[True, False], [True, False]])
