from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-exposure-objective-v1/exposure_objective.py"
SPEC = importlib.util.spec_from_file_location("_slp11_exposure_objective_test", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_precision_uses_only_variance_context_and_exposure_with_floor() -> None:
    tau = np.asarray([[1.0, 0.0], [4.0, 0.0]])
    sigma = np.asarray([[8.0, 1.0], [18.0, 0.0]])
    actual = MODULE.exposure_precision([2.0, 6.0], np.asarray([0, 1]), tau, sigma, scale_floor=0.5)
    np.testing.assert_allclose(actual, [[1 / 5, 1 / 0.5], [1 / 7, 4.0]])


def test_global_normalization_matches_old_weighted_precision_once() -> None:
    new = np.asarray([[2.0, 4.0], [8.0, 16.0]])
    scale = np.asarray([[1.0, 0.5], [0.5, 0.25]])
    context = np.asarray([0, 1])
    observed = np.ones((2, 2), dtype=bool)
    weights = np.asarray([3.0, 1.0])
    result = MODULE.match_global_precision(new, scale, context, observed, weights)
    old_mean = (3 * np.mean([1.0, 4.0]) + np.mean([4.0, 16.0])) / 4
    new_mean = (3 * 3.0 + 12.0) / 4
    assert result.old_weighted_mean_precision == old_mean
    assert result.new_weighted_mean_precision_before == new_mean
    assert result.new_weighted_mean_precision_after == pytest.approx(old_mean)


def test_masked_nonfinite_padding_is_inert_and_weights_are_not_batch_normalized() -> None:
    prediction = torch.tensor([[2.0, 99.0], [4.0, 6.0]])
    target = torch.tensor([[1.0, float("nan")], [2.0, 4.0]])
    observed = torch.tensor([[True, False], [True, True]])
    precision = torch.tensor([[3.0, float("nan")], [2.0, 1.0]])
    weight = torch.tensor([2.0, 4.0])
    loss = MODULE.masked_precision_mse(prediction, target, observed, precision, weight)
    expected = (2.0 * 3.0 + 4.0 * ((4.0 * 2.0 + 4.0 * 1.0) / 2.0)) / 2.0
    assert loss.item() == expected


def test_invalid_exposures_are_rejected() -> None:
    with pytest.raises(MODULE.ExposureObjectiveError):
        MODULE.exposure_precision([0.0], np.asarray([0]), np.ones((1, 1)), np.ones((1, 1)))
