import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "modules/slp-1-1-molecular-mean-objective-v1/molecular_mean_objective.py"
SPEC = importlib.util.spec_from_file_location("molecular_mean_objective_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_expected_rates_are_mixed_before_log():
    rates = torch.tensor([[[0., 3.], [8., 15.]]], dtype=torch.float64)
    weights = torch.tensor([[1., 3.]], dtype=torch.float64)
    actual = MODULE.population_log1p_mean(rates, weights)
    expected = torch.log1p(torch.tensor([[6., 12.]], dtype=torch.float64))
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, (rates.log1p() * .25 * weights[..., None]).sum(1))
    torch.testing.assert_close(actual, MODULE.population_log1p_mean(rates.flip(1), weights.flip(1)))


def test_gradient_matches_independent_analytic_derivative():
    rates = torch.tensor([[[2., 3.], [8., 15.]], [[1., 2.], [6., 7.]]],
                         dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[1., 3.], [2., 3.]], dtype=torch.float64)
    target = torch.tensor([[.2, .7], [.3, .1]], dtype=torch.float64)
    prediction = MODULE.population_log1p_mean(rates, weights)
    MODULE.normalized_profile_mse(prediction, target, .4).backward()
    w = weights / weights.sum(1, keepdim=True)
    mixed = (rates.detach() * w[..., None]).sum(1)
    expected = (2 * (mixed.log1p() - target) / (4 * .4 * (1 + mixed)))[:, None] * w[..., None]
    torch.testing.assert_close(rates.grad, expected, atol=1e-14, rtol=1e-14)


def test_identical_context_split_does_not_change_endpoint():
    rates = torch.tensor([[[2., 5.]]])
    original = MODULE.population_log1p_mean(rates, torch.ones(1, 1))
    split = MODULE.population_log1p_mean(rates.repeat(1, 2, 1), torch.tensor([[.3, .7]]))
    torch.testing.assert_close(split, original)


def test_invalid_support_and_nonfinite_targets_rejected():
    with pytest.raises(ValueError, match="context support"):
        MODULE.population_log1p_mean(torch.ones(1, 2, 3), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        MODULE.population_log1p_mean(-torch.ones(1, 2, 3), torch.ones(1, 2))
    for scale in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="fitting scale"):
            MODULE.normalized_profile_mse(torch.ones(2, 3), torch.ones(2, 3), scale)
    with pytest.raises(ValueError, match="finite"):
        MODULE.normalized_profile_mse(torch.ones(2, 3), torch.full((2, 3), float("nan")), 1.)
